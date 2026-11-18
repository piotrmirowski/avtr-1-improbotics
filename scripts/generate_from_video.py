# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Render a talking-head video driven by a reference video (Video-to-Video / Face Reenactment).

Extracts per-frame head pose (R) and 3D facial expression deltas (exp) from a
driving video using the LivePortrait motion extractor cascade (SCRFD -> 106/203
landmarks -> motion_extractor) and renders the target avatar via the AVTR-1
renderer pipeline (stitch -> warp -> decoder -> MODNet putback).

Usage:
    pixi run generate_video --driving path/to/driving.mp4
    pixi run generate_video --driving path/to/driving.mp4 --avatar maria --bg minimal_office
    pixi run generate_video --driving path/to/driving.mp4 --portrait path/to/custom.png --bg plain_white
    pixi run generate_video --driving path/to/driving.mp4 --no-relative
    pixi run generate_video --driving path/to/driving.mp4 --lipsync-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import av
import cv2
import imageio_ffmpeg
import numpy as np
import torch

from avtr1_renderer.avatar_loader import Avatar, AvatarLoader, _img_crop_to_bchw256
from avtr1_renderer.avtr1_artifact_manager import (
    find_engine_or_onnx,
    get_artifact_manager,
    get_trt_engine_path,
)
from avtr1_renderer.backgrounds import load_background
from avtr1_renderer.components.face_landmarks import landmark203
from avtr1_renderer.components.liveportrait.motion_extractor import extract_motion
from avtr1_renderer.components.liveportrait.motion_stitch import MotionFrame
from avtr1_renderer.components.source_crop import crop_image
from avtr1_renderer.constants import LIPSYNC_COORDS
from avtr1_renderer.frame_sink import pack_frames
from avtr1_renderer.models.decoder import DecoderInput, DecoderOutput
from avtr1_renderer.models.matting import MODNetInput, MODNetOutput
from avtr1_renderer.models.stitch import StitchInput, StitchOutput
from avtr1_renderer.models.warp import WarpInput, WarpOutput
from avtr1_renderer.pipeline import TRANSPARENT_BG_ID
from avtr1_renderer.renderer import render_chunk, render_chunk_streaming
from avtr1_renderer.runtime import load_engine
from avtr1_renderer.types import KPInfo

_LIPSYNC_INDEX = list(LIPSYNC_COORDS)
_LM203_DSIZE = 224
_MAX_BATCH = 5


def _has_audio_stream(video_path: Path) -> bool:
    try:
        with av.open(str(video_path)) as container:
            return len(container.streams.audio) > 0
    except Exception:
        return False


def _extract_driving_kp(
    frame_rgb: np.ndarray,
    loader: AvatarLoader,
    prev_lmk203: np.ndarray | None,
) -> tuple[KPInfo | None, np.ndarray | None]:
    """Extract KPInfo from one RGB frame, using landmark tracking when available."""
    img_f64 = frame_rgb.astype(np.float64)
    lmk203_pts: np.ndarray | None = None

    if prev_lmk203 is not None:
        try:
            crop_224 = crop_image(
                img_f64,
                prev_lmk203,
                dsize=_LM203_DSIZE,
                scale=1.5,
                vx_ratio=0.0,
                vy_ratio=-0.1,
                flag_do_rot=True,
            )
            lmk203_pts = landmark203(crop_224.img_crop, crop_224.M_c2o, lm203=loader.lmk203)
            crop_512 = crop_image(
                img_f64,
                lmk203_pts,
                dsize=512,
                scale=loader.crop_cfg.scale,
                vx_ratio=loader.crop_cfg.vx_ratio,
                vy_ratio=loader.crop_cfg.vy_ratio,
                flag_do_rot=loader.crop_cfg.flag_do_rot,
            )
            img_crop_512 = crop_512.img_crop
        except Exception:
            lmk203_pts = None

    if lmk203_pts is None:
        try:
            img_crop_512, _, lmk203_pts = loader._crop_512(img_f64)
        except RuntimeError:
            return None, None

    rgb_256 = _img_crop_to_bchw256(img_crop_512)
    rgb_256_t = torch.from_numpy(rgb_256).cuda()
    kp_info = extract_motion(rgb_256_t, motion=loader.motion)
    return kp_info, lmk203_pts


def _build_motion_frame(
    driving_kps: list[KPInfo],
    avatar: Avatar,
    *,
    relative: bool,
    lipsync_only: bool,
    lipsync_only_static_head: bool = False,
) -> MotionFrame:
    """Convert extracted driving KPInfo list into a stacked MotionFrame(R, exp).

    ``motion_extractor`` predicts ``exp`` in post-rotation space (``kp @ R + exp``),
    whereas ``_transform_driving`` applies rotation after adding ``x_d_delta``
    (``(kp + x_d_delta) @ R_new``). Converting ``exp`` into canonical (pre-rotation)
    space via ``exp @ R.T`` ensures ``(kp + exp_in_R) @ R == kp @ R + exp``.
    """
    n = len(driving_kps)
    R_s = avatar.kp_info.R  # (1, 3, 3)
    exp_s_in_R = avatar.kp_info.exp @ R_s.transpose(-1, -2)  # (1, 21, 3)

    R_d = torch.cat([kp.R for kp in driving_kps], dim=0)  # (N, 3, 3)
    exp_d = torch.cat([kp.exp for kp in driving_kps], dim=0)  # (N, 21, 3)
    exp_d_in_R = exp_d @ R_d.transpose(-1, -2)  # (N, 21, 3)

    if relative:
        R_d0 = R_d[0:1]  # (1, 3, 3)
        exp_d0_in_R = exp_d_in_R[0:1]  # (1, 21, 3)
        R_out = (R_d @ R_d0.transpose(-1, -2)) @ R_s
        exp_delta = exp_s_in_R + (exp_d_in_R - exp_d0_in_R)
    else:
        R_out = R_d
        exp_delta = exp_d_in_R

    if lipsync_only_static_head:
        R_out = R_s.expand(n, -1, -1)

    if lipsync_only or lipsync_only_static_head:
        full = exp_s_in_R.reshape(1, 63).expand(n, -1).clone()
        full[:, _LIPSYNC_INDEX] = exp_delta.reshape(n, 63)[:, _LIPSYNC_INDEX]
        exp_delta = full.reshape(n, 21, 3)

    return MotionFrame(
        R=R_out.contiguous(),
        exp=exp_delta.contiguous(),
        lock_static_kps=lipsync_only_static_head,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--driving",
        type=Path,
        required=True,
        help="Path to the driving video file (e.g. .mp4, .mov, .avi)",
    )
    parser.add_argument(
        "--avatar",
        default="maria",
        help="Avatar ID from reference_frames (ignored if --portrait is set)",
    )
    parser.add_argument(
        "--portrait",
        type=Path,
        default=None,
        help="Optional path to a custom portrait PNG/JPG image",
    )
    parser.add_argument(
        "--bg",
        default="plain_white",
        help="Background ID (must match a file in backgrounds artifact, default: 'plain_white')",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("demo_video_driven.mp4"),
        help="Output video path (default: demo_video_driven.mp4)",
    )
    parser.add_argument(
        "--relative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply motion relative to the first driving frame (default: on). Use --no-relative for absolute transfer.",
    )
    parser.add_argument(
        "--lipsync-only",
        action="store_true",
        help="Transfer only lips/eyelids/eyebrows expression coordinates + head rotation.",
    )
    parser.add_argument(
        "--lipsync-only-static-head",
        action="store_true",
        help="Move only lips, eyelids, and eyebrows while keeping the head rotation, nose, and neck completely fixed.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of frames to process.",
    )
    parser.add_argument(
        "--no-mux",
        dest="mux",
        action="store_false",
        help="Don't mux the driving video's audio track into the output video.",
    )
    parser.add_argument(
        "--stream-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Yield each frame as soon as it's ready (default: on).",
    )
    args = parser.parse_args()

    if not args.driving.is_file():
        parser.error(f"Driving video not found: {args.driving}")

    out_h, out_w = 720, 1280
    mgr = get_artifact_manager()
    print("Ensuring renderer and avatar artifacts...")
    for name in (
        "insightface_det",
        "landmark106",
        "landmark203",
        "appearance_extractor",
        "motion_extractor",
        "pasteback_mask",
        "backgrounds",
        "reference_frames",
    ):
        mgr.get_artifact_path(name)

    # Load background
    if args.bg == TRANSPARENT_BG_ID:
        bg = torch.zeros((1, 3, out_h, out_w), dtype=torch.float32, device="cuda")
    else:
        bg_dir = mgr.get_artifact_path("backgrounds")
        bg_path = bg_dir / f"{args.bg}.png"
        if not bg_path.is_file():
            available = sorted(p.stem for p in bg_dir.glob("*.png"))
            parser.error(f"Unknown --bg {args.bg!r}; available: {available}")
        bg = load_background(bg_path, out_h, out_w)

    # Load renderer engines (TRT if built, else ONNX fallback)
    print("Loading renderer engines...")
    decoder = load_engine(
        find_engine_or_onnx("decoder", "decoder_onnx"),
        DecoderInput,
        DecoderOutput,
    )
    warp_trt = get_trt_engine_path("warp_network")
    if warp_trt.is_file():
        plugin_path = mgr.storage_path("warp_plugin")
        use_plugin = (
            plugin_path.is_file()
            and not (sys.platform == "win32" and plugin_path.suffix.lower() == ".so")
        )
        warp = load_engine(
            warp_trt,
            WarpInput,
            WarpOutput,
            plugin_files=[str(plugin_path)] if use_plugin else [],
        )
    else:
        warp_onnx_key = "warp_network_ori_onnx" if sys.platform == "win32" else "warp_network_onnx"
        warp = load_engine(
            find_engine_or_onnx("warp_network", warp_onnx_key),
            WarpInput,
            WarpOutput,
        )
    stitch = load_engine(
        find_engine_or_onnx("stitch_network", "stitch_network_onnx"),
        StitchInput,
        StitchOutput,
    )
    matting = load_engine(
        find_engine_or_onnx("modnet", "modnet_onnx"),
        MODNetInput,
        MODNetOutput,
    )

    loader = AvatarLoader(
        engine_files={
            "insightface_det": mgr.get_artifact_path("insightface_det"),
            "landmark106": mgr.get_artifact_path("landmark106"),
            "landmark203": mgr.get_artifact_path("landmark203"),
            "appearance_extractor": mgr.get_artifact_path("appearance_extractor"),
            "motion_extractor": mgr.get_artifact_path("motion_extractor"),
        },
        mask_template_path=mgr.get_artifact_path("pasteback_mask"),
        out_h=out_h,
        out_w=out_w,
        max_dim=max(out_h, out_w),
    )

    if args.portrait is not None:
        if not args.portrait.is_file():
            parser.error(f"Portrait file not found: {args.portrait}")
        portrait_path = args.portrait
        avatar_id = args.portrait.stem
    else:
        portraits_dir = mgr.get_artifact_path("reference_frames")
        portrait_path = portraits_dir / f"{args.avatar}.png"
        if not portrait_path.is_file():
            available = sorted(p.stem for p in portraits_dir.glob("*.png"))
            parser.error(f"Unknown --avatar {args.avatar!r}; available: {available}")
        avatar_id = args.avatar

    print(f"Loading avatar '{avatar_id}' from {portrait_path}...")
    avatar = loader.load(portrait_path, avatar_id=avatar_id)

    # Read driving video frames and extract motion
    cap = cv2.VideoCapture(str(args.driving))
    if not cap.isOpened():
        parser.error(f"Could not open driving video: {args.driving}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    total_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames is not None and total_est > 0:
        total_est = min(total_est, args.max_frames)

    print(f"Extracting motion from '{args.driving}' (fps={fps:.2f}, est_frames={total_est})...")
    driving_kps: list[KPInfo] = []
    prev_lmk203: np.ndarray | None = None
    last_kp: KPInfo | None = None

    frame_idx = 0
    t_extract0 = time.perf_counter()
    try:
        while True:
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            kp, prev_lmk203 = _extract_driving_kp(frame_rgb, loader, prev_lmk203)
            if kp is None:
                kp = last_kp if last_kp is not None else avatar.kp_info
            else:
                last_kp = kp
            driving_kps.append(kp)
            frame_idx += 1
            if frame_idx % 50 == 0:
                elapsed = time.perf_counter() - t_extract0
                print(f"  extracted {frame_idx} frames ({frame_idx / elapsed:.1f} fps)")
    finally:
        cap.release()

    if not driving_kps:
        raise RuntimeError(f"No frames decoded from {args.driving}")

    n_frames = len(driving_kps)
    print(f"Extracted {n_frames} driving frames in {time.perf_counter() - t_extract0:.1f}s")

    motions = _build_motion_frame(
        driving_kps,
        avatar,
        relative=args.relative,
        lipsync_only=args.lipsync_only,
        lipsync_only_static_head=args.lipsync_only_static_head,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mux_audio = args.mux and _has_audio_stream(args.driving)
    writer = imageio_ffmpeg.write_frames(
        str(args.out),
        size=(out_w, out_h),
        fps=fps,
        codec="libx264",
        pix_fmt_in="yuv420p",
        pix_fmt_out="yuv420p",
        quality=8,
        macro_block_size=1,
        audio_path=str(args.driving) if mux_audio else None,
        audio_codec="aac" if mux_audio else None,
    )
    writer.send(None)

    print(
        f"Rendering {n_frames} frames "
        f"(relative={args.relative}, lipsync_only={args.lipsync_only}, "
        f"lipsync_only_static_head={args.lipsync_only_static_head}, mux_audio={mux_audio})..."
    )
    produced = 0
    t_render0 = time.perf_counter()
    try:
        for start in range(0, n_frames, _MAX_BATCH):
            chunk_motions = motions[start : start + _MAX_BATCH]
            if args.stream_frames:
                stream = render_chunk_streaming(
                    chunk_motions,
                    avatar,
                    bg,
                    stitch=stitch,
                    warp=warp,
                    decoder=decoder,
                    matting=matting,
                )
                for rgb, alpha in stream:
                    frame = pack_frames(rgb, alpha, pixel_format="yuv_i420")[0]
                    writer.send(frame.data.tobytes())
                    produced += 1
            else:
                rgb, alpha = render_chunk(
                    chunk_motions,
                    avatar,
                    bg,
                    stitch=stitch,
                    warp=warp,
                    decoder=decoder,
                    matting=matting,
                )
                for frame in pack_frames(rgb, alpha, pixel_format="yuv_i420"):
                    writer.send(frame.data.tobytes())
                    produced += 1
    finally:
        writer.close()

    elapsed_render = time.perf_counter() - t_render0
    print(
        f"\nDone. {produced} frames ({produced / fps:.1f}s @ {fps:.1f} fps, "
        f"render speed {produced / elapsed_render:.1f} fps) -> {args.out}"
    )


if __name__ == "__main__":
    main()
