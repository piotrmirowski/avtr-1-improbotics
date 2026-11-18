# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Render a talking-head video driven by a reference video or live webcam.

Extracts per-frame head pose (R) and 3D facial expression deltas (exp) from a
driving video or live webcam using the LivePortrait motion extractor cascade
(SCRFD -> 106/203 landmarks -> motion_extractor) and renders the target avatar
via the AVTR-1 renderer pipeline (stitch -> warp -> decoder -> MODNet putback).

Offline video usage:
    pixi run generate_video --driving path/to/driving.mp4
    pixi run generate_video --driving path/to/driving.mp4 --avatar maria --bg minimal_office
    pixi run generate_video --driving path/to/driving.mp4 --portrait path/to/custom.png --bg plain_white
    pixi run generate_video --driving path/to/driving.mp4 --lipsync-only-static-head

Live webcam usage (inspired by Deep-Live-Cam):
    pixi run generate_video --webcam --avatar maria --bg plain_white
    pixi run generate_video --webcam --camera-index 0 --lipsync-only-static-head
    pixi run generate_video --webcam --device "Integrated Webcam" --port 8001
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import av
import cv2
import imageio_ffmpeg
import numpy as np
import torch

from avtr1_renderer.avatar_loader import Avatar, AvatarLoader, _img_crop_to_bchw256
from avtr1_renderer.avtr1_artifact_manager import (
    find_engine_or_onnx,
    get_artifact_manager,
    get_storage_root,
    get_trt_engine_path,
)
from avtr1_renderer.avtr1_motion_generator import AVTR1MotionGenerator, AVTR1State, Normalizer
from avtr1_renderer.backgrounds import load_background
from avtr1_renderer.components.face_landmarks import landmark203
from avtr1_renderer.components.liveportrait.motion_extractor import extract_motion
from avtr1_renderer.components.liveportrait.motion_stitch import MotionFrame
from avtr1_renderer.components.source_crop import crop_image
from avtr1_renderer.constants import LIPSYNC_COORDS
from avtr1_renderer.frame_sink import pack_frames
from avtr1_renderer.models.avtr1 import (
    Avtr1DecodeInput,
    Avtr1DecodeOutput,
    Avtr1EncodeInput,
    Avtr1EncodeOutput,
)
from avtr1_renderer.models.decoder import DecoderEngine, DecoderInput, DecoderOutput
from avtr1_renderer.models.hubert import HubertInput, HubertOutput
from avtr1_renderer.models.matting import MODNetEngine, MODNetInput, MODNetOutput
from avtr1_renderer.models.stitch import StitchEngine, StitchInput, StitchOutput
from avtr1_renderer.models.warp import WarpEngine, WarpInput, WarpOutput
from avtr1_renderer.pipeline import TRANSPARENT_BG_ID
from avtr1_renderer.renderer import render_chunk, render_chunk_streaming
from avtr1_renderer.runtime import load_engine
from avtr1_renderer.types import Chunk, KPInfo

_LIPSYNC_INDEX = list(LIPSYNC_COORDS)
_LM203_DSIZE = 224
_MAX_BATCH = 5
_OUTPUT_FPS = 60


# ---------------------------------------------------------------------------
# Camera & image utilities (adapted from Deep-Live-Cam utils.py)
# ---------------------------------------------------------------------------


def log(msg: str, msg_type: str = "info") -> None:
    print(f"[{msg_type}] {msg}")


def list_webcams(device_name: str) -> int:
    """Enumerate all connected webcams and find the first matching the given device name."""
    ids: list[int] = []
    try:
        from cv2_enumerate_cameras import enumerate_cameras

        for camera_info in enumerate_cameras():
            print(f"{camera_info.index}: {camera_info.name}")
            if device_name and device_name.lower() in camera_info.name.lower():
                ids.append(camera_info.index)
    except ImportError:
        log("cv2_enumerate_cameras not installed; probing camera indices 0..3", "warning")
        for idx in range(4):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"{idx}: Camera {idx}")
                ids.append(idx)
            cap.release()
    return min(ids) if ids else 0


def write_numpy_to_byte_string(image: np.ndarray | None) -> bytes | None:
    """Convert a NumPy BGR image array into a JPEG byte stream."""
    if image is None:
        return None
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    return buf.tobytes()


def _has_audio_stream(video_path: Path) -> bool:
    try:
        with av.open(str(video_path)) as container:
            return len(container.streams.audio) > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Motion extraction & MotionFrame construction
# ---------------------------------------------------------------------------


def _extract_driving_kp(
    frame_rgb: np.ndarray,
    loader: AvatarLoader,
    prev_lmk203: np.ndarray | None = None,
) -> tuple[KPInfo | None, np.ndarray | None]:
    """Extract KPInfo from one RGB frame, verifying a face is detected via SCRFD."""
    _ = prev_lmk203
    img_f64 = frame_rgb.astype(np.float64)
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
    reference_kp: KPInfo | None = None,
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
        if reference_kp is not None:
            R_d0 = reference_kp.R  # (1, 3, 3)
            exp_d0_in_R = reference_kp.exp @ R_d0.transpose(-1, -2)  # (1, 21, 3)
        else:
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


def list_microphones(device_name: str = "") -> int | None:
    """Enumerate audio input devices via sounddevice and return matching device index."""
    try:
        import sounddevice as sd

        matched: list[int] = []
        for idx, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0:
                name = str(dev.get("name", f"Device {idx}"))
                print(f"[mic] {idx}: {name}")
                if device_name and device_name.lower() in name.lower():
                    matched.append(idx)
        return min(matched) if matched else None
    except ImportError:
        log("sounddevice not installed; host mic capture unavailable (browser mic via Web UI still works).", "warning")
        return None
    except Exception as exc:
        log(f"Could not enumerate microphones: {exc}", "warning")
        return None


# ---------------------------------------------------------------------------
# Complementary Live Microphone Audio Driver (AVTR-1 speech-to-motion)
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 16_000


class MicrophoneAudioDriver:
    """Captures 16 kHz mono microphone audio and drives AVTR1MotionGenerator."""

    def __init__(
        self,
        avatar: Avatar,
        *,
        mic_device_name: str = "",
        mic_index: int | None = None,
    ) -> None:
        self.avatar = avatar
        self.mic_device_name = mic_device_name
        self.mic_index = mic_index

        self._motion_generator: AVTR1MotionGenerator | None = None
        self._state: AVTR1State | None = None
        self._pending_frames: list[MotionFrame] = []

        self._lock = threading.Lock()
        # 3200 samples = 200 ms step (5 frames @ 25 fps); 6480 samples = full HuBERT window
        self._step_samples = 3200
        self._window_samples = 6480
        self._ring = np.zeros(self._window_samples, dtype=np.float32)
        self.audio_rms: float = 0.0

        self._sd_stream: Any = None
        self.mic_streaming: bool = False

    def ensure_loaded(self) -> None:
        """Lazily load HuBERT + AVTR-1 encode/decode TRT engines when audio mode is activated."""
        if self._motion_generator is not None:
            return
        encode_path = get_trt_engine_path("avtr1_encode")
        decode_path = get_trt_engine_path("avtr1_decode")
        norm_path = get_storage_root() / "avtr1_normalizer.safetensors"
        if not encode_path.is_file() or not decode_path.is_file() or not norm_path.is_file():
            raise RuntimeError(
                "AVTR-1 audio TRT engines not found. Run `pixi run build-trt-engines-avtr1` "
                "(and `pixi run build-trt-engines-hubert`) first."
            )
        log("Loading AVTR-1 audio-to-motion engines (HuBERT + encode + decode)...", "audio")
        encode = load_engine(encode_path, Avtr1EncodeInput, Avtr1EncodeOutput)
        decode = load_engine(decode_path, Avtr1DecodeInput, Avtr1DecodeOutput)
        normalizer = Normalizer.from_safetensors(norm_path)
        hubert = load_engine(
            find_engine_or_onnx("hubert_lbs", "hubert_onnx"),
            HubertInput,
            HubertOutput,
        )
        self._motion_generator = AVTR1MotionGenerator(
            hubert=hubert,
            encode_engine=encode,
            decode_engine=decode,
            normalizer=normalizer,
        )
        self._step_samples = self._motion_generator.chunk_size * self._motion_generator.frame_len
        self._window_samples = (
            (self._motion_generator.chunk_size + self._motion_generator.future_size)
            * self._motion_generator.frame_len
            + self._motion_generator.audio_shift
        )
        with self._lock:
            self._ring = np.zeros(self._window_samples, dtype=np.float32)
        self._state = self._motion_generator.initial_state(self.avatar)
        log("AVTR-1 audio-to-motion generator ready.", "audio")

    def start_mic(self) -> None:
        """Start host microphone capture via sounddevice (if installed)."""
        if self.mic_streaming:
            return
        try:
            import sounddevice as sd

            dev = (
                self.mic_index
                if self.mic_index is not None
                else list_microphones(self.mic_device_name)
            )

            def _callback(indata: np.ndarray, _frames: int, _time_info: Any, status: Any) -> None:
                if status:
                    pass
                mono = indata[:, 0].astype(np.float32, copy=True)
                self.push_audio_samples(mono)

            self._sd_stream = sd.InputStream(
                samplerate=_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=1600,
                device=dev,
                callback=_callback,
            )
            self._sd_stream.start()
            self.mic_streaming = True
            log(f"Started host microphone stream (device={dev}).", "audio")
        except Exception as exc:
            log(f"Host microphone stream not started ({exc}); you can use Browser Mic in the UI.", "warning")

    def stop_mic(self) -> None:
        if self._sd_stream is not None:
            try:
                self._sd_stream.stop()
                self._sd_stream.close()
            except Exception:
                pass
            self._sd_stream = None
        self.mic_streaming = False

    def push_audio_samples(self, samples_16k: np.ndarray) -> None:
        """Append 16 kHz mono float32 samples into the rolling audio buffer."""
        if samples_16k.size == 0:
            return
        samples_16k = np.asarray(samples_16k, dtype=np.float32).reshape(-1)
        rms = float(np.sqrt(np.mean(samples_16k * samples_16k)))
        self.audio_rms = 0.7 * self.audio_rms + 0.3 * rms
        with self._lock:
            n = len(samples_16k)
            if n >= self._step_samples:
                # Keep the latest _step_samples in the current slot, zero future lookahead for low latency
                self._ring[: self._step_samples] = samples_16k[-self._step_samples :]
                self._ring[self._step_samples :] = 0.0
            else:
                self._ring[: self._step_samples - n] = self._ring[n : self._step_samples]
                self._ring[self._step_samples - n : self._step_samples] = samples_16k
                self._ring[self._step_samples :] = 0.0

    def reset_state(self) -> None:
        if self._motion_generator is not None:
            self._state = self._motion_generator.initial_state(self.avatar)
        self._pending_frames.clear()

    def next_motion_frame(self, *, static_head: bool = False) -> MotionFrame:
        """Return the next single-frame MotionFrame predicted from live microphone audio."""
        self.ensure_loaded()
        assert self._motion_generator is not None
        if self._state is None:
            self._state = self._motion_generator.initial_state(self.avatar)

        if not self._pending_frames:
            with self._lock:
                speech_win = self._ring.copy()
            listen_win = np.zeros_like(speech_win)
            chunk = Chunk(audio_speech=speech_win, audio_listen=listen_win)
            motions_5, self._state = self._motion_generator.generate_chunk(
                chunk, self.avatar, self._state
            )
            self._pending_frames = [motions_5[i] for i in range(len(motions_5))]

        frame_motion = self._pending_frames.pop(0)
        if static_head:
            R_s = self.avatar.kp_info.R
            return MotionFrame(
                R=R_s.clone(),
                exp=frame_motion.exp,
                lock_static_kps=True,
            )
        return frame_motion


# ---------------------------------------------------------------------------
# Live Webcam + Microphone Driver (threaded loop inspired by Deep-Live-Cam)
# ---------------------------------------------------------------------------


def _compute_avatar_crop_boxes(
    avatar: Avatar,
    loader: AvatarLoader,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Compute (full_4_3_box, face_crop_4_3_box) as (x1, y1, x2, y2) on the 1280x720 frame.

    - ``full_4_3_box``: 960x720 (4:3) window centered horizontally on the avatar's face.
    - ``face_crop_4_3_box``: 4:3 window spanning from ~2 cm above the eyebrows to
      ~2 cm below the lower lip (where eyebrow-to-lower-lip on an adult face is ~10 cm,
      so 2 cm is 0.20 * H_brow_to_lip above the eyebrows and below the lower lip).
    """
    from avtr1_renderer.components.face_detection import detect_faces
    from avtr1_renderer.components.face_landmarks import landmark106

    h, w = int(avatar.source.shape[-2]), int(avatar.source.shape[-1])  # 720, 1280
    target_w_full = int(round(h * 4.0 / 3.0))  # 960

    src_rgb = (
        avatar.source.permute(1, 2, 0).detach().cpu().numpy() * 255.0
    ).astype(np.float64)

    try:
        det, _ = detect_faces(src_rgb, det=loader.det)
        order = np.argsort(-(det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]))
        bbox = det[order[0]]
        lmk106_pts = landmark106(src_rgb, bbox, lm106=loader.lmk106)
        _, _, lmk203_pts = loader._crop_512(src_rgb)

        cx = float(0.5 * (lmk203_pts[:, 0].min() + lmk203_pts[:, 0].max()))
        y_eyebrow = float(min(lmk106_pts[:, 1].min(), lmk203_pts[:, 1].min()))
        y_chin = float(max(lmk106_pts[:, 1].max(), lmk203_pts[:, 1].max()))

        # In InsightFace 106-landmark layout, indices 52..71 are the mouth/lips.
        lip_max_y = float(lmk106_pts[52:72, 1].max())
        expected_min = y_eyebrow + 0.55 * (y_chin - y_eyebrow)
        expected_max = y_eyebrow + 0.90 * (y_chin - y_eyebrow)
        if expected_min <= lip_max_y <= expected_max:
            y_lower_lip = lip_max_y
        else:
            y_lower_lip = y_eyebrow + 0.78 * (y_chin - y_eyebrow)

        # Eyebrow-to-lower-lip is ~10 cm on an adult human face -> 2 cm = 0.20 * H_brow_to_lip
        h_brow_to_lip = max(40.0, y_lower_lip - y_eyebrow)
        margin_0_8_cm = 0.08 * h_brow_to_lip
        margin_1_5_cm = 0.15 * h_brow_to_lip

        y1_f = y_eyebrow - margin_0_8_cm
        y2_f = y_lower_lip + margin_1_5_cm
        crop_h = min(float(h), max(80.0, y2_f - y1_f))
        crop_w = min(float(w), crop_h * (4.0 / 3.0))
        crop_h = crop_w * (3.0 / 4.0)

        cy = 0.5 * (y1_f + y2_f)
        fy1 = int(round(np.clip(cy - 0.5 * crop_h, 0, h - crop_h)))
        fy2 = fy1 + int(round(crop_h))
        fx1 = int(round(np.clip(cx - 0.5 * crop_w, 0, w - crop_w)))
        fx2 = fx1 + int(round(crop_w))
        face_box = (fx1, fy1, fx2, fy2)
    except Exception:
        cx = w * 0.5
        face_box = ((w - 480) // 2, (h - 360) // 2, (w + 480) // 2, (h + 360) // 2)

    full_x1 = int(round(np.clip(cx - 0.5 * target_w_full, 0, w - target_w_full)))
    full_x2 = full_x1 + target_w_full
    full_box = (full_x1, 0, full_x2, h)
    return full_box, face_box


def _crop_to_4_3(
    img: np.ndarray,
    box: tuple[int, int, int, int],
    out_w: int,
    out_h: int,
) -> np.ndarray:
    x1, y1, x2, y2 = box
    cropped = img[y1:y2, x1:x2]
    if cropped.shape[1] == out_w and cropped.shape[0] == out_h:
        return np.ascontiguousarray(cropped)
    return cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def _build_cylindrical_remap(
    box: tuple[int, int, int, int],
    out_w: int,
    out_h: int,
    fov_deg: float = 90.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute cv2.remap tables for projecting the planar face crop onto a
    vertical cylinder covering ``fov_deg`` (90 degrees) along the horizontal axis.

    For normalized output coordinates (u_hat, v_hat) in [-1, 1] x [-1, 1]:
        theta = u_hat * (fov_rad / 2)
        x_planar = sin(theta)
        y_planar = v_hat   (unmodified in y)
    This elongates the image horizontally along x to compensate for the 90-deg
    cylindrical curvature while leaving y completely unmodified.
    """
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    half_w = 0.5 * (x2 - x1)
    half_h = 0.5 * (y2 - y1)

    half_fov = np.deg2rad(fov_deg * 0.5)

    u_hat = np.linspace(-1.0, 1.0, out_w, dtype=np.float32)  # (out_w,)
    v_hat = np.linspace(-1.0, 1.0, out_h, dtype=np.float32)  # (out_h,)

    theta = u_hat * float(half_fov)  # (out_w,) in [-pi/4, +pi/4]
    x_src_norm = np.sin(theta)  # (out_w,) in [-sqrt(2)/2, +sqrt(2)/2]

    # Broadcast to (out_h, out_w): elongated along x, unmodified in y
    map_x = (cx + x_src_norm[None, :] * half_w).repeat(out_h, axis=0).astype(np.float32)
    map_y = (cy + v_hat[:, None] * half_h).repeat(out_w, axis=1).astype(np.float32)
    return map_x, map_y


class WebcamAvatarDriver:
    """Captures webcam frames and drives the AVTR-1 renderer in a background thread."""

    def __init__(
        self,
        *,
        avatar: Avatar,
        bg: torch.Tensor,
        loader: AvatarLoader,
        stitch: StitchEngine,
        warp: WarpEngine,
        decoder: DecoderEngine,
        matting: MODNetEngine,
        device_name: str = "Integrated Webcam",
        camera_index: int | None = None,
        camera_width: int = 960,
        camera_height: int = 540,
        relative: bool = True,
        lipsync_only: bool = False,
        lipsync_only_static_head: bool = False,
        crop_to_face: bool = False,
        mic_audio_mode: bool = False,
        mic_device_name: str = "",
        mic_index: int | None = None,
        use_virtual_cam: bool = True,
        record_out: Path | None = None,
    ) -> None:
        self.avatar = avatar
        self._default_avatar = avatar
        self._avatar_lock = threading.Lock()
        self.bg = bg
        self.loader = loader
        self.stitch = stitch
        self.warp = warp
        self.decoder = decoder
        self.matting = matting

        self.relative = relative
        self.lipsync_only = lipsync_only
        self.lipsync_only_static_head = lipsync_only_static_head
        self.crop_to_face = crop_to_face
        self.mic_audio_mode = mic_audio_mode
        self.hybrid_webcam_mic = False
        self.use_virtual_cam = use_virtual_cam
        self.record_out = record_out

        self._last_timestamp_stream = 0.0
        # Output stream is 4:3 (720x540, matching Deep-Live-Cam so browser stream views don't overflow)
        self._render_height, self._render_width = int(avatar.source.shape[-2]), int(avatar.source.shape[-1])
        self._output_width = 720
        self._output_height = 540
        self._camera_width = camera_width
        self._camera_height = camera_height

        self.source_image: dict[str, Any] = {"image": None, "byte_string": None, "timestamp": 0.0}
        self._latest_raw_camera_bgr: np.ndarray | None = None

        self.mic_driver = MicrophoneAudioDriver(
            avatar,
            mic_device_name=mic_device_name,
            mic_index=mic_index,
        )
        self._set_active_avatar(avatar)
        if self.mic_audio_mode:
            self.mic_driver.ensure_loaded()
            self.mic_driver.start_mic()

        if camera_index is not None:
            list_webcams(device_name)
            self._device = camera_index
            log(f"Using camera index {self._device} specified by --camera-index", "info")
        else:
            self._device = list_webcams(device_name)

        self.current_camera: dict[str, Any] = {"image": None, "byte_string": None, "timestamp": 0.0}
        self.current_deepfake: dict[str, Any] = {
            "image": None,
            "byte_string": None,
            "timestamp": 0.0,
            "active": True,
        }

        self.camera_streaming = True
        self._running = True
        self._prev_lmk203: np.ndarray | None = None
        self._reference_kp: KPInfo | None = None
        self._last_kp: KPInfo | None = None
        self._missing_face_frames = 0

        self._cap: cv2.VideoCapture | None = cv2.VideoCapture(self._device)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._camera_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._camera_height)
        self._cap.set(cv2.CAP_PROP_FPS, _OUTPUT_FPS)
        print(f"[camera] Camera device index: {self._device}")
        print(f"[camera] Camera Resolution: {self._camera_width}x{self._camera_height}")
        print(f"[camera] Output Resolution: {self._output_width}x{self._output_height}")

        self._thread: threading.Thread | None = None
        self.start()

    def ping_stream(self) -> None:
        self._last_timestamp_stream = time.time()

    def _set_active_avatar(self, new_avatar: Avatar) -> None:
        """Install an Avatar in memory, recompute 4:3 crop/remap tables, and update source_image."""
        full_box, face_box = _compute_avatar_crop_boxes(new_avatar, self.loader)
        cyl_x, cyl_y = _build_cylindrical_remap(
            face_box,
            self._output_width,
            self._output_height,
            fov_deg=90.0,
        )
        src_rgb = (
            (new_avatar.source.clamp(0.0, 1.0) * 255.0)
            .to(torch.uint8)
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
        )
        src_4_3_rgb = _crop_to_4_3(src_rgb, full_box, self._output_width, self._output_height)
        src_4_3_bgr = cv2.cvtColor(src_4_3_rgb, cv2.COLOR_RGB2BGR)

        with self._avatar_lock:
            self.avatar = new_avatar
            self._full_4_3_box = full_box
            self._face_crop_4_3_box = face_box
            self._cyl_map_x = cyl_x
            self._cyl_map_y = cyl_y
            self.mic_driver.avatar = new_avatar
            self.mic_driver.reset_state()
            self._reference_kp = None
            self._last_kp = None
            self._missing_face_frames = 0

        self.source_image["image"] = src_4_3_bgr
        self.source_image["byte_string"] = write_numpy_to_byte_string(src_4_3_bgr)
        self.source_image["timestamp"] = time.time()

    def capture_source_image_from_camera(self) -> float:
        """Capture the current webcam frame in memory and build a new target Avatar (no disk I/O)."""
        raw_bgr = self._latest_raw_camera_bgr
        if raw_bgr is None:
            raise RuntimeError("No camera frame available in memory yet.")

        log("Capturing camera frame in memory to replace target avatar...", "source")
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        ch, cw = raw_rgb.shape[:2]
        # Scale proportionally to fit _render_height (720) and center inside _render_width (1280)
        scale = float(self._render_height) / float(ch)
        scaled_w = min(self._render_width, int(round(cw * scale)))
        resized_rgb = cv2.resize(raw_rgb, (scaled_w, self._render_height), interpolation=cv2.INTER_LINEAR)
        canvas_rgb = np.zeros((self._render_height, self._render_width, 3), dtype=np.uint8)
        x_off = (self._render_width - scaled_w) // 2
        canvas_rgb[:, x_off : x_off + scaled_w] = resized_rgb

        new_avatar = self.loader.load_from_array(
            canvas_rgb,
            avatar_id="webcam_capture",
            no_matting=False,
        )
        self._set_active_avatar(new_avatar)
        log("Replaced target avatar from webcam capture in memory.", "source")
        return float(self.source_image["timestamp"])

    def restore_default_avatar(self) -> float:
        """Restore the initial startup Avatar in memory."""
        self._set_active_avatar(self._default_avatar)
        log(f"Restored default avatar '{self._default_avatar.id}' in memory.", "source")
        return float(self.source_image["timestamp"])

    def reset_reference(self) -> None:
        """Reset relative-motion reference frame and landmark tracker."""
        self._reference_kp = None
        self._prev_lmk203 = None
        self._missing_face_frames = 0
        log("Reset relative-motion reference pose to next detected frame.", "info")

    def start_camera(self) -> None:
        if not self.camera_streaming:
            log("Starting camera...", "info")
            self._cap = cv2.VideoCapture(self._device)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._camera_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._camera_height)
            self._cap.set(cv2.CAP_PROP_FPS, _OUTPUT_FPS)
            self.camera_streaming = True

    def stop_camera(self) -> None:
        if self.camera_streaming:
            log("Stopping camera...", "info")
            self.camera_streaming = False
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            placeholder = np.zeros((self._output_height, self._output_width, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                "Camera Stopped",
                (self._output_width // 2 - 150, self._output_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            now = time.time()
            self.current_camera["image"] = placeholder
            self.current_camera["timestamp"] = now
            self.current_camera["byte_string"] = write_numpy_to_byte_string(placeholder)
            self.current_deepfake["image"] = placeholder
            self.current_deepfake["timestamp"] = now
            self.current_deepfake["byte_string"] = write_numpy_to_byte_string(placeholder)

    def stop(self) -> None:
        self._running = False
        self.mic_driver.stop_mic()
        self.stop_camera()

    def _render_motion_rgb(self, motions: MotionFrame) -> np.ndarray:
        with self._avatar_lock:
            avatar = self.avatar
            full_box = self._full_4_3_box
            cyl_x = self._cyl_map_x
            cyl_y = self._cyl_map_y
        with torch.no_grad():
            rgb_gpu, _ = render_chunk(
                motions,
                avatar,
                self.bg,
                stitch=self.stitch,
                warp=self.warp,
                decoder=self.decoder,
                matting=self.matting,
            )
            rgb_u8 = (
                (rgb_gpu[0].clamp(0.0, 1.0) * 255.0)
                .to(torch.uint8)
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
        use_face_box = self.crop_to_face and self.lipsync_only_static_head
        if use_face_box:
            return cv2.remap(
                rgb_u8,
                cyl_x,
                cyl_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        return _crop_to_4_3(rgb_u8, full_box, self._output_width, self._output_height)

    def _render_single_frame_rgb(self, kp: KPInfo) -> np.ndarray:
        """Run one frame through motion_stitch -> warp -> decoder -> MODNet putback and return RGB uint8."""
        with self._avatar_lock:
            avatar = self.avatar
        motions = _build_motion_frame(
            [kp],
            avatar,
            relative=self.relative,
            lipsync_only=self.lipsync_only,
            lipsync_only_static_head=self.lipsync_only_static_head,
            reference_kp=self._reference_kp,
        )
        return self._render_motion_rgb(motions)

    def _run_deep_fake_loop(self) -> None:
        """Background thread loop that reads the camera, extracts motion, and renders the avatar."""
        t_frame = time.time()
        delta_t_frame = 0.0
        last_stream_time = self._last_timestamp_stream
        delta_t_stream = 0.0
        stream_fps = 0.0

        vcam = None
        if self.use_virtual_cam:
            try:
                import pyvirtualcam

                vcam = pyvirtualcam.Camera(
                    width=self._output_width,
                    height=self._output_height,
                    fps=_OUTPUT_FPS,
                )
                log(f"Using virtual camera: {vcam.device}", "camera")
            except Exception as exc:
                log(f"Virtual camera unavailable ({exc}); continuing without pyvirtualcam.", "warning")
                vcam = None

        writer = None
        if self.record_out is not None:
            self.record_out.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio_ffmpeg.write_frames(
                str(self.record_out),
                size=(self._output_width, self._output_height),
                fps=25.0,
                codec="libx264",
                pix_fmt_in="rgb24",
                pix_fmt_out="yuv420p",
                quality=8,
                macro_block_size=1,
            )
            writer.send(None)
            log(f"Recording live webcam render to {self.record_out}", "info")

        try:
            while self._running:
                if not self.camera_streaming or self._cap is None:
                    time.sleep(0.1)
                    continue

                # 1. Read the camera
                t0 = time.time()
                camera_return, camera_buffer = self._cap.read()
                if not camera_return or camera_buffer is None:
                    log("Cannot get camera input.", "error")
                    time.sleep(0.1)
                    continue
                camera_frame = camera_buffer.copy()
                self._latest_raw_camera_bgr = camera_frame
                delta_t_read = time.time() - t0

                # 2. Store current camera frame (cropped to 4:3 aspect ratio for display)
                t1 = time.time()
                ch, cw = camera_frame.shape[:2]
                cam_target_w = min(cw, int(round(ch * 4.0 / 3.0)))
                cam_x1 = (cw - cam_target_w) // 2
                camera_4_3 = camera_frame[:, cam_x1 : cam_x1 + cam_target_w]
                self.current_camera["image"] = camera_4_3
                self.current_camera["timestamp"] = time.time()
                self.current_camera["byte_string"] = write_numpy_to_byte_string(camera_4_3)
                delta_t_copy = time.time() - t1

                # 3. Extract motion from webcam frame and/or microphone audio
                t2 = time.time()
                frame_rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                if self.current_deepfake["active"]:
                    if self.mic_audio_mode and not self.hybrid_webcam_mic:
                        # Pure microphone audio-driven mode (AVTR-1)
                        audio_motion = self.mic_driver.next_motion_frame(
                            static_head=self.lipsync_only_static_head
                        )
                        delta_t_extract = time.time() - t2
                        t3 = time.time()
                        rendered_rgb = self._render_motion_rgb(audio_motion)
                        rendered_bgr = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
                        delta_t_render = time.time() - t3
                    else:
                        kp, self._prev_lmk203 = _extract_driving_kp(
                            frame_rgb, self.loader, self._prev_lmk203
                        )
                        if kp is None:
                            self._missing_face_frames += 1
                            self._prev_lmk203 = None
                            if self._missing_face_frames >= 5 or self._last_kp is None:
                                # Return to neutral pose after 5 consecutive frames with no face
                                kp = (
                                    self._reference_kp
                                    if (self.relative and self._reference_kp is not None)
                                    else self.avatar.kp_info
                                )
                            else:
                                kp = self._last_kp
                        else:
                            self._missing_face_frames = 0
                            self._last_kp = kp
                            if self._reference_kp is None:
                                self._reference_kp = kp

                        if self.hybrid_webcam_mic:
                            # Hybrid mode: head rotation from webcam + lips/expression from mic audio
                            cam_motion = _build_motion_frame(
                                [kp],
                                self.avatar,
                                relative=self.relative,
                                lipsync_only=False,
                                lipsync_only_static_head=False,
                                reference_kp=self._reference_kp,
                            )
                            audio_motion = self.mic_driver.next_motion_frame(static_head=False)
                            hybrid_motion = MotionFrame(
                                R=cam_motion.R,
                                exp=audio_motion.exp,
                                lock_static_kps=False,
                            )
                            delta_t_extract = time.time() - t2
                            t3 = time.time()
                            rendered_rgb = self._render_motion_rgb(hybrid_motion)
                            rendered_bgr = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
                            delta_t_render = time.time() - t3
                        else:
                            delta_t_extract = time.time() - t2
                            # 4. Render avatar frame on GPU
                            t3 = time.time()
                            rendered_rgb = self._render_single_frame_rgb(kp)
                            rendered_bgr = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
                            delta_t_render = time.time() - t3
                else:
                    delta_t_extract = 0.0
                    t3 = time.time()
                    rendered_bgr = cv2.resize(
                        camera_4_3, (self._output_width, self._output_height)
                    )
                    rendered_rgb = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)
                    delta_t_render = time.time() - t3

                # 5. Store rendered frame and push to outputs
                self.current_deepfake["image"] = rendered_bgr
                self.current_deepfake["byte_string"] = write_numpy_to_byte_string(rendered_bgr)
                self.current_deepfake["timestamp"] = time.time()

                if vcam is not None:
                    try:
                        vcam.send(rendered_rgb)
                    except Exception:
                        pass

                if writer is not None:
                    try:
                        writer.send(rendered_rgb.tobytes())
                    except Exception:
                        pass

                delta_t_frame = 0.5 * (time.time() - t_frame) + delta_t_frame * 0.5
                t_frame = time.time()

                # Compute streaming FPS
                if time.time() - self._last_timestamp_stream < 5.0:
                    if self._last_timestamp_stream != last_stream_time:
                        dt = self._last_timestamp_stream - last_stream_time
                        delta_t_stream = 0.5 * dt + delta_t_stream * 0.5
                        last_stream_time = self._last_timestamp_stream
                        stream_fps = 1.0 / (delta_t_stream + 1e-8)
                else:
                    stream_fps = 0.0

                fps_now = 1.0 / max(delta_t_frame, 1e-8)
                print(
                    f"[info] \033[36mFPS: {fps_now:.2f}\033[0m, "
                    f"\033[31mstream FPS: {stream_fps:.2f}\033[0m, "
                    f"\033[32mread: {delta_t_read:.3f}s\033[0m, "
                    f"\033[32mcopy: {delta_t_copy:.3f}s\033[0m, "
                    f"\033[33mextract: {delta_t_extract:.3f}s\033[0m, "
                    f"\033[35mrender: {delta_t_render:.3f}s\033[0m\033[K",
                    end="\r",
                    flush=True,
                )
        finally:
            if vcam is not None:
                vcam.close()
            if writer is not None:
                writer.close()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_deep_fake_loop, args=(), daemon=True)
        self._thread.start()


def _run_webcam_server(driver: WebcamAvatarDriver, port: int) -> None:
    """Serve live MJPEG streams (/stream, /camera) and control endpoints via FastAPI/uvicorn."""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
    import uvicorn

    app = FastAPI()

    def _mjpeg_generator(slot_getter, ping: bool = False):
        latest_bytes: bytes | None = None
        latest_ts = 0.0
        while driver._running:
            slot = slot_getter()
            if latest_ts < slot["timestamp"]:
                latest_ts = slot["timestamp"]
                latest_bytes = slot["byte_string"]
                if ping:
                    driver.ping_stream()
                if latest_bytes is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + latest_bytes + b"\r\n"
                    )
            else:
                if latest_bytes is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + latest_bytes + b"\r\n"
                    )
                    time.sleep(0.02)
                else:
                    time.sleep(0.005)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>AVTR-1 Live Webcam Reenactment</title>
  <style>
    * { box-sizing: border-box; }
    ::-webkit-scrollbar { display: none; }
    html, body {
      background: #111;
      color: #eee;
      font-family: sans-serif;
      margin: 0;
      padding: 8px 12px;
      height: 100vh;
      width: 100vw;
      overflow: hidden;
      text-align: center;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      align-items: center;
    }
    h2 { margin: 4px 0; font-size: 20px; }
    .streams {
      display: flex;
      gap: 16px;
      justify-content: center;
      align-items: center;
      flex: 1;
      min-height: 0;
      width: 100%;
    }
    .card {
      background: #1d1d1d;
      padding: 8px;
      border-radius: 10px;
      display: flex;
      flex-direction: column;
      align-items: center;
      max-height: 100%;
    }
    img {
      max-width: 30vw;
      max-height: calc(100vh - 190px);
      width: auto;
      height: auto;
      border-radius: 6px;
      display: block;
      object-fit: contain;
    }
    .controls { margin-top: 6px; display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }
    button { padding: 8px 14px; border-radius: 6px; border: none; background: #3b82f6; color: white; font-size: 13px; cursor: pointer; }
    button:hover { background: #2563eb; }
    button.mode-btn.selected { background: #16a34a; font-weight: bold; }
  </style>
</head>
<body>
  <h2>AVTR-1 Live Webcam Reenactment</h2>
  <div class="streams">
    <div class="card">
      <div>Rendered Avatar</div>
      <img src="/stream" alt="Rendered Stream">
    </div>
    <div class="card">
      <div>Target Face (In Memory)</div>
      <img src="/source" alt="Target Source Stream">
    </div>
    <div class="card">
      <div>Webcam Input</div>
      <img src="/camera" alt="Camera Stream">
    </div>
  </div>
  <div class="controls">
    <button id="btn-free" class="mode-btn" onclick="setMode('/free_head', 'btn-free')">Webcam: Free Head Movement</button>
    <button id="btn-lipsync" class="mode-btn" onclick="setMode('/lipsync_only', 'btn-lipsync')">Webcam: Lipsync + Head Pose</button>
    <button id="btn-static" class="mode-btn" onclick="setMode('/lipsync_only_static_head', 'btn-static')">Webcam: Lipsync Only (Static Head)</button>
  </div>
  <div class="controls">
    <button id="btn-mic" class="mode-btn" onclick="setMode('/mic_audio', 'btn-mic', true)">Mic Audio (AVTR-1)</button>
    <button id="btn-mic-static" class="mode-btn" onclick="setMode('/mic_audio_static_head', 'btn-mic-static', true)">Mic Audio (Static Head)</button>
    <button id="btn-hybrid" class="mode-btn" onclick="setMode('/hybrid_webcam_mic', 'btn-hybrid', true)">Hybrid: Webcam Head + Mic Lips</button>
  </div>
  <div class="controls">
    <button onclick="takePicture()" style="background:#eab308;color:#111;font-weight:bold;">Take Picture (Set Target)</button>
    <button onclick="fetch('/restore_avatar')">Restore Default Avatar</button>
    <button onclick="fetch('/reset_target')">Reset Neutral Pose</button>
    <button id="btn-crop-face" onclick="toggleCropFace()">Crop Face (Eyebrows/lips ±1cm)</button>
    <button onclick="fetch('/active')">Active</button>
    <button onclick="fetch('/inactive')">Bypass</button>
    <button onclick="fetch('/start_camera')">Start Camera</button>
    <button onclick="fetch('/stop_camera')">Stop Camera</button>
    <button id="btn-browser-mic" onclick="toggleBrowserMic()">Enable Browser Mic</button>
    <span style="display:inline-flex;align-items:center;gap:6px;font-size:13px;color:#aaa;">
      Mic Level:
      <span style="width:90px;height:10px;background:#333;border-radius:5px;overflow:hidden;display:inline-block;">
        <span id="mic-bar" style="display:block;height:100%;width:0%;background:#22c55e;transition:width 0.08s;"></span>
      </span>
    </span>
  </div>
  <script>
    let micWs = null;
    let audioCtx = null;
    let micStream = null;
    let scriptNode = null;

    async function startBrowserMic() {
      if (micWs) return;
      try {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: 16000 } });
        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        const source = audioCtx.createMediaStreamSource(micStream);
        scriptNode = audioCtx.createScriptProcessor(2048, 1, 1);
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        micWs = new WebSocket(proto + '//' + location.host + '/ws/mic');
        micWs.binaryType = 'arraybuffer';
        scriptNode.onaudioprocess = (e) => {
          if (micWs && micWs.readyState === WebSocket.OPEN) {
            const input = e.inputBuffer.getChannelData(0);
            micWs.send(input.buffer.slice(0));
          }
        };
        source.connect(scriptNode);
        scriptNode.connect(audioCtx.destination);
        const btn = document.getElementById('btn-browser-mic');
        if (btn) { btn.textContent = 'Disable Browser Mic'; btn.style.background = '#dc2626'; }
      } catch (err) {
        console.warn('Browser mic not started:', err);
      }
    }

    function stopBrowserMic() {
      if (scriptNode) { scriptNode.disconnect(); scriptNode = null; }
      if (audioCtx) { audioCtx.close(); audioCtx = null; }
      if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
      if (micWs) { micWs.close(); micWs = null; }
      const btn = document.getElementById('btn-browser-mic');
      if (btn) { btn.textContent = 'Enable Browser Mic'; btn.style.background = '#3b82f6'; }
    }

    function toggleBrowserMic() {
      if (micWs) stopBrowserMic();
      else startBrowserMic();
    }

    function updateCropBtn(active) {
      const btn = document.getElementById('btn-crop-face');
      if (!btn) return;
      btn.style.background = active ? '#16a34a' : '#3b82f6';
      btn.style.fontWeight = active ? 'bold' : 'normal';
    }

    async function toggleCropFace() {
      const res = await fetch('/toggle_crop_face');
      const state = await res.text();
      updateCropBtn(state === 'crop_to_face_on');
    }

    async function takePicture() {
      const res = await fetch('/click');
      if (!res.ok) {
        const msg = await res.text();
        alert(msg);
      }
    }

    async function setMode(endpoint, btnId, needsMic = false) {
      if (needsMic && !micWs) {
        startBrowserMic();
      }
      const res = await fetch(endpoint);
      if (!res.ok) {
        const msg = await res.text();
        alert(msg);
        return;
      }
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('selected'));
      const el = document.getElementById(btnId);
      if (el) el.classList.add('selected');
    }
    async function syncMode() {
      try {
        const res = await fetch('/mode');
        const mode = await res.text();
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('selected'));
        const map = {
          'free_head': 'btn-free',
          'lipsync_only': 'btn-lipsync',
          'lipsync_only_static_head': 'btn-static',
          'mic_audio': 'btn-mic',
          'mic_audio_static_head': 'btn-mic-static',
          'hybrid_webcam_mic': 'btn-hybrid'
        };
        const el = document.getElementById(map[mode]);
        if (el) el.classList.add('selected');
      } catch (e) {}
    }
    setInterval(async () => {
      try {
        const r = await fetch('/mic_status');
        const data = await r.json();
        const bar = document.getElementById('mic-bar');
        if (bar) {
          const pct = Math.min(100, Math.round((data.rms || 0) * 400));
          bar.style.width = pct + '%';
        }
        updateCropBtn(Boolean(data.crop_to_face));
      } catch (e) {}
    }, 150);
    syncMode();
  </script>
</body>
</html>"""

    @app.get("/mode", response_class=PlainTextResponse)
    def get_mode() -> str:
        if driver.hybrid_webcam_mic:
            return "hybrid_webcam_mic"
        if driver.mic_audio_mode:
            if driver.lipsync_only_static_head:
                return "mic_audio_static_head"
            return "mic_audio"
        if driver.lipsync_only_static_head:
            return "lipsync_only_static_head"
        if driver.lipsync_only:
            return "lipsync_only"
        return "free_head"

    @app.get("/mic_status", response_class=JSONResponse)
    def mic_status() -> dict[str, Any]:
        return {
            "rms": driver.mic_driver.audio_rms,
            "host_mic_streaming": driver.mic_driver.mic_streaming,
            "mode": get_mode(),
            "crop_to_face": driver.crop_to_face,
        }

    @app.get("/crop_to_face_on", response_class=PlainTextResponse)
    def crop_to_face_on() -> str:
        driver.crop_to_face = True
        log("Face crop (1cm above eyebrows to 1cm below lower lip) enabled.", "info")
        return "crop_to_face_on"

    @app.get("/crop_to_face_off", response_class=PlainTextResponse)
    def crop_to_face_off() -> str:
        driver.crop_to_face = False
        log("Face crop disabled (full 4:3 frame).", "info")
        return "crop_to_face_off"

    @app.get("/toggle_crop_face", response_class=PlainTextResponse)
    def toggle_crop_face() -> str:
        driver.crop_to_face = not driver.crop_to_face
        state = "crop_to_face_on" if driver.crop_to_face else "crop_to_face_off"
        log(f"Toggled face crop: {state}", "info")
        return state

    @app.websocket("/ws/mic")
    async def ws_mic(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                raw = await websocket.receive_bytes()
                samples = np.frombuffer(raw, dtype=np.float32)
                driver.mic_driver.push_audio_samples(samples)
        except WebSocketDisconnect:
            pass

    @app.get("/free_head", response_class=PlainTextResponse)
    def free_head() -> str:
        driver.mic_audio_mode = False
        driver.hybrid_webcam_mic = False
        driver.lipsync_only = False
        driver.lipsync_only_static_head = False
        log("Switched mode to: Webcam Free Head Movement", "info")
        return "free_head"

    @app.get("/lipsync_only", response_class=PlainTextResponse)
    def lipsync_only() -> str:
        driver.mic_audio_mode = False
        driver.hybrid_webcam_mic = False
        driver.lipsync_only = True
        driver.lipsync_only_static_head = False
        log("Switched mode to: Webcam Lipsync + Head Pose", "info")
        return "lipsync_only"

    @app.get("/lipsync_only_static_head", response_class=PlainTextResponse)
    def lipsync_only_static_head() -> str:
        driver.mic_audio_mode = False
        driver.hybrid_webcam_mic = False
        driver.lipsync_only = False
        driver.lipsync_only_static_head = True
        log("Switched mode to: Webcam Lipsync Only (Static Head)", "info")
        return "lipsync_only_static_head"

    @app.get("/mic_audio", response_class=PlainTextResponse)
    def mic_audio() -> PlainTextResponse:
        try:
            driver.mic_driver.ensure_loaded()
            driver.mic_driver.start_mic()
        except Exception as exc:
            return PlainTextResponse(str(exc), status_code=400)
        driver.mic_audio_mode = True
        driver.hybrid_webcam_mic = False
        driver.lipsync_only = False
        driver.lipsync_only_static_head = False
        log("Switched mode to: Mic Audio (AVTR-1)", "audio")
        return PlainTextResponse("mic_audio")

    @app.get("/mic_audio_static_head", response_class=PlainTextResponse)
    def mic_audio_static_head() -> PlainTextResponse:
        try:
            driver.mic_driver.ensure_loaded()
            driver.mic_driver.start_mic()
        except Exception as exc:
            return PlainTextResponse(str(exc), status_code=400)
        driver.mic_audio_mode = True
        driver.hybrid_webcam_mic = False
        driver.lipsync_only = False
        driver.lipsync_only_static_head = True
        log("Switched mode to: Mic Audio (Static Head)", "audio")
        return PlainTextResponse("mic_audio_static_head")

    @app.get("/hybrid_webcam_mic", response_class=PlainTextResponse)
    def hybrid_webcam_mic() -> PlainTextResponse:
        try:
            driver.mic_driver.ensure_loaded()
            driver.mic_driver.start_mic()
        except Exception as exc:
            return PlainTextResponse(str(exc), status_code=400)
        driver.mic_audio_mode = True
        driver.hybrid_webcam_mic = True
        driver.lipsync_only = False
        driver.lipsync_only_static_head = False
        log("Switched mode to: Hybrid (Webcam Head + Mic Lips)", "audio")
        return PlainTextResponse("hybrid_webcam_mic")

    @app.get("/stage", response_class=HTMLResponse)
    def stage() -> str:
        return """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>AVTR-1 Stage Stream</title>
  <style>
    ::-webkit-scrollbar { display: none; }
    html, body {
      margin: 0;
      padding: 0;
      width: 100vw;
      height: 100vh;
      background-color: black;
      overflow: hidden;
      display: flex;
      justify-content: center;
      align-items: center;
    }
    img {
      display: block;
      width: auto;
      height: 100vh;
      max-width: 100vw;
      object-fit: contain;
    }
  </style>
</head>
<body>
  <img src="/stream" alt="Stage Stream">
</body>
</html>"""

    @app.get("/stream")
    def stream() -> StreamingResponse:
        return StreamingResponse(
            _mjpeg_generator(lambda: driver.current_deepfake, ping=True),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/camera")
    def camera() -> StreamingResponse:
        return StreamingResponse(
            _mjpeg_generator(lambda: driver.current_camera, ping=False),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/source")
    def source() -> StreamingResponse:
        return StreamingResponse(
            _mjpeg_generator(lambda: driver.source_image, ping=False),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/click", response_class=PlainTextResponse)
    def click() -> PlainTextResponse:
        try:
            ts = driver.capture_source_image_from_camera()
            return PlainTextResponse(str(ts))
        except Exception as exc:
            log(f"Failed to capture target face from camera: {exc}", "error")
            return PlainTextResponse(f"Could not capture face from camera: {exc}", status_code=400)

    @app.get("/restore_avatar", response_class=PlainTextResponse)
    def restore_avatar() -> str:
        ts = driver.restore_default_avatar()
        return str(ts)

    @app.get("/reset_target", response_class=PlainTextResponse)
    def reset_target() -> str:
        driver.reset_reference()
        return "reset_target"

    @app.get("/active", response_class=PlainTextResponse)
    def active() -> str:
        driver.current_deepfake["active"] = True
        return "active"

    @app.get("/inactive", response_class=PlainTextResponse)
    def inactive() -> str:
        driver.current_deepfake["active"] = False
        return "inactive"

    @app.get("/start_camera", response_class=PlainTextResponse)
    def start_camera() -> str:
        driver.start_camera()
        return "camera_started"

    @app.get("/stop_camera", response_class=PlainTextResponse)
    def stop_camera() -> str:
        driver.stop_camera()
        return "camera_stopped"

    print(f"\n[webcam] Live preview server running at http://127.0.0.1:{port}/ (and /stream). Press Ctrl+C to stop.")
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]

    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    while driver._running and server_thread.is_alive():
        time.sleep(0.1)
    server.should_exit = True
    server.force_exit = True


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--driving",
        type=Path,
        default=None,
        help="Path to the driving video file (e.g. .mp4, .mov, .avi). Omit when using --webcam.",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Use live webcam input instead of a driving video file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="Integrated Webcam",
        help="Webcam device name substring to match (default: 'Integrated Webcam')",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Explicit webcam device index (overrides --device)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Webcam capture width in pixels (default: 960)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=540,
        help="Webcam capture height in pixels (default: 540)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Local HTTP port for live webcam MJPEG stream (default: 8001, 0 to disable)",
    )
    parser.add_argument(
        "--no-virtual-cam",
        dest="virtual_cam",
        action="store_false",
        help="Disable pyvirtualcam output in webcam mode.",
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
        default=None,
        help="Output video path (default: demo_video_driven.mp4 in offline mode; optional recording path in webcam mode)",
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
        "--crop-face",
        action="store_true",
        help="In static head mode, crop the 4:3 output stream to ~1cm above the eyebrows and ~1cm below the lower lip.",
    )
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Start in live microphone audio-driven mode (AVTR-1).",
    )
    parser.add_argument(
        "--mic-device",
        type=str,
        default="",
        help="Microphone device name substring to match (optional).",
    )
    parser.add_argument(
        "--mic-index",
        type=int,
        default=None,
        help="Explicit microphone device index (optional).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of frames to process in offline mode.",
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
        help="Yield each frame as soon as it's ready in offline mode (default: on).",
    )
    args = parser.parse_args()

    use_webcam = args.webcam or args.mic or (args.driving is None and args.camera_index is not None)
    if not use_webcam and args.driving is None:
        parser.error("Provide either --driving <video_path> or --webcam (or --camera-index).")
    if not use_webcam and not args.driving.is_file():
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

    # ---- Live Webcam Mode ---------------------------------------------------
    if use_webcam:
        driver = WebcamAvatarDriver(
            avatar=avatar,
            bg=bg,
            loader=loader,
            stitch=stitch,
            warp=warp,
            decoder=decoder,
            matting=matting,
            device_name=args.device,
            camera_index=args.camera_index,
            camera_width=args.width,
            camera_height=args.height,
            relative=args.relative,
            lipsync_only=args.lipsync_only,
            lipsync_only_static_head=args.lipsync_only_static_head,
            crop_to_face=args.crop_face,
            mic_audio_mode=args.mic,
            mic_device_name=args.mic_device,
            mic_index=args.mic_index,
            use_virtual_cam=args.virtual_cam,
            record_out=args.out,
        )

        def _handle_sigint(_signum: int, _frame: object) -> None:
            print()
            log("Ctrl+C pressed. Exiting...", "info")
            driver.stop()
            if driver._thread is not None and driver._thread.is_alive():
                driver._thread.join(timeout=1.5)
            os._exit(0)

        signal.signal(signal.SIGINT, _handle_sigint)
        signal.signal(signal.SIGTERM, _handle_sigint)

        try:
            if args.port > 0:
                _run_webcam_server(driver, args.port)
            else:
                while driver._running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print()
            log("Ctrl+C pressed. Exiting...", "info")
        finally:
            driver.stop()
            if driver._thread is not None and driver._thread.is_alive():
                driver._thread.join(timeout=1.5)
            os._exit(0)
        return

    # ---- Offline Video Mode -------------------------------------------------
    out_path = args.out if args.out is not None else Path("demo_video_driven.mp4")

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
    first_kp: KPInfo | None = None
    missing_frames = 0

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
                missing_frames += 1
                if missing_frames >= 5 or last_kp is None:
                    kp = first_kp if (args.relative and first_kp is not None) else avatar.kp_info
                else:
                    kp = last_kp
            else:
                missing_frames = 0
                last_kp = kp
                if first_kp is None:
                    first_kp = kp
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mux_audio = args.mux and _has_audio_stream(args.driving)
    writer = imageio_ffmpeg.write_frames(
        str(out_path),
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
        f"render speed {produced / elapsed_render:.1f} fps) -> {out_path}"
    )


if __name__ == "__main__":
    main()
