ECHO off
ECHO Run deep fake using Logitech webcam
call pixi run generate_video --webcam --camera-index 704 --avatar arnold --lipsync-only-static-head --mic --bg plain_black