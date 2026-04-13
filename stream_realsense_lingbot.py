#!/usr/bin/env python3
"""
Realtime Intel RealSense + LingBot-Depth streaming.

Displays:
1) Raw RGB image
2) Raw depth (colorized)
3) LingBot-Depth refined depth (colorized)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise ImportError(
        "pyrealsense2 is required. Install it with: pip install pyrealsense2"
    ) from exc


def add_lingbot_to_python_path() -> None:
    """Expose ../lingbot-depth as importable module root."""
    this_dir = Path(__file__).resolve().parent
    lingbot_dir = (this_dir.parent / "lingbot-depth").resolve()
    if not lingbot_dir.exists():
        raise FileNotFoundError(
            f"Could not find lingbot-depth at: {lingbot_dir}. "
            "Expected structure: k_depth/lingbot-depth and k_depth/depth_stream."
        )
    sys.path.insert(0, str(lingbot_dir))


add_lingbot_to_python_path()
from mdm.model.v2 import MDMModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream RealSense RGB-D + LingBot-Depth output in realtime."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="robbyant/lingbot-depth-pretrain-vitl-14-v0.5",
        help="Hugging Face model id or local checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Inference device",
    )
    parser.add_argument("--width", type=int, default=640, help="Stream width")
    parser.add_argument("--height", type=int, default=480, help="Stream height")
    parser.add_argument("--fps", type=int, default=60, help="Camera FPS target")
    parser.add_argument(
        "--resolution-level",
        type=int,
        default=6,
        help="Model resolution level (0-9). Lower is faster.",
    )
    parser.add_argument(
        "--infer-every",
        type=int,
        default=2,
        help="Run model once every N frames (1 = infer every frame).",
    )
    parser.add_argument(
        "--infer-width",
        type=int,
        default=0,
        help="Inference width (0 uses stream resolution).",
    )
    parser.add_argument(
        "--infer-height",
        type=int,
        default=0,
        help="Inference height (0 uses stream resolution).",
    )
    parser.add_argument("--min-depth", type=float, default=0.2, help="Color min depth (m)")
    parser.add_argument("--max-depth", type=float, default=5.0, help="Color max depth (m)")
    parser.add_argument(
        "--window-scale",
        type=float,
        default=1.0,
        help="Display scale factor (e.g. 0.8)",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Disable model mask application",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="captures",
        help="Snapshot directory when pressing 's'",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Start camera with negotiated mode and exit (no window).",
    )
    return parser.parse_args()


def colorize_depth(depth_m: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    depth_clipped = np.clip(depth_m, min_depth, max_depth)
    depth_u8 = (
        (depth_clipped - min_depth) / (max_depth - min_depth + 1e-8) * 255.0
    ).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    depth_color[~valid] = 0
    return depth_color


def build_intrinsics_normalized(intr: rs.intrinsics, width: int, height: int) -> np.ndarray:
    return np.array(
        [
            [intr.fx / width, 0.0, intr.ppx / width],
            [0.0, intr.fy / height, intr.ppy / height],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _fps_sort_key(mode: tuple[int, int, int]) -> tuple[int, int, int]:
    return mode[0], mode[1], mode[2]


def query_color_depth_modes(
    device: rs.device,
) -> tuple[dict[tuple[int, int, int], set], set[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """Get color/depth modes and same-resolution common RGB-D modes."""
    color_mode_formats: dict[tuple[int, int, int], set] = {}
    depth_modes: set[tuple[int, int, int]] = set()
    for sensor in device.sensors:
        for profile in sensor.get_stream_profiles():
            try:
                vprof = profile.as_video_stream_profile()
            except Exception:
                continue

            whf = (vprof.width(), vprof.height(), vprof.fps())
            stream = profile.stream_type()
            fmt = profile.format()

            if stream == rs.stream.color and fmt in (rs.format.bgr8, rs.format.rgb8):
                color_mode_formats.setdefault(whf, set()).add(fmt)
            if stream == rs.stream.depth and fmt == rs.format.z16:
                depth_modes.add(whf)

    common_modes = sorted([m for m in color_mode_formats if m in depth_modes], key=_fps_sort_key)
    return color_mode_formats, depth_modes, common_modes


def sort_stream_pairs_by_preference(
    color_modes: list[tuple[int, int, int]],
    depth_modes: list[tuple[int, int, int]],
    target: tuple[int, int, int],
) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    tw, th, tfps = target

    same_fps_pairs = [
        (cmode, dmode) for cmode in color_modes for dmode in depth_modes if cmode[2] == dmode[2]
    ]
    candidate_pairs = same_fps_pairs if same_fps_pairs else [
        (cmode, dmode) for cmode in color_modes for dmode in depth_modes
    ]

    def score(pair: tuple[tuple[int, int, int], tuple[int, int, int]]) -> tuple[int, int, int, int, int, int, int]:
        (cw, ch, cfps), (dw, dh, dfps) = pair
        same_fps = 0 if cfps == dfps else 1
        fps_penalty = 0 if cfps <= tfps else 1
        fps_delta = abs(cfps - tfps)
        color_same_res = 0 if (cw == tw and ch == th) else 1
        color_res_delta = abs(cw - tw) + abs(ch - th)
        depth_res_delta = abs(dw - tw) + abs(dh - th)
        stream_res_mismatch = 0 if (cw == dw and ch == dh) else 1
        return (
            same_fps,
            fps_penalty,
            fps_delta,
            color_same_res,
            color_res_delta,
            stream_res_mismatch,
            depth_res_delta,
        )

    return sorted(candidate_pairs, key=score)


def start_pipeline_with_mode_negotiation(
    pipeline: rs.pipeline,
    requested_mode: tuple[int, int, int],
) -> tuple[
    rs.pipeline_profile,
    tuple[int, int, int],
    tuple[int, int, int],
    rs.format,
    list[tuple[int, int, int]],
]:
    """Start RealSense with best matching color/depth modes for requested settings."""
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No Intel RealSense device found.")

    device = devices[0]
    color_mode_formats, depth_modes, common_modes = query_color_depth_modes(device)
    if not color_mode_formats or not depth_modes:
        raise RuntimeError("No valid color/depth modes found for color(BGR/RGB) + depth(Z16).")

    candidate_pairs = sort_stream_pairs_by_preference(
        list(color_mode_formats.keys()),
        sorted(depth_modes, key=_fps_sort_key),
        requested_mode,
    )

    last_error = None
    for color_mode, depth_mode in candidate_pairs:
        color_formats = []
        if rs.format.bgr8 in color_mode_formats.get(color_mode, set()):
            color_formats.append(rs.format.bgr8)
        if rs.format.rgb8 in color_mode_formats.get(color_mode, set()):
            color_formats.append(rs.format.rgb8)

        for color_format in color_formats:
            config = rs.config()
            cw, ch, cfps = color_mode
            dw, dh, dfps = depth_mode
            config.enable_stream(rs.stream.color, cw, ch, color_format, cfps)
            config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, dfps)
            try:
                profile = pipeline.start(config)
                return profile, color_mode, depth_mode, color_format, common_modes
            except RuntimeError as exc:
                last_error = exc

    raise RuntimeError(
        f"Failed to start RealSense stream with negotiated modes. Last error: {last_error}"
    )


def overlay_titles(frame: np.ndarray, tile_width: int) -> None:
    labels = ("Raw RGB", "Raw Depth (Colorized)", "LingBot-Depth")
    for i, label in enumerate(labels):
        cv2.putText(
            frame,
            label,
            (i * tile_width + 8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def main() -> int:
    args = parse_args()
    if args.min_depth >= args.max_depth:
        raise ValueError("--min-depth must be smaller than --max-depth")
    if args.infer_every < 1:
        raise ValueError("--infer-every must be >= 1")
    if (args.infer_width == 0) ^ (args.infer_height == 0):
        raise ValueError("Set both --infer-width and --infer-height, or neither.")
    if args.infer_width < 0 or args.infer_height < 0:
        raise ValueError("--infer-width and --infer-height must be >= 0")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Loading model: {args.model}")
    model = MDMModel.from_pretrained(args.model).to(device)
    model.eval()

    pipeline = rs.pipeline()
    requested_mode = (args.width, args.height, args.fps)
    (
        profile,
        selected_color_mode,
        selected_depth_mode,
        color_format,
        common_modes,
    ) = start_pipeline_with_mode_negotiation(pipeline, requested_mode)
    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    if selected_color_mode != requested_mode or selected_depth_mode != requested_mode:
        print(
            f"Requested mode {requested_mode[0]}x{requested_mode[1]}@{requested_mode[2]} is not available."
        )
        print(
            "Using negotiated streams: "
            f"color {selected_color_mode[0]}x{selected_color_mode[1]}@{selected_color_mode[2]}, "
            f"depth {selected_depth_mode[0]}x{selected_depth_mode[1]}@{selected_depth_mode[2]}"
        )
    print(
        "Supported same-resolution RGB-D modes: "
        + (", ".join([f"{w}x{h}@{fps}" for (w, h, fps) in common_modes]) if common_modes else "none")
    )
    print(f"Color format: {str(color_format).split('.')[-1]}")
    print(f"Depth scale: {depth_scale:.8f} meters/unit")
    print("Controls: q/Esc quit, s save snapshot")

    if args.probe_only:
        pipeline.stop()
        return 0

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    window_name = "LingBot-Depth RealSense Stream"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    fps_ema = None
    model_fps_ema = None
    last_depth_pred = None
    last_infer_ms = 0.0
    frame_idx = 0

    try:
        while True:
            t0 = time.perf_counter()
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_data = np.asanyarray(color_frame.get_data())
            if color_format == rs.format.rgb8:
                color_bgr = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)
            else:
                color_bgr = color_data
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m = depth_raw.astype(np.float32) * depth_scale

            h, w = color_bgr.shape[:2]
            intrinsics_rs = depth_frame.profile.as_video_stream_profile().get_intrinsics()
            intrinsics = build_intrinsics_normalized(intrinsics_rs, w, h)

            run_infer = (frame_idx % args.infer_every == 0) or (last_depth_pred is None)
            if args.infer_width > 0:
                infer_w, infer_h = args.infer_width, args.infer_height
                color_for_model = cv2.resize(
                    color_bgr, (infer_w, infer_h), interpolation=cv2.INTER_AREA
                )
                depth_for_model = cv2.resize(
                    depth_m, (infer_w, infer_h), interpolation=cv2.INTER_NEAREST
                )
            else:
                infer_w, infer_h = w, h
                color_for_model = color_bgr
                depth_for_model = depth_m

            if run_infer:
                image_rgb = cv2.cvtColor(color_for_model, cv2.COLOR_BGR2RGB)
                image_tensor = (
                    torch.from_numpy(image_rgb)
                    .to(device=device, dtype=torch.float32)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    / 255.0
                )
                depth_tensor = (
                    torch.from_numpy(depth_for_model)
                    .to(device=device, dtype=torch.float32)
                    .unsqueeze(0)
                )
                intrinsics_tensor = (
                    torch.from_numpy(intrinsics)
                    .to(device=device, dtype=torch.float32)
                    .unsqueeze(0)
                )

                infer_t0 = time.perf_counter()
                output = model.infer(
                    image_tensor,
                    depth_in=depth_tensor,
                    intrinsics=intrinsics_tensor,
                    resolution_level=args.resolution_level,
                    apply_mask=not args.no_mask,
                    use_fp16=(device.type == "cuda"),
                )
                last_infer_ms = (time.perf_counter() - infer_t0) * 1000.0

                depth_pred_infer = output["depth"].squeeze(0).detach().cpu().numpy()
                if (infer_w, infer_h) != (w, h):
                    depth_pred = cv2.resize(
                        depth_pred_infer, (w, h), interpolation=cv2.INTER_LINEAR
                    )
                else:
                    depth_pred = depth_pred_infer
                last_depth_pred = depth_pred

                model_fps = 1000.0 / max(last_infer_ms, 1e-6)
                model_fps_ema = (
                    model_fps
                    if model_fps_ema is None
                    else (0.9 * model_fps_ema + 0.1 * model_fps)
                )

            depth_pred = last_depth_pred if last_depth_pred is not None else depth_m

            depth_raw_color = colorize_depth(depth_m, args.min_depth, args.max_depth)
            depth_pred_color = colorize_depth(depth_pred, args.min_depth, args.max_depth)

            tiled = np.hstack((color_bgr, depth_raw_color, depth_pred_color))
            overlay_titles(tiled, w)

            dt = time.perf_counter() - t0
            fps = 1.0 / max(dt, 1e-6)
            fps_ema = fps if fps_ema is None else (0.9 * fps_ema + 0.1 * fps)

            model_fps_txt = f"{(model_fps_ema or 0.0):.1f}"
            infer_tag = "new" if run_infer else "reuse"
            footer = (
                f"Display FPS: {fps_ema:.1f} | Model FPS: {model_fps_txt} | "
                f"Infer: {last_infer_ms:.1f} ms ({infer_tag}) | Device: {device.type.upper()}"
            )
            cv2.rectangle(tiled, (0, h - 32), (tiled.shape[1], h), (0, 0, 0), -1)
            cv2.putText(
                tiled,
                footer,
                (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if args.window_scale != 1.0:
                new_w = max(1, int(tiled.shape[1] * args.window_scale))
                new_h = max(1, int(tiled.shape[0] * args.window_scale))
                tiled = cv2.resize(tiled, (new_w, new_h), interpolation=cv2.INTER_AREA)

            cv2.imshow(window_name, tiled)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                ts = time.strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(str(save_dir / f"{ts}_rgb.png"), color_bgr)
                cv2.imwrite(str(save_dir / f"{ts}_raw_depth_color.png"), depth_raw_color)
                cv2.imwrite(str(save_dir / f"{ts}_lingbot_depth_color.png"), depth_pred_color)
                np.save(save_dir / f"{ts}_raw_depth_m.npy", depth_m)
                np.save(save_dir / f"{ts}_lingbot_depth_m.npy", depth_pred)
                print(f"Saved snapshots to {save_dir}")

            frame_idx += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
