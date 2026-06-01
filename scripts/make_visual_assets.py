#!/usr/bin/env python3
"""Create poster/report visual assets from downloaded rollout videos."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


VIDEO_SPECS = {
    "walk_loose": {
        "label": "Walk loose",
        "path": Path("motions/from_modal/1780201118_a-person-walks-forward_seed0_loose-video/logs/rsl_rl/g1_tracking/2026-05-31_04-21-56_1780201118_a-person-walks-forward_seed0_loose-video/videos/train/rl-video-step-40000.mp4"),
    },
    "walk_strict": {
        "label": "Walk strict",
        "path": Path("motions/from_modal/1780200983_a-person-walks-forward_seed0_strict/logs/rsl_rl/g1_tracking/2026-05-31_04-19-27_1780200983_a-person-walks-forward_seed0_strict/videos/train/rl-video-step-40000.mp4"),
    },
    "walk_curriculum": {
        "label": "Walk curriculum",
        "path": Path("motions/from_modal/1780201423_a-person-walks-forward_seed0_curriculum/logs/rsl_rl/g1_tracking/2026-05-31_04-46-33_1780201423_a-person-walks-forward_seed0_curriculum_stage2_strict/videos/train/rl-video-step-20000.mp4"),
    },
    "squat_loose": {
        "label": "Squat loose",
        "path": Path("motions/from_modal/1780201117_a-person-squats-down-and-stands-up_seed0_loose-video/logs/rsl_rl/g1_tracking/2026-05-31_04-21-52_1780201117_a-person-squats-down-and-stands-up_seed0_loose-video/videos/train/rl-video-step-40000.mp4"),
    },
    "squat_strict": {
        "label": "Squat strict",
        "path": Path("motions/from_modal/1780201117_a-person-squats-down-and-stands-up_seed0_strict/logs/rsl_rl/g1_tracking/2026-05-31_04-21-56_1780201117_a-person-squats-down-and-stands-up_seed0_strict/videos/train/rl-video-step-40000.mp4"),
    },
    "wave_strict": {
        "label": "Wave strict",
        "path": Path("motions/from_modal/1780201117_a-person-waves-with-their-right-hand_seed0_strict/logs/rsl_rl/g1_tracking/2026-05-31_04-23-46_1780201117_a-person-waves-with-their-right-hand_seed0_strict/videos/train/rl-video-step-40000.mp4"),
    },
    "tap_strict": {
        "label": "Tap strict",
        "path": Path("motions/from_modal/1780201117_a-person-taps-themselves-on-the-head_seed0_strict/logs/rsl_rl/g1_tracking/2026-05-31_04-23-45_1780201117_a-person-taps-themselves-on-the-head_seed0_strict/videos/train/rl-video-step-40000.mp4"),
    },
    "squat_gradual_thr2": {
        "label": "Squat gradual thr=2",
        "path": Path("motions/from_modal/1780296901_a-person-squats-down-and-stands-up_seed0_squat-gradual-curriculum/logs/rsl_rl/g1_tracking/2026-06-01_07-38-53_1780296901_a-person-squats-down-and-stands-up_seed0_squat-gradual-curriculum_stage05_thr2/videos/train/rl-video-step-10000.mp4"),
    },
    "squat_gradual_thr1": {
        "label": "Squat gradual thr=1",
        "path": Path("motions/from_modal/1780296901_a-person-squats-down-and-stands-up_seed0_squat-gradual-curriculum/logs/rsl_rl/g1_tracking/2026-06-01_07-49-05_1780296901_a-person-squats-down-and-stands-up_seed0_squat-gradual-curriculum_stage06_thr1/videos/train/rl-video-step-10000.mp4"),
    },
    "squat_gradual_strict": {
        "label": "Squat gradual strict",
        "path": Path("motions/from_modal/1780296901_a-person-squats-down-and-stands-up_seed0_squat-gradual-curriculum/logs/rsl_rl/g1_tracking/2026-06-01_07-59-18_1780296901_a-person-squats-down-and-stands-up_seed0_squat-gradual-curriculum_stage07_strict/videos/train/rl-video-step-10000.mp4"),
    },
    "jump_loose": {
        "label": "Jump loose",
        "path": Path("motions/from_modal/1780296901_a-person-jumps_seed0_jump-loose-extra/logs/rsl_rl/g1_tracking/2026-06-01_07-00-29_1780296901_a-person-jumps_seed0_jump-loose-extra/videos/train/rl-video-step-40000.mp4"),
    },
}

HARD_REFERENCE_VIDEO_SPECS = {
    "jump": {
        "label": "Jump reference",
        "path": Path("motions/from_modal/1780272604_a-person-jumps_seed0_hard-reference/reference_motion.mp4"),
    },
    "roll": {
        "label": "Roll reference",
        "path": Path("motions/from_modal/1780272604_a-person-rolls-forward-on-the-ground_seed0_hard-reference/reference_motion.mp4"),
    },
    "backflip": {
        "label": "Backflip reference",
        "path": Path("motions/from_modal/1780273575_a-person-does-a-backflip_seed0_hard-reference-extra/reference_motion.mp4"),
    },
    "cartwheel": {
        "label": "Cartwheel reference",
        "path": Path("motions/from_modal/1780273575_a-person-does-a-cartwheel_seed0_hard-reference-extra/reference_motion.mp4"),
    },
}

PANELS = [
    {
        "filename": "final_walk_sequence_panel.png",
        "title": "Walk rollouts at final checkpoint",
        "rows": ["walk_loose", "walk_strict", "walk_curriculum"],
        "times": [0.2, 0.8, 1.4, 2.0],
    },
    {
        "filename": "final_squat_sequence_panel.png",
        "title": "Squat rollouts: loose succeeds, strict terminates early",
        "rows": ["squat_loose", "squat_strict"],
        "times": [0.2, 0.8, 1.4, 2.0],
    },
    {
        "filename": "final_gesture_sequence_panel.png",
        "title": "Gesture rollouts under strict termination",
        "rows": ["wave_strict", "tap_strict"],
        "times": [0.2, 0.8, 1.4, 2.0],
    },
    {
        "filename": "final_gradual_curriculum_sequence_panel.png",
        "title": "Gradual squat curriculum: strict transfer collapses",
        "rows": ["squat_gradual_thr2", "squat_gradual_thr1", "squat_gradual_strict"],
        "times": [0.2, 0.8, 1.4, 2.0],
    },
    {
        "filename": "final_jump_sequence_panel.png",
        "title": "Jump rollout under loose termination",
        "rows": ["jump_loose"],
        "times": [0.2, 0.8, 1.4, 2.0],
    },
]

HARD_REFERENCE_PANEL = {
    "filename": "final_hard_reference_panel.png",
    "title": "Hard text-generated references before PPO",
    "rows": ["jump", "roll", "backflip", "cartwheel"],
    "times": [0.2, 0.8, 1.4, 2.0],
}


def _require_tools() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not on PATH")


def _load_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise SystemExit("Pillow is required to generate visual assets") from exc
    return Image, ImageDraw, ImageFont


def _font(size: int):
    _, _, ImageFont = _load_pillow()
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _extract_frame(video_path: Path, timestamp: float, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(output_path),
        ],
        check=True,
    )


def _draw_centered(draw, box: tuple[int, int, int, int], text: str, font, fill: str) -> None:
    x0, y0, x1, y1 = box
    text_box = draw.textbbox((0, 0), text, font=font)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    draw.text(
        (x0 + (x1 - x0 - text_w) // 2, y0 + (y1 - y0 - text_h) // 2),
        text,
        fill=fill,
        font=font,
    )


def create_panel(
    panel: dict[str, object],
    output_dir: Path,
    video_specs: dict[str, dict[str, object]] | None = None,
) -> None:
    Image, ImageDraw, _ = _load_pillow()
    video_specs = video_specs or VIDEO_SPECS

    cell_w, cell_h = 320, 190
    row_label_w = 150
    header_h = 42
    title_h = 46
    padding = 12
    times = list(panel["times"])
    row_keys = list(panel["rows"])

    width = row_label_w + len(times) * cell_w + (len(times) + 2) * padding
    height = title_h + header_h + len(row_keys) * cell_h + (len(row_keys) + 3) * padding
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = _font(24)
    label_font = _font(18)
    small_font = _font(15)
    _draw_centered(draw, (0, padding, width, padding + title_h), str(panel["title"]), title_font, "black")

    for col, timestamp in enumerate(times):
        x0 = row_label_w + padding * 2 + col * cell_w
        y0 = padding * 2 + title_h
        _draw_centered(draw, (x0, y0, x0 + cell_w, y0 + header_h), f"t={timestamp:.1f}s", small_font, "black")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for row, key in enumerate(row_keys):
            spec = video_specs[key]
            video_path = Path(spec["path"])
            if not video_path.exists():
                raise SystemExit(f"Missing video: {video_path}")

            y0 = padding * 3 + title_h + header_h + row * cell_h
            _draw_centered(
                draw,
                (padding, y0, row_label_w + padding, y0 + cell_h),
                str(spec["label"]),
                label_font,
                "black",
            )

            for col, timestamp in enumerate(times):
                frame_path = tmpdir / f"{key}_{col}.jpg"
                _extract_frame(video_path, float(timestamp), frame_path)
                frame = Image.open(frame_path).convert("RGB")
                frame.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
                x0 = row_label_w + padding * 2 + col * cell_w
                canvas.paste(frame, (x0 + (cell_w - frame.width) // 2, y0 + (cell_h - frame.height) // 2))

    output_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(output_dir / str(panel["filename"]))


def create_hard_reference_panel(output_dir: Path) -> None:
    create_panel(HARD_REFERENCE_PANEL, output_dir, HARD_REFERENCE_VIDEO_SPECS)


def copy_selected_videos(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, spec in VIDEO_SPECS.items():
        source = Path(spec["path"])
        if not source.exists():
            raise SystemExit(f"Missing video: {source}")
        shutil.copy2(source, output_dir / f"{key}.mp4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--media-dir", type=Path, default=Path("media/final_videos"))
    parser.add_argument("--no-copy-videos", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require_tools()

    for panel in PANELS:
        create_panel(panel, args.figures_dir)
        print(f"Wrote {args.figures_dir / str(panel['filename'])}")

    create_hard_reference_panel(args.figures_dir)
    print(f"Wrote {args.figures_dir / str(HARD_REFERENCE_PANEL['filename'])}")

    if not args.no_copy_videos:
        copy_selected_videos(args.media_dir)
        print(f"Copied selected videos to {args.media_dir}")


if __name__ == "__main__":
    main()
