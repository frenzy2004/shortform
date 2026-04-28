import json
from pathlib import Path

from humeo.best_of import build_best_of_review_pack


def _write_clip_plan(path: Path, clips: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"clips": ' + json.dumps(clips) + "}\n", encoding="utf-8")


def test_build_best_of_review_pack_picks_best_clip_per_source(tmp_path: Path):
    batch_root = tmp_path / "batch"
    source_a = batch_root / "videoplayback_7"
    source_b = batch_root / "videoplayback_8"
    source_a.mkdir(parents=True)
    source_b.mkdir(parents=True)

    for folder, names in (
        (source_a, ["short_001.mp4", "short_002.mp4"]),
        (source_b, ["short_001.mp4", "short_002.mp4"]),
    ):
        for name in names:
            (folder / name).write_bytes(b"mp4")

    work_a = tmp_path / ".humeo_batch_videoplayback7"
    work_b = tmp_path / ".humeo_batch_videoplayback8"
    _write_clip_plan(
        work_a / "clips.json",
        [
            {
                "clip_id": "001",
                "topic": "generic",
                "start_time_sec": 0.0,
                "end_time_sec": 60.0,
                "virality_score": 0.8,
                "suggested_overlay_title": "Big Opportunity",
                "viral_hook": "This changes everything",
            },
            {
                "clip_id": "002",
                "topic": "specific",
                "start_time_sec": 100.0,
                "end_time_sec": 160.0,
                "virality_score": 0.8,
                "suggested_overlay_title": "Robots Cut Costs 90%",
                "viral_hook": "Robots cut costs by 90%",
            },
        ],
    )
    _write_clip_plan(
        work_b / "clips.json",
        [
            {
                "clip_id": "001",
                "topic": "shorter",
                "start_time_sec": 0.0,
                "end_time_sec": 58.0,
                "virality_score": 0.77,
                "suggested_overlay_title": "Delivery Under $1",
                "viral_hook": "Delivery falls under $1",
            },
            {
                "clip_id": "002",
                "topic": "late hook",
                "start_time_sec": 100.0,
                "end_time_sec": 158.0,
                "virality_score": 0.8,
                "suggested_overlay_title": "Delivery Under $1",
                "viral_hook": "Delivery falls under $1",
                "hook_start_sec": 9.0,
                "hook_end_sec": 12.0,
            },
        ],
    )

    destination = tmp_path / "best_of"
    copied = build_best_of_review_pack(
        batch_root,
        destination,
        per_source=1,
        work_dir_map={
            "videoplayback_7": work_a,
            "videoplayback_8": work_b,
        },
    )

    copied_names = sorted(path.name for path in copied)
    assert copied_names == [
        "videoplayback_7__pick01__short_002.mp4",
        "videoplayback_8__pick01__short_001.mp4",
    ]
    assert (destination / "best_of_manifest.json").is_file()
