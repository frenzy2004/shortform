"""layout_vision parsing (no API calls)."""

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from humeo.layout_vision import (
    _call_gemini_vision,
    _face_center_x,
    _instruction_from_gemini_json,
    _tracking_points_from_centers,
    infer_layout_instructions,
)
from humeo_core.schemas import BoundingBox, LayoutKind, Scene


def test_instruction_from_gemini_json_split_with_bboxes():
    data = {
        "layout": "split_chart_person",
        "person_bbox": {"x1": 0.62, "y1": 0.05, "x2": 0.99, "y2": 0.95},
        "chart_bbox": {"x1": 0.02, "y1": 0.05, "x2": 0.58, "y2": 0.92},
        "reason": "webinar",
    }
    instr = _instruction_from_gemini_json("005", data)
    assert instr.layout == LayoutKind.SPLIT_CHART_PERSON
    assert instr.split_chart_region is not None
    assert instr.split_person_region is not None


def test_instruction_from_gemini_json_split_with_pixel_bboxes():
    data = {
        "layout": "split_chart_person",
        "person_bbox": {"x1": 400, "y1": 40, "x2": 620, "y2": 340},
        "face_bbox": {"x1": 450, "y1": 40, "x2": 540, "y2": 140},
        "chart_bbox": {"x1": 20, "y1": 25, "x2": 360, "y2": 250},
        "reason": "speaker right, chart left",
    }
    instr = _instruction_from_gemini_json("005", data, image_size=(640, 360))
    assert instr.layout == LayoutKind.SPLIT_CHART_PERSON
    assert instr.split_chart_region is not None
    assert instr.split_person_region is not None
    assert math.isclose(instr.split_chart_region.x1, 20 / 640, abs_tol=1e-6)
    assert math.isclose(instr.split_person_region.x1, 400 / 640, abs_tol=1e-6)


def test_instruction_from_gemini_json_split_with_thousand_grid_bboxes():
    data = {
        "layout": "split_chart_person",
        "person_bbox": {"x1": 508, "y1": 66, "x2": 999, "y2": 1000},
        "face_bbox": {"x1": 692, "y1": 66, "x2": 866, "y2": 314},
        "chart_bbox": {"x1": 20, "y1": 28, "x2": 581, "y2": 698},
        "reason": "0..1000 pseudo-normalized values",
    }
    instr = _instruction_from_gemini_json("004", data, image_size=(640, 360))
    assert instr.layout == LayoutKind.SPLIT_CHART_PERSON
    assert instr.split_chart_region is not None
    assert instr.split_person_region is not None
    assert math.isclose(instr.split_person_region.x1, 0.508, abs_tol=1e-6)
    assert math.isclose(instr.split_chart_region.x2, 0.581, abs_tol=1e-6)
    assert math.isclose(instr.person_x_norm, 0.779, abs_tol=1e-6)


def test_instruction_from_gemini_json_split_with_mixed_bbox_units():
    data = {
        "layout": "split_chart_person",
        "person_bbox": {"x1": 0.585, "y1": 65, "x2": 0.985, "y2": 985},
        "face_bbox": {"x1": 0.685, "y1": 0.065, "x2": 0.885, "y2": 350},
        "chart_bbox": {"x1": 0.02, "y1": 30, "x2": 0.58, "y2": 0.69},
        "reason": "mixed normalized/pixel values",
    }
    instr = _instruction_from_gemini_json("004", data, image_size=(1000, 1000))
    assert instr.layout == LayoutKind.SPLIT_CHART_PERSON
    assert instr.split_chart_region is not None
    assert instr.split_person_region is not None
    assert math.isclose(instr.split_person_region.y1, 0.065, abs_tol=1e-6)
    assert math.isclose(instr.split_chart_region.y1, 0.03, abs_tol=1e-6)
    assert math.isclose(instr.person_x_norm, 0.785, abs_tol=1e-6)


def test_instruction_from_gemini_json_split_two_persons_mixed_bbox_falls_back_to_sit_center():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "ticket_c_videoplayback3_short002_layout.json").read_text(
            encoding="utf-8"
        )
    )

    instr = _instruction_from_gemini_json(
        fixture["scene_id"],
        fixture["data"],
        image_size=tuple(fixture["image_size"]),
    )

    assert instr.layout == LayoutKind.SIT_CENTER
    assert instr.split_person_region is None
    assert instr.split_second_person_region is None
    assert math.isclose(instr.person_x_norm, 0.76, abs_tol=1e-6)


def test_instruction_from_gemini_json_split_two_persons_consistent_boxes_preserved():
    data = {
        "layout": "split_two_persons",
        "person_bbox": {"x1": 0.12, "y1": 0.18, "x2": 0.36, "y2": 0.92},
        "face_bbox": {"x1": 0.18, "y1": 0.22, "x2": 0.28, "y2": 0.38},
        "second_person_bbox": {"x1": 0.64, "y1": 0.18, "x2": 0.88, "y2": 0.92},
        "second_face_bbox": {"x1": 0.71, "y1": 0.22, "x2": 0.81, "y2": 0.38},
        "reason": "clean two-up shot",
    }

    instr = _instruction_from_gemini_json("006", data, image_size=(640, 360))

    assert instr.layout == LayoutKind.SPLIT_TWO_PERSONS
    assert instr.split_person_region is not None
    assert instr.split_second_person_region is not None
    assert math.isclose(instr.split_person_region.x1, 0.12, abs_tol=1e-6)
    assert math.isclose(instr.split_second_person_region.x1, 0.64, abs_tol=1e-6)


def test_instruction_from_gemini_json_sit_center():
    data = {
        "layout": "sit_center",
        "person_bbox": {"x1": 0.3, "y1": 0.1, "x2": 0.7, "y2": 0.9},
        "chart_bbox": None,
        "reason": "talking head",
    }
    instr = _instruction_from_gemini_json("001", data)
    assert instr.layout == LayoutKind.SIT_CENTER
    assert instr.split_chart_region is None


def test_face_bbox_pulls_person_x_norm_toward_the_face():
    """Regression for the off-center subject bug.

    Reproduces clip 001 from the Dr. Mike failing run: subject sitting in
    profile, head around x≈0.23, tank top + arm extend the body bbox out
    to x2=0.75. The wide person_bbox center alone gave person_x_norm=0.415,
    which cropped the final 9:16 short on the torso and pushed the face off
    the left edge. With the face_bbox hint, person_x_norm must track the
    face instead.
    """
    data = {
        "layout": "sit_center",
        "person_bbox": {"x1": 0.08, "y1": 0.10, "x2": 0.75, "y2": 0.95},
        "face_bbox":   {"x1": 0.18, "y1": 0.12, "x2": 0.30, "y2": 0.32},
        "chart_bbox": None,
        "reason": "profile speaker off-center left",
    }
    instr = _instruction_from_gemini_json("001", data)
    assert instr.layout == LayoutKind.SIT_CENTER
    # Face center is 0.24. person-bbox center is 0.415. Must follow the face.
    assert math.isclose(instr.person_x_norm, 0.24, abs_tol=1e-6), (
        f"person_x_norm should track face center (0.24), got {instr.person_x_norm}"
    )


def test_face_bbox_missing_falls_back_to_person_bbox_center():
    data = {
        "layout": "sit_center",
        "person_bbox": {"x1": 0.30, "y1": 0.10, "x2": 0.70, "y2": 0.90},
        "face_bbox": None,
        "chart_bbox": None,
        "reason": "centered talking head",
    }
    instr = _instruction_from_gemini_json("002", data)
    assert math.isclose(instr.person_x_norm, 0.50, abs_tol=1e-6)


def test_face_bbox_rejected_when_as_wide_as_person_bbox():
    """If Gemini echoes the person bbox into face_bbox we get no new info.

    In that case fall back to the person-bbox center, not a spurious face
    center — we don't want the "fix" to regress the centered case.
    """
    data = {
        "layout": "sit_center",
        "person_bbox": {"x1": 0.10, "y1": 0.10, "x2": 0.90, "y2": 0.95},
        "face_bbox":   {"x1": 0.10, "y1": 0.10, "x2": 0.90, "y2": 0.95},
        "chart_bbox": None,
        "reason": "echoed bbox",
    }
    instr = _instruction_from_gemini_json("003", data)
    # Fall back to person-bbox center (0.5) — face_bbox too wide to trust.
    assert math.isclose(instr.person_x_norm, 0.50, abs_tol=1e-6)


def test_face_bbox_outside_person_bbox_is_ignored():
    """If face_bbox center sits outside person_bbox the model got confused."""
    data = {
        "layout": "sit_center",
        "person_bbox": {"x1": 0.60, "y1": 0.10, "x2": 0.95, "y2": 0.95},
        "face_bbox":   {"x1": 0.05, "y1": 0.10, "x2": 0.15, "y2": 0.25},
        "chart_bbox": None,
        "reason": "mismatched face and person",
    }
    instr = _instruction_from_gemini_json("004", data)
    # Person bbox center = 0.775; we must not jump to face center (0.10).
    assert math.isclose(instr.person_x_norm, 0.775, abs_tol=1e-6)


def test_face_center_helper_unit():
    # Clean case: tight face inside the body.
    face = BoundingBox(x1=0.20, y1=0.10, x2=0.30, y2=0.25)
    body = BoundingBox(x1=0.10, y1=0.10, x2=0.70, y2=0.95)
    assert _face_center_x(face, body) == 0.25

    # No face.
    assert _face_center_x(None, body) is None

    # Face suspiciously wide (> 40% of frame): ignore.
    wide = BoundingBox(x1=0.10, y1=0.10, x2=0.60, y2=0.95)
    assert _face_center_x(wide, body) is None


@patch("humeo.layout_vision.OpenAI")
def test_call_gemini_vision_uses_openrouter_image_payload(mock_openai_cls, monkeypatch, tmp_path):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")

    mock_inst = MagicMock()
    mock_openai_cls.return_value = mock_inst
    mock_inst.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"layout":"sit_center"}'))]
    )

    out = _call_gemini_vision(str(image_path), "gemini-3.1-flash-lite-preview")

    assert out["layout"] == "sit_center"
    mock_openai_cls.assert_called_once()
    call_kwargs = mock_inst.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "google/gemini-3.1-flash-lite-preview"
    content = call_kwargs["messages"][1]["content"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_tracking_points_filter_obvious_outlier_sample():
    points = _tracking_points_from_centers(
        4.0,
        [
            (0.0, 0.20),
            (1.0, 0.25),
            (2.0, 0.75),
            (3.0, 0.30),
            (4.0, 0.35),
        ],
    )
    xs = [point.x_norm for point in points]
    assert max(xs) < 0.5
    assert points[2].x_norm == pytest.approx((0.25 + 0.30) / 2.0)


@patch("humeo.layout_vision._call_gemini_vision")
@patch("humeo.layout_vision._extract_frame_at_time")
def test_infer_layout_instructions_adds_person_tracking(
    mock_extract_frame_at_time,
    mock_call_gemini_vision,
    tmp_path,
):
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"fake")
    keyframe_path = tmp_path / "001.jpg"
    keyframe_path.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")

    scene = Scene(scene_id="001", start_time=10.0, end_time=20.0, keyframe_path=str(keyframe_path))

    def fake_extract(source_path: Path, time_sec: float, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
        return output_path

    def fake_vision(path: str, model_name: str) -> dict[str, object]:
        name = Path(path).name
        if name == "001.jpg":
            return {
                "layout": "sit_center",
                "person_bbox": {"x1": 0.10, "y1": 0.10, "x2": 0.40, "y2": 0.95},
                "face_bbox": {"x1": 0.16, "y1": 0.12, "x2": 0.26, "y2": 0.28},
                "reason": "speaker left at midpoint",
            }
        if name.endswith("001000.jpg"):
            face = {"x1": 0.08, "y1": 0.12, "x2": 0.18, "y2": 0.28}
        elif name.endswith("003000.jpg"):
            face = {"x1": 0.22, "y1": 0.12, "x2": 0.32, "y2": 0.28}
        elif name.endswith("007000.jpg"):
            face = {"x1": 0.56, "y1": 0.12, "x2": 0.66, "y2": 0.28}
        else:
            face = {"x1": 0.70, "y1": 0.12, "x2": 0.80, "y2": 0.28}
        return {
            "layout": "sit_center",
            "person_bbox": {"x1": max(0.0, face["x1"] - 0.08), "y1": 0.10, "x2": min(1.0, face["x2"] + 0.12), "y2": 0.95},
            "face_bbox": face,
            "reason": "tracked speaker",
        }

    mock_extract_frame_at_time.side_effect = fake_extract
    mock_call_gemini_vision.side_effect = fake_vision

    instructions, raw_by_clip = infer_layout_instructions(
        [scene],
        gemini_vision_model="gemini-test",
        source_video=source_video,
        tracking_dir=tmp_path / "tracking",
    )

    instr = instructions["001"]
    assert instr.layout == LayoutKind.SIT_CENTER
    assert len(instr.person_tracking) >= 4
    assert instr.person_tracking[0].t_sec == 0.0
    assert instr.person_tracking[-1].t_sec == 10.0
    assert instr.person_tracking[0].x_norm < instr.person_tracking[-1].x_norm

    samples = raw_by_clip["001"]["person_tracking_samples"]
    assert samples
    assert samples[0]["sample_kind"] == "midpoint_keyframe"


@patch("humeo.layout_vision._keyframe_dimensions", return_value=(640, 360))
@patch("humeo.layout_vision._call_gemini_vision")
@patch("humeo.layout_vision._extract_frame_at_time")
def test_infer_layout_instructions_ignores_split_tracking_sample_for_single_person_midpoint(
    mock_extract_frame_at_time,
    mock_call_gemini_vision,
    _mock_keyframe_dimensions,
    tmp_path,
):
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "ticket_c_videoplayback4_short003_tracking.json").read_text(
            encoding="utf-8"
        )
    )

    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"fake")
    keyframe_path = tmp_path / fixture["scene"]["keyframe_name"]
    keyframe_path.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")

    scene = Scene(
        scene_id=fixture["scene"]["scene_id"],
        start_time=fixture["scene"]["start_time"],
        end_time=fixture["scene"]["end_time"],
        keyframe_path=str(keyframe_path),
    )

    def fake_extract(source_path: Path, time_sec: float, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
        return output_path

    def fake_vision(path: str, model_name: str) -> dict[str, object]:
        name = Path(path).name
        if name == fixture["scene"]["keyframe_name"]:
            return fixture["midpoint_data"]
        return fixture["tracking_samples"][name]

    mock_extract_frame_at_time.side_effect = fake_extract
    mock_call_gemini_vision.side_effect = fake_vision

    instructions, raw_by_clip = infer_layout_instructions(
        [scene],
        gemini_vision_model="gemini-test",
        source_video=source_video,
        tracking_dir=tmp_path / "tracking",
    )

    instr = instructions["003"]
    assert instr.layout == LayoutKind.SIT_CENTER
    assert instr.person_tracking
    assert instr.person_tracking[0].t_sec == 0.0
    assert instr.person_tracking[0].x_norm == pytest.approx(0.425, abs=0.02)
    assert min(point.x_norm for point in instr.person_tracking) > 0.35

    samples = raw_by_clip["003"]["person_tracking_samples"]
    split_sample = next(
        sample
        for sample in samples
        if sample["sample_kind"] == "tracking_frame"
        and isinstance(sample.get("raw"), dict)
        and sample["raw"].get("layout") == "split_two_persons"
    )
    assert split_sample["center_x_norm"] is None
