from pathlib import Path

from humeo.cutter import generate_ass
from humeo_core.schemas import Clip, RenderTheme


def test_generate_ass_native_highlight_emits_overlay_styles(tmp_path: Path):
    clip = Clip(
        clip_id="001",
        topic="t",
        start_time_sec=0.0,
        end_time_sec=2.4,
        transcript="Did you know there are already",
    )
    transcript = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.4,
                "text": "Did you know there are already",
                "words": [
                    {"word": "Did", "start": 0.0, "end": 0.3},
                    {"word": "you", "start": 0.3, "end": 0.6},
                    {"word": "know", "start": 0.6, "end": 0.9},
                    {"word": "there", "start": 0.9, "end": 1.3},
                    {"word": "are", "start": 1.3, "end": 1.6},
                    {"word": "already", "start": 1.6, "end": 2.1},
                ],
            }
        ]
    }

    ass_path = generate_ass(
        clip,
        transcript,
        tmp_path,
        play_res_x=1080,
        play_res_y=1920,
        render_theme=RenderTheme.NATIVE_HIGHLIGHT,
    )

    text = ass_path.read_text(encoding="utf-8")
    assert "Style: Base,Arial" in text
    assert "Style: Highlight,Arial" in text
    assert r"\an7\pos(" in text
    assert r"\blur0.8" in text


def test_generate_ass_native_highlight_moves_word_by_word(tmp_path: Path):
    clip = Clip(
        clip_id="001",
        topic="t",
        start_time_sec=0.0,
        end_time_sec=2.2,
        transcript="to all of the innovation taking",
    )
    transcript = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.2,
                "text": "to all of the innovation taking",
                "words": [
                    {"word": "to", "start": 0.0, "end": 0.2},
                    {"word": "all", "start": 0.2, "end": 0.4},
                    {"word": "of", "start": 0.4, "end": 0.6},
                    {"word": "the", "start": 0.6, "end": 0.8},
                    {"word": "innovation", "start": 0.8, "end": 1.3},
                    {"word": "taking", "start": 1.3, "end": 1.7},
                ],
            }
        ]
    }

    ass_path = generate_ass(
        clip,
        transcript,
        tmp_path,
        play_res_x=1080,
        play_res_y=1920,
        render_theme=RenderTheme.NATIVE_HIGHLIGHT,
    )

    highlight_lines = [
        line for line in ass_path.read_text(encoding="utf-8").splitlines() if line.startswith("Dialogue: 1,")
    ]
    assert len(highlight_lines) >= 3
    assert any("innovation" in line.lower() for line in highlight_lines)
    assert any("taking" in line.lower() for line in highlight_lines)
    assert any(line.split(",")[1] != line.split(",")[2] for line in highlight_lines)


def test_generate_ass_native_highlight_never_groups_awkward_pairs(tmp_path: Path):
    clip = Clip(
        clip_id="001",
        topic="t",
        start_time_sec=0.0,
        end_time_sec=2.0,
        transcript="companies their productivity",
    )
    transcript = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "companies their productivity",
                "words": [
                    {"word": "companies,", "start": 0.0, "end": 0.5},
                    {"word": "their", "start": 0.5, "end": 0.8},
                    {"word": "productivity,", "start": 0.8, "end": 1.5},
                ],
            }
        ]
    }

    ass_path = generate_ass(
        clip,
        transcript,
        tmp_path,
        play_res_x=1080,
        play_res_y=1920,
        render_theme=RenderTheme.NATIVE_HIGHLIGHT,
    )

    highlight_lines = [
        line for line in ass_path.read_text(encoding="utf-8").splitlines() if line.startswith("Dialogue: 1,")
    ]
    assert not any("companies, their" in line.lower() for line in highlight_lines)
    assert any("companies" in line.lower() or "productivity" in line.lower() for line in highlight_lines)
