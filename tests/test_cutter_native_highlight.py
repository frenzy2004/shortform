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
