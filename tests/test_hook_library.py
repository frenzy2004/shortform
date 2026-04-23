from pathlib import Path

from humeo.hook_library import (
    format_hook_examples,
    hook_library_fingerprint,
    load_hook_library,
    retrieve_hook_examples,
)


def test_load_hook_library_parses_markdown_directory(tmp_path):
    md = tmp_path / "Educational_Hooks.md"
    md.write_text(
        "# Educational Hooks\n\n"
        "1. Hook: Here’s why [belief] is a myth<br>"
        "Example: Here’s why traction is a myth.<br>"
        "Psychology: Belief violation creates curiosity.\n",
        encoding="utf-8",
    )

    items = load_hook_library(tmp_path)

    assert len(items) == 1
    assert items[0].category == "Educational"
    assert "traction is a myth" in items[0].example


def test_retrieve_hook_examples_ranks_by_query_overlap(tmp_path):
    (tmp_path / "Authority_and_Expertise_Hooks.md").write_text(
        "# Authority and Expertise Hooks\n\n"
        "1. Hook: The #1 way to ___ (backed by results)<br>"
        "Example: The #1 way to protect your focus (backed by data).<br>"
        "Psychology: Ranked certainty plus credibility.\n",
        encoding="utf-8",
    )
    (tmp_path / "Shocking_Truths_and_Discoveries.md").write_text(
        "# Shocking Truths and Discoveries Hooks\n\n"
        "1. Hook: What nobody tells you about ___<br>"
        "Example: What nobody tells you about prediction markets.<br>"
        "Psychology: Insider knowledge creates clicks.\n",
        encoding="utf-8",
    )

    results = retrieve_hook_examples(
        "prediction markets backed by data",
        topic="markets",
        path=tmp_path,
        limit=1,
    )

    assert len(results) == 1
    assert "prediction markets" in results[0].example.lower()


def test_hook_library_fingerprint_changes_with_content(tmp_path):
    md = tmp_path / "Storytelling_Hooks.md"
    md.write_text("1. Hook: A<br>Example: B<br>Psychology: C\n", encoding="utf-8")
    first = hook_library_fingerprint(tmp_path)
    md.write_text("1. Hook: A<br>Example: Changed<br>Psychology: C\n", encoding="utf-8")
    second = hook_library_fingerprint(tmp_path)
    assert first != second


def test_format_hook_examples_human_readable():
    tmp = Path(".")
    items = load_hook_library(None)
    assert items == []
    text = format_hook_examples([])
    assert text == ""
