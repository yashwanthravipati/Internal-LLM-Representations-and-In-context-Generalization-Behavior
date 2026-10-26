"""Prompt templates. Each renders demos + query into text and records the character span of the
query item, so token positions (`query_item`, `decision`) are recovered exactly downstream.

Labels carry their leading space everywhere (" foo"); the template inserts them verbatim, and
they are scored at the final position.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Template:
    name: str
    demo: str  # format with {item} and {label}; {label} includes its leading space
    query: str  # format with {item}; the prompt ends right before the label's leading space
    header: str = ""  # optional preamble (used for the explicit-rule control)

    def render(
        self, demos: list[tuple[str, str]], query_item: str, header: str | None = None
    ) -> tuple[str, tuple[int, int]]:
        """Returns (text, (start, end) char span of the query item)."""
        text = header if header is not None else self.header
        for item, label in demos:
            text += self.demo.format(item=item, label=label)
        pre, post = self.query.split("{item}")
        start = len(text) + len(pre)
        text += pre + query_item + post
        return text, (start, start + len(query_item))


TEMPLATES = {
    # Items start a line (no leading space): most words become 2+ tokens in Qwen3.5.
    "arrow": Template("arrow", demo="{item} ->{label}\n", query="{item} ->"),
    # Items follow a space: ~2x more single-token items.
    "io": Template("io", demo="Input: {item}\nOutput:{label}\n\n", query="Input: {item}\nOutput:"),
}
