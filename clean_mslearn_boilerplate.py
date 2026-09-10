"""
One-time cleanup: strip website-chrome boilerplate junk that got scraped
into the start of every data/raw/mslearn/*.json content field. Fixes both
the QLoRA classifier training data and the underlying RAG corpus quality.

Uses line-pattern matching instead of exact-string matching, since the
first attempt's exact strings didn't match the real whitespace/formatting.
"""

import json
import re
from pathlib import Path

MSLEARN_DIR = Path("data/raw/mslearn")

# Any line containing one of these markers is website-chrome junk, not
# real article content - drop the whole line.
JUNK_LINE_MARKERS = [
    "privateUnauthorizedTemplate",
    "requires authorization",
    "Slot where the client will render",
    "[Read in English]",
    "[Edit](https://github.com/MicrosoftDocs",
]


def clean_content(content: str, title: str) -> str:
    lines = content.split("\n")
    kept_lines = []
    for line in lines:
        if any(marker in line for marker in JUNK_LINE_MARKERS):
            continue
        kept_lines.append(line)
    cleaned = "\n".join(kept_lines)

    # Collapse a duplicated title heading (appears twice in the original
    # scrape) down to one occurrence.
    title_line = f"# {title}"
    occurrences = [m.start() for m in re.finditer(re.escape(title_line), cleaned)]
    if len(occurrences) > 1:
        first_end = occurrences[0] + len(title_line)
        second_start = occurrences[1]
        cleaned = cleaned[:first_end] + cleaned[second_start + len(title_line):]

    # Collapse resulting multiple blank lines down to at most one
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def main():
    cleaned_count = 0
    for f in sorted(MSLEARN_DIR.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        original = data["content"]
        cleaned = clean_content(original, data.get("title", ""))

        if cleaned != original:
            data["content"] = cleaned
            with open(f, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            cleaned_count += 1
            print(f"Cleaned {f.name}: {len(original)} -> {len(cleaned)} chars")

    print(f"\nDone. Cleaned {cleaned_count} files.")


if __name__ == "__main__":
    main()