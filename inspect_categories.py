import json
from pathlib import Path
from collections import Counter

RAW_DIR = Path("data/raw")

for source_dir in ["mslearn", "proxmox", "serverfault"]:
    path = RAW_DIR / source_dir
    if not path.exists():
        print(f"\n{source_dir}: directory not found, skipping")
        continue

    print(f"\n=== {source_dir} ===")
    counter = Counter()
    for f in path.glob("*.json"):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        metadata = data.get("metadata", {})
        # Try both "category" and "tags" since sources differ
        cat = metadata.get("category")
        tags = metadata.get("tags")
        if cat:
            counter[cat] += 1
        elif tags:
            for t in tags:
                counter[t] += 1
        else:
            counter["<no category/tags found>"] += 1

    for label, count in counter.most_common():
        print(f"  {label}: {count}")

print("\n=== Existing ticket categories (from tickets.json) ===")
tickets_path = Path("data/synthetic_tickets/tickets.json")
if tickets_path.exists():
    with open(tickets_path, "r", encoding="utf-8") as f:
        tickets = json.load(f)
    ticket_cats = Counter(t["category"] for t in tickets)
    for label, count in ticket_cats.most_common():
        print(f"  {label}: {count}")