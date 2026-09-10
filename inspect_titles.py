import json
from pathlib import Path

def show_titles(source_dir, filter_fn=None):
    path = Path("data/raw") / source_dir
    for f in sorted(path.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        metadata = data.get("metadata", {})
        cat = metadata.get("category")
        tags = metadata.get("tags", [])
        if filter_fn and not filter_fn(cat, tags):
            continue
        print(f"  [{f.name}] {data.get('title', '(no title)')}")

print("=== Proxmox: 'Cluster Filesystem' and 'General Troubleshooting' ===")
show_titles("proxmox", lambda cat, tags: cat in ("Cluster Filesystem", "General Troubleshooting"))

print("\n=== ServerFault: 'proxmox' tag ===")
show_titles("serverfault", lambda cat, tags: "proxmox" in tags)

print("\n=== ServerFault: 'windows-server-2019' tag ===")
show_titles("serverfault", lambda cat, tags: "windows-server-2019" in tags)