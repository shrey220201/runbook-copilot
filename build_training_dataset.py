"""
Build the training dataset for the QLoRA incident-category classifier.

Pulls category-labeled text from:
1. Ingested documents (data/raw/mslearn, proxmox, serverfault, nakivo)
   - mapped from their source-specific category/tag labels to our
     5-category taxonomy (active-directory, virtualization, networking,
     storage, backup)
2. Synthetic tickets (data/synthetic_tickets/tickets.json) - already
   labeled with the target taxonomy

Excludes a small number of documents whose source tag doesn't reliably
reflect their actual topic (see EXCLUDE_IDS below) rather than force-fit
a mislabel.

Output: data/qlora_training/classifier_dataset.jsonl
Each line: {"text": "<title>. <first ~400 chars of content>", "label": "<category>"}
"""

import html
import json
from pathlib import Path

RAW_DIR = Path("data/raw")
TICKETS_PATH = Path("data/synthetic_tickets/tickets.json")
OUTPUT_PATH = Path("data/qlora_training/classifier_dataset.jsonl")

# Documents whose tag doesn't reliably reflect actual topic - excluded
# rather than mislabeled.
EXCLUDE_IDS = {
    "serverfault-941855",  # missing /var/run/sshd - generic Linux boot issue
    "serverfault-971937",  # automatic Windows Update installation - generic
    "serverfault-980387",  # Shutdown Event Tracker on login - generic
}

# Direct category-string -> taxonomy mapping (mslearn, proxmox)
CATEGORY_MAP = {
    "Active Directory": "active-directory",
    "Hyper-V Virtualization": "virtualization",
    "Networking & DNS": "networking",
    "Backup & Disaster Recovery": "backup",
    "Cluster Filesystem": "virtualization",
    "Cluster & Corosync": "virtualization",
    "Storage & Hardware": "storage",
    "Firewall & Security": "networking",
    "High Availability": "virtualization",
    "Virtual Machines (KVM)": "virtualization",
    "LXC Containers": "virtualization",
    "Networking": "networking",
    "Cluster Networking": "networking",
    "Storage": "storage",
    "General Troubleshooting": "virtualization",
    "ZFS Storage": "storage",
}

# ServerFault tag -> taxonomy mapping (used when category isn't a direct
# match above)
TAG_MAP = {
    "active-directory": "active-directory",
    "hyper-v": "virtualization",
    "networking": "networking",
    "windows-server-2019": None,  # too generic alone, handled per-file below
    "proxmox": None,  # too generic alone, handled per-file below
}

# Per-file overrides for ambiguous serverfault docs, based on manual title review
SERVERFAULT_OVERRIDES = {
    "serverfault-1021251": "storage",       # FIO benchmarking / RAID misconfig
    "serverfault-618408": "networking",     # Proxmox NAT networking issue
    "serverfault-731400": "virtualization", # migrate LXC container to Proxmox
    "serverfault-950794": "storage",        # limit ZFS writes on NVMe RAID1
    "serverfault-1031988": "storage",       # Windows 2019 drive full
    "serverfault-1067632": "active-directory",  # force DC AD DNS re-registration
    "serverfault-1166382": "networking",    # OpenSSH server won't start
}


def truncate(text: str, max_chars: int = 400) -> str:
    text = html.unescape(text)  # fix HTML entities like &#39; -> '
    text = " ".join(text.split())  # collapse whitespace/newlines
    return text[:max_chars]


def load_ingested_docs():
    examples = []
    skipped = []

    for source_dir in ["mslearn", "proxmox", "serverfault", "nakivo"]:
        path = RAW_DIR / source_dir
        if not path.exists():
            continue

        for f in sorted(path.glob("*.json")):
            doc_id = f.stem
            if doc_id in EXCLUDE_IDS:
                skipped.append((doc_id, "excluded (generic/ambiguous tag)"))
                continue

            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            metadata = data.get("metadata", {})
            category = metadata.get("category")
            tags = metadata.get("tags", [])

            label = None
            if doc_id in SERVERFAULT_OVERRIDES:
                label = SERVERFAULT_OVERRIDES[doc_id]
            elif category in CATEGORY_MAP:
                label = CATEGORY_MAP[category]
            elif tags:
                for t in tags:
                    if TAG_MAP.get(t):
                        label = TAG_MAP[t]
                        break

            if not label:
                skipped.append((doc_id, f"no mapping found (category={category}, tags={tags})"))
                continue

            title = html.unescape(data.get("title", ""))
            text = f"{title}. {truncate(data.get('content', ''))}"
            examples.append({"text": text, "label": label})

    return examples, skipped


def load_tickets():
    examples = []
    if not TICKETS_PATH.exists():
        return examples
    with open(TICKETS_PATH, "r", encoding="utf-8") as f:
        tickets = json.load(f)
    for t in tickets:
        examples.append({"text": t["incident_description"], "label": t["category"]})
    return examples


def main():
    doc_examples, skipped = load_ingested_docs()
    ticket_examples = load_tickets()
    all_examples = doc_examples + ticket_examples

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(all_examples)} labeled examples to {OUTPUT_PATH}")
    print(f"  ({len(doc_examples)} from ingested docs, {len(ticket_examples)} from tickets)")

    if skipped:
        print(f"\nSkipped {len(skipped)} documents:")
        for doc_id, reason in skipped:
            print(f"  {doc_id}: {reason}")

    # Print label distribution
    from collections import Counter
    label_counts = Counter(ex["label"] for ex in all_examples)
    print("\nLabel distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()