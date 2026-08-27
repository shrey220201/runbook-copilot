"""Data loaders for Runbook Copilot sources.

Fetches and parses real documentation from:
1. Microsoft Learn technical troubleshooting guides (Active Directory, Hyper-V, Networking)
2. Proxmox VE Wiki administration and troubleshooting pages
3. ServerFault Stack Exchange real questions and accepted answers via Stack Exchange API v2.3

Each loader is independently runnable via CLI and includes rate-limiting, politeness delays,
and graceful exception handling.
"""

import os
import re
import json
import time
import argparse
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup, NavigableString, Tag

from config import RAW_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_REQUEST_HEADERS = {
    "User-Agent": "RunbookCopilotResearchBot/1.0 (Educational RAG Project; DevOps Troubleshooting Runbooks)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


@dataclass
class RawDocument:
    id: str
    title: str
    content: str
    source: str
    url: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def clean_output_directory(dir_path: Path) -> int:
    """Ensure output directory exists and delete existing JSON files to prevent stale artifacts."""
    dir_path.mkdir(parents=True, exist_ok=True)
    deleted_count = 0
    for file_path in dir_path.glob("*.json"):
        try:
            file_path.unlink()
            deleted_count += 1
        except OSError as e:
            logger.warning(f"Could not delete stale file {file_path}: {e}")
    if deleted_count > 0:
        logger.info(f"Cleared {deleted_count} stale .json file(s) from {dir_path}")
    return deleted_count


# =====================================================================
# HTML to Markdown Conversion Helper
# =====================================================================

def html_element_to_markdown(elem: Tag) -> str:
    """Convert an HTML element tree into clean, structured Markdown."""
    if elem is None:
        return ""

    # Remove non-content elements
    for unwanted in elem.select("script, style, noscript, iframe, svg, button, form, .visually-hidden, .sr-only, .display-none"):
        unwanted.decompose()

    def process_node(node) -> str:
        if isinstance(node, NavigableString):
            text = str(node)
            # Normalize whitespace within inline text
            return re.sub(r"[ \t]+", " ", text)

        if not isinstance(node, Tag):
            return ""

        tag_name = node.name.lower()

        # Headers
        if tag_name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            level = int(tag_name[1])
            heading_text = "".join(process_node(child) for child in node.children).strip()
            if heading_text:
                return f"\n\n{'#' * level} {heading_text}\n\n"
            return ""

        # Code Blocks
        if tag_name == "pre":
            code_tag = node.find("code")
            lang = ""
            if code_tag:
                classes = code_tag.get("class", [])
                if isinstance(classes, list):
                    for c in classes:
                        if c.startswith("language-") or c.startswith("lang-"):
                            lang = c.replace("language-", "").replace("lang-", "")
                            break
                code_text = code_tag.get_text()
            else:
                code_text = node.get_text()
            return f"\n\n```{lang}\n{code_text.strip()}\n```\n\n"

        # Inline Code
        if tag_name == "code":
            inline_text = node.get_text()
            return f"`{inline_text}`"

        # Paragraphs
        if tag_name == "p":
            p_text = "".join(process_node(child) for child in node.children).strip()
            if p_text:
                return f"\n\n{p_text}\n\n"
            return ""

        # Lists
        if tag_name in ["ul", "ol"]:
            list_items = []
            is_ordered = tag_name == "ol"
            idx = 1
            for child in node.find_all("li", recursive=False):
                item_text = "".join(process_node(c) for c in child.children).strip()
                if item_text:
                    prefix = f"{idx}." if is_ordered else "-"
                    list_items.append(f"{prefix} {item_text}")
                    idx += 1
            if list_items:
                return "\n\n" + "\n".join(list_items) + "\n\n"
            return ""

        # Blockquotes / Alerts
        if tag_name == "blockquote":
            quote_text = "".join(process_node(child) for child in node.children).strip()
            if quote_text:
                lines = quote_text.splitlines()
                quoted = "\n".join(f"> {line}" for line in lines if line.strip())
                return f"\n\n{quoted}\n\n"
            return ""

        # Tables
        if tag_name == "table":
            rows = []
            for tr in node.find_all("tr"):
                cells = [c.get_text().strip().replace("\n", " ") for c in tr.find_all(["th", "td"])]
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                header = rows[0]
                divider = " | ".join(["---"] * len(rows[0].split(" | ")))
                body = "\n".join(rows[1:]) if len(rows) > 1 else ""
                table_md = f"\n\n{header}\n{divider}"
                if body:
                    table_md += f"\n{body}"
                return table_md + "\n\n"
            return ""

        # Bold / Italic / Links
        if tag_name in ["strong", "b"]:
            inner = "".join(process_node(c) for c in node.children).strip()
            return f"**{inner}**" if inner else ""
        if tag_name in ["em", "i"]:
            inner = "".join(process_node(c) for c in node.children).strip()
            return f"*{inner}*" if inner else ""
        if tag_name == "a":
            inner = "".join(process_node(c) for c in node.children).strip()
            href = node.get("href", "")
            return f"[{inner}]({href})" if inner and href else inner

        # Generic container tags (div, section, article, span)
        return "".join(process_node(child) for child in node.children)

    raw_md = process_node(elem)
    # Clean excessive newlines and trailing spaces
    cleaned = re.sub(r"\n{3,}", "\n\n", raw_md).strip()
    return cleaned


# =====================================================================
# 1. Microsoft Learn Docs Loader
# =====================================================================

MSLEARN_URLS = [
    # Active Directory (7)
    {
        "id": "mslearn-ad-0xc00002e1-error",
        "url": "https://learn.microsoft.com/en-us/troubleshoot/windows-server/active-directory/0xc00002e1-error-start-domain-controller",
        "category": "Active Directory",
    },
    {
        "id": "mslearn-ad-event-4780-pdc",
        "url": "https://learn.microsoft.com/en-us/troubleshoot/windows-server/active-directory/a-batch-of-event-4780-logged-pdc",
        "category": "Active Directory",
    },
    {
        "id": "mslearn-ad-abandoned-objects-gc",
        "url": "https://learn.microsoft.com/en-us/troubleshoot/windows-server/active-directory/abandoned-objects-block-replication-global-catalog-naming-contexts",
        "category": "Active Directory",
    },
    {
        "id": "mslearn-ad-access-denied-ntds-settings",
        "url": "https://learn.microsoft.com/en-us/troubleshoot/windows-server/active-directory/access-denied-create-ntds-settings-object",
        "category": "Active Directory",
    },
    {
        "id": "mslearn-ad-access-denied-dcpromo",
        "url": "https://learn.microsoft.com/en-us/troubleshoot/windows-server/active-directory/access-denied-error-occurs-dcpromo",
        "category": "Active Directory",
    },
    {
        "id": "mslearn-ad-access-denied-domain-admin",
        "url": "https://learn.microsoft.com/en-us/troubleshoot/windows-server/active-directory/access-denied-log-on-administrator-domain-account",
        "category": "Active Directory",
    },
    {
        "id": "mslearn-ad-access-denied-join-domain",
        "url": "https://learn.microsoft.com/en-us/troubleshoot/windows-server/active-directory/access-denied-when-joining-computers",
        "category": "Active Directory",
    },
    # Hyper-V & Virtualization (7)
    {
        "id": "mslearn-hyperv-antivirus-exclusions",
        "url": "https://learn.microsoft.com/en-us/troubleshoot/windows-server/virtualization/antivirus-exclusions-for-hyper-v-hosts",
        "category": "Hyper-V Virtualization",
    },
    {
        "id": "mslearn-hyperv-backup-parent-partition",
        "url": "https://learn.microsoft.com/en-us/troubleshoot/windows-server/virtualization/back-up-hyper-v-vm-from-parent-partition",
        "category": "Hyper-V Virtualization",
    },
    {
        "id": "mslearn-hyperv-backup-restore-bmr",
        "url": "https://learn.microsoft.com/en-us/troubleshoot/windows-server/virtualization/back-up-restore-hyper-v-vm-bmr-data-backup",
        "category": "Hyper-V Virtualization",
    },
    {
        "id": "mslearn-hyperv-bios-update-requirements",
        "url": "https://learn.microsoft.com/en-us/troubleshoot/windows-server/virtualization/bios-update-for-hyper-v",
        "category": "Hyper-V Virtualization",
    },
    {
        "id": "mslearn-hyperv-block-virtualization-features",
        "url": "https://learn.microsoft.com/en-us/troubleshoot/windows-server/virtualization/block-users-from-running-virtualization-features-on-specific-computers",
        "category": "Hyper-V Virtualization",
    },
    {
        "id": "mslearn-hyperv-cannot-access-vmconnect",
        "url": "https://learn.microsoft.com/en-us/troubleshoot/windows-server/virtualization/cannot-access-hyper-v-virtual-machine-connect",
        "category": "Hyper-V Virtualization",
    },
    {
        "id": "mslearn-hyperv-delete-recovery-checkpoint",
        "url": "https://learn.microsoft.com/en-us/troubleshoot/windows-server/virtualization/cannot-delete-recovery-checkpoint-vm",
        "category": "Hyper-V Virtualization",
    },
    # Windows Server Administration & Networking (6)
    {
        "id": "mslearn-net-a-record-dns-registration",
        "url": "https://learn.microsoft.com/en-us/troubleshoot/windows-server/networking/a-record-registered-host-dns",
        "category": "Networking & DNS",
    },
    {
        "id": "mslearn-net-access-denied-akamai-cdn",
        "url": "https://learn.microsoft.com/en-us/troubleshoot/windows-server/networking/access-denied-visit-website-hosted-akamai-cdn",
        "category": "Networking & DNS",
    },
    {
        "id": "mslearn-net-cname-loopback-access-denied",
        "url": "https://learn.microsoft.com/en-us/troubleshoot/windows-server/networking/accessing-server-locally-with-fqdn-cname-alias-denied",
        "category": "Networking & DNS",
    },
    {
        "id": "mslearn-net-system-error-1331-disabled",
        "url": "https://learn.microsoft.com/en-us/troubleshoot/windows-server/networking/account-currently-disabled-system-error-1331",
        "category": "Networking & DNS",
    },
    {
        "id": "mslearn-net-ad-dns-serial-number-issue",
        "url": "https://learn.microsoft.com/en-us/troubleshoot/windows-server/networking/ad-integrated-dns-zone-serial-no-dot-behavior",
        "category": "Networking & DNS",
    },
    {
        "id": "mslearn-net-arp-caching-behavior",
        "url": "https://learn.microsoft.com/en-us/troubleshoot/windows-server/networking/address-resolution-protocol-arp-caching-behavior",
        "category": "Networking & DNS",
    },
]


def load_mslearn_docs(
    save_to_disk: bool = True,
    delay_seconds: float = 1.5,
) -> List[RawDocument]:
    """Fetch and parse real Microsoft Learn troubleshooting articles."""
    logger.info(f"Starting Microsoft Learn loader ({len(MSLEARN_URLS)} target URLs)...")
    out_dir = RAW_DATA_DIR / "mslearn"
    if save_to_disk:
        clean_output_directory(out_dir)

    documents: List[RawDocument] = []

    for index, target in enumerate(MSLEARN_URLS, 1):
        doc_id = target["id"]
        url = target["url"]
        category = target["category"]

        logger.info(f"[{index}/{len(MSLEARN_URLS)}] Fetching MS Learn: {url}")
        try:
            resp = requests.get(url, headers=DEFAULT_REQUEST_HEADERS, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"Failed to fetch {url} - Status HTTP {resp.status_code}. Skipping.")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract Title
            title_elem = soup.find("h1")
            title = title_elem.get_text().strip() if title_elem else soup.title.get_text().strip() if soup.title else doc_id

            # Locate primary content container on Microsoft Learn
            content_container = soup.find("main", id="main") or soup.find("div", class_="content") or soup.find("article")
            if not content_container:
                logger.warning(f"Could not locate main content container for {url}. Skipping.")
                continue

            # Remove unwanted sidebars/feedback forms inside article
            for extra in content_container.select(".page-metadata, .feedback-section, .action-container, nav, .breadcrumbs"):
                extra.decompose()

            markdown_body = html_element_to_markdown(content_container)

            # Ensure document has a top-level H1 header
            if not markdown_body.startswith("# "):
                markdown_body = f"# {title}\n\n{markdown_body}"

            doc = RawDocument(
                id=doc_id,
                title=title,
                content=markdown_body,
                source="Microsoft Learn",
                url=url,
                metadata={
                    "category": category,
                    "platform": "windows-server",
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            documents.append(doc)

            if save_to_disk:
                file_path = out_dir / f"{doc_id}.json"
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(doc.to_dict(), f, indent=2, ensure_ascii=False)

        except requests.exceptions.RequestException as e:
            logger.warning(f"Network error fetching {url}: {e}. Skipping.")
        except Exception as e:
            logger.warning(f"Unexpected error processing {url}: {e}. Skipping.", exc_info=True)

        if index < len(MSLEARN_URLS):
            time.sleep(delay_seconds)

    logger.info(f"Finished Microsoft Learn loader. Successfully loaded {len(documents)}/{len(MSLEARN_URLS)} documents.")
    return documents


# =====================================================================
# 2. Proxmox VE Wiki Loader
# =====================================================================

PROXMOX_URLS = [
    {
        "id": "proxmox-cluster-manager",
        "url": "https://pve.proxmox.com/wiki/Cluster_Manager",
        "category": "Cluster & Corosync",
    },
    {
        "id": "proxmox-high-availability",
        "url": "https://pve.proxmox.com/wiki/High_Availability",
        "category": "High Availability",
    },
    {
        "id": "proxmox-kvm-virtual-machines",
        "url": "https://pve.proxmox.com/wiki/Qemu/KVM_Virtual_Machines",
        "category": "Virtual Machines (KVM)",
    },
    {
        "id": "proxmox-linux-container",
        "url": "https://pve.proxmox.com/wiki/Linux_Container",
        "category": "LXC Containers",
    },
    {
        "id": "proxmox-storage-backends",
        "url": "https://pve.proxmox.com/wiki/Storage",
        "category": "Storage",
    },
    {
        "id": "proxmox-zfs-on-linux",
        "url": "https://pve.proxmox.com/wiki/ZFS_on_Linux",
        "category": "ZFS Storage",
    },
    {
        "id": "proxmox-network-configuration",
        "url": "https://pve.proxmox.com/wiki/Network_Configuration",
        "category": "Networking",
    },
    {
        "id": "proxmox-firewall",
        "url": "https://pve.proxmox.com/wiki/Firewall",
        "category": "Firewall & Security",
    },
    {
        "id": "proxmox-backup-and-restore",
        "url": "https://pve.proxmox.com/wiki/Backup_and_Restore",
        "category": "Backup & Disaster Recovery",
    },
    {
        "id": "proxmox-cluster-filesystem-pmxcfs",
        "url": "https://pve.proxmox.com/wiki/Proxmox_Cluster_File_System_(pmxcfs)",
        "category": "Cluster Filesystem",
    },
    {
        "id": "proxmox-troubleshooting-guide",
        "url": "https://pve.proxmox.com/wiki/Troubleshooting",
        "category": "General Troubleshooting",
    },
    {
        "id": "proxmox-disk-health-monitoring",
        "url": "https://pve.proxmox.com/wiki/Disk_Health_Monitoring",
        "category": "Storage & Hardware",
    },
    {
        "id": "proxmox-separate-cluster-network",
        "url": "https://pve.proxmox.com/wiki/Separate_Cluster_Network",
        "category": "Cluster Networking",
    },
]


def load_proxmox_docs(
    save_to_disk: bool = True,
    delay_seconds: float = 1.5,
) -> List[RawDocument]:
    """Fetch and parse real Proxmox VE MediaWiki documentation articles."""
    logger.info(f"Starting Proxmox VE Wiki loader ({len(PROXMOX_URLS)} target URLs)...")
    out_dir = RAW_DATA_DIR / "proxmox"
    if save_to_disk:
        clean_output_directory(out_dir)

    documents: List[RawDocument] = []

    for index, target in enumerate(PROXMOX_URLS, 1):
        doc_id = target["id"]
        url = target["url"]
        category = target["category"]

        logger.info(f"[{index}/{len(PROXMOX_URLS)}] Fetching Proxmox Wiki: {url}")
        try:
            resp = requests.get(url, headers=DEFAULT_REQUEST_HEADERS, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"Failed to fetch {url} - Status HTTP {resp.status_code}. Skipping.")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract Title from MediaWiki heading
            heading_elem = soup.find("h1", id="firstHeading")
            title = heading_elem.get_text().strip() if heading_elem else soup.title.get_text().strip() if soup.title else doc_id

            # Locate MediaWiki content body
            content_container = soup.find("div", id="mw-content-text")
            if not content_container:
                logger.warning(f"Could not locate #mw-content-text for {url}. Skipping.")
                continue

            # Remove MediaWiki Table of Contents, Edit section links, Categories, and navigation boxes
            for extra in content_container.select("#toc, .toc, .mw-editsection, #catlinks, .navbox, .printfooter"):
                extra.decompose()

            markdown_body = html_element_to_markdown(content_container)

            # Ensure document starts with H1 title
            if not markdown_body.startswith("# "):
                markdown_body = f"# Proxmox VE: {title}\n\n{markdown_body}"

            doc = RawDocument(
                id=doc_id,
                title=f"Proxmox VE: {title}",
                content=markdown_body,
                source="Proxmox VE Wiki",
                url=url,
                metadata={
                    "category": category,
                    "platform": "proxmox/linux",
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            documents.append(doc)

            if save_to_disk:
                file_path = out_dir / f"{doc_id}.json"
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(doc.to_dict(), f, indent=2, ensure_ascii=False)

        except requests.exceptions.RequestException as e:
            logger.warning(f"Network error fetching {url}: {e}. Skipping.")
        except Exception as e:
            logger.warning(f"Unexpected error processing {url}: {e}. Skipping.", exc_info=True)

        if index < len(PROXMOX_URLS):
            time.sleep(delay_seconds)

    logger.info(f"Finished Proxmox VE Wiki loader. Successfully loaded {len(documents)}/{len(PROXMOX_URLS)} documents.")
    return documents


# =====================================================================
# 3. ServerFault Stack Exchange API Loader
# =====================================================================

SERVERFAULT_TAGS = [
    "windows-server-2019",
    "hyper-v",
    "active-directory",
    "proxmox",
    "networking",
]


def load_serverfault_docs(
    save_to_disk: bool = True,
    items_per_tag: int = 5,
    delay_seconds: float = 1.5,
) -> List[RawDocument]:
    """Fetch real questions and accepted answers from ServerFault via Stack Exchange API v2.3."""
    logger.info(f"Starting ServerFault API loader (Tags: {', '.join(SERVERFAULT_TAGS)}, {items_per_tag} items/tag)...")
    out_dir = RAW_DATA_DIR / "serverfault"
    if save_to_disk:
        clean_output_directory(out_dir)

    documents: List[RawDocument] = []
    seen_question_ids = set()

    for tag_index, tag in enumerate(SERVERFAULT_TAGS, 1):
        logger.info(f"[{tag_index}/{len(SERVERFAULT_TAGS)}] Querying ServerFault questions for tag: '{tag}'")
        try:
            # Query top voted questions with accepted answers
            api_url = "https://api.stackexchange.com/2.3/questions"
            params = {
                "order": "desc",
                "sort": "votes",
                "tagged": tag,
                "site": "serverfault",
                "filter": "withbody",
                "pagesize": 15,  # Fetch slightly more to ensure finding items with accepted answers
            }

            resp = requests.get(api_url, params=params, headers=DEFAULT_REQUEST_HEADERS, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"Stack Exchange API error for tag '{tag}' - Status HTTP {resp.status_code}: {resp.text}. Skipping tag.")
                continue

            data = resp.json()
            questions = data.get("items", [])

            # Filter for questions with accepted answers not yet seen
            target_questions = []
            for q in questions:
                if q.get("accepted_answer_id") and q["question_id"] not in seen_question_ids:
                    target_questions.append(q)
                    seen_question_ids.add(q["question_id"])
                    if len(target_questions) >= items_per_tag:
                        break

            if not target_questions:
                logger.warning(f"No questions with accepted answers found for tag '{tag}'. Skipping.")
                continue

            # Fetch accepted answer bodies in batch
            accepted_ids = [str(q["accepted_answer_id"]) for q in target_questions]
            ids_joined = ";".join(accepted_ids)
            time.sleep(delay_seconds)

            answers_url = f"https://api.stackexchange.com/2.3/answers/{ids_joined}"
            ans_params = {
                "site": "serverfault",
                "filter": "withbody",
                "pagesize": len(accepted_ids),
            }

            ans_resp = requests.get(answers_url, params=ans_params, headers=DEFAULT_REQUEST_HEADERS, timeout=15)
            answers_map = {}
            if ans_resp.status_code == 200:
                for ans in ans_resp.json().get("items", []):
                    answers_map[ans["answer_id"]] = ans
            else:
                logger.warning(f"Failed to fetch answers for tag '{tag}' (HTTP {ans_resp.status_code}).")

            # Parse each Q&A pair into a RawDocument
            for q in target_questions:
                q_id = q["question_id"]
                q_title = q.get("title", f"ServerFault Question {q_id}")
                q_link = q.get("link", f"https://serverfault.com/questions/{q_id}")
                q_score = q.get("score", 0)
                q_tags = q.get("tags", [])

                # Clean Question Body HTML
                q_soup = BeautifulSoup(q.get("body", ""), "html.parser")
                q_md = html_element_to_markdown(q_soup)

                # Clean Answer Body HTML
                ans_obj = answers_map.get(q.get("accepted_answer_id"))
                if ans_obj:
                    ans_score = ans_obj.get("score", 0)
                    ans_soup = BeautifulSoup(ans_obj.get("body", ""), "html.parser")
                    ans_md = html_element_to_markdown(ans_soup)
                else:
                    ans_score = 0
                    ans_md = "Accepted answer content could not be retrieved."

                doc_id = f"serverfault-{q_id}"
                combined_content = f"""# ServerFault: {q_title}

## Question (Score: {q_score} votes | Tags: {', '.join(q_tags)})
{q_md}

## Accepted Solution (Score: {ans_score} votes)
{ans_md}
"""

                doc = RawDocument(
                    id=doc_id,
                    title=f"ServerFault: {q_title}",
                    content=combined_content.strip(),
                    source="ServerFault",
                    url=q_link,
                    metadata={
                        "category": tag,
                        "question_id": q_id,
                        "score": q_score,
                        "tags": q_tags,
                        "platform": "serverfault/sysadmin",
                        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                )
                documents.append(doc)

                if save_to_disk:
                    file_path = out_dir / f"{doc_id}.json"
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(doc.to_dict(), f, indent=2, ensure_ascii=False)

        except requests.exceptions.RequestException as e:
            logger.warning(f"Network error querying ServerFault for tag '{tag}': {e}. Skipping.")
        except Exception as e:
            logger.warning(f"Unexpected error processing ServerFault tag '{tag}': {e}. Skipping.", exc_info=True)

        if tag_index < len(SERVERFAULT_TAGS):
            time.sleep(delay_seconds)

    logger.info(f"Finished ServerFault loader. Successfully loaded {len(documents)} real Q&A documents.")
    return documents


# =====================================================================
# Main Unified Loader
# =====================================================================

def load_all_sources(
    save_to_disk: bool = True,
    delay_seconds: float = 1.5,
) -> List[RawDocument]:
    """Run all three source loaders and return aggregated raw documents."""
    all_docs: List[RawDocument] = []
    all_docs.extend(load_mslearn_docs(save_to_disk=save_to_disk, delay_seconds=delay_seconds))
    all_docs.extend(load_proxmox_docs(save_to_disk=save_to_disk, delay_seconds=delay_seconds))
    all_docs.extend(load_serverfault_docs(save_to_disk=save_to_disk, delay_seconds=delay_seconds))
    logger.info(f"Total raw documents loaded across all sources: {len(all_docs)}")
    return all_docs


def load_documents_from_disk(source: Optional[str] = None) -> List[RawDocument]:
    """Load previously fetched and saved RawDocuments from data/raw directory."""
    documents: List[RawDocument] = []
    sources = [source] if source else ["mslearn", "proxmox", "serverfault"]

    for src in sources:
        src_dir = RAW_DATA_DIR / src
        if not src_dir.exists():
            continue
        for file_path in src_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    doc = RawDocument(
                        id=data["id"],
                        title=data["title"],
                        content=data["content"],
                        source=data["source"],
                        url=data["url"],
                        metadata=data.get("metadata", {}),
                    )
                    documents.append(doc)
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")

    logger.info(f"Loaded {len(documents)} document(s) from disk ({RAW_DATA_DIR})")
    return documents


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runbook Copilot Real Data Loaders")
    parser.add_argument(
        "--source",
        choices=["mslearn", "proxmox", "serverfault", "all"],
        default="all",
        help="Specify source to load (default: all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Politeness delay in seconds between HTTP requests (default: 1.5)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write raw JSON files to disk (memory only)",
    )
    args = parser.parse_args()

    save = not args.no_save
    if args.source == "mslearn":
        load_mslearn_docs(save_to_disk=save, delay_seconds=args.delay)
    elif args.source == "proxmox":
        load_proxmox_docs(save_to_disk=save, delay_seconds=args.delay)
    elif args.source == "serverfault":
        load_serverfault_docs(save_to_disk=save, delay_seconds=args.delay)
    else:
        load_all_sources(save_to_disk=save, delay_seconds=args.delay)
