"""Technical-doc-aware chunking for Runbook Copilot.

Respects markdown headers, hierarchical context breadcrumbs, code blocks,
and procedural runbook steps rather than performing naive fixed-length text splitting.
"""

import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from ingestion.loaders import RawDocument


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    title: str
    source: str
    url: str
    header_path: str
    content: str
    has_code: bool
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TechnicalMarkdownChunker:
    """Chunks technical markdown documents while preserving code blocks and header context."""

    def __init__(
        self,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 200,
        overlap_size: int = 150,
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size

    def _extract_blocks(self, text: str) -> List[Dict[str, Any]]:
        """Parse text into semantic blocks (headings, code blocks, paragraphs)."""
        lines = text.splitlines()
        blocks = []
        current_block_lines = []
        in_code_block = False
        code_fence = ""

        for line in lines:
            # Check for code fence start/end
            fence_match = re.match(r"^(```|~~~)", line)
            if fence_match:
                if not in_code_block:
                    # Flush prior normal text block
                    if current_block_lines:
                        content = "\n".join(current_block_lines).strip()
                        if content:
                            blocks.append({"type": "text", "content": content})
                        current_block_lines = []
                    in_code_block = True
                    code_fence = fence_match.group(1)
                    current_block_lines.append(line)
                else:
                    # Closing fence
                    current_block_lines.append(line)
                    if line.startswith(code_fence):
                        in_code_block = False
                        code_content = "\n".join(current_block_lines).strip()
                        blocks.append({"type": "code", "content": code_content})
                        current_block_lines = []
                continue

            if in_code_block:
                current_block_lines.append(line)
                continue

            # Check for Markdown Headings (# Header)
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
            if header_match:
                if current_block_lines:
                    content = "\n".join(current_block_lines).strip()
                    if content:
                        blocks.append({"type": "text", "content": content})
                    current_block_lines = []
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                blocks.append({"type": "header", "level": level, "title": title, "content": line.strip()})
                continue

            # Normal line
            current_block_lines.append(line)

        # Flush any trailing lines
        if current_block_lines:
            content = "\n".join(current_block_lines).strip()
            if content:
                block_type = "code" if in_code_block else "text"
                blocks.append({"type": block_type, "content": content})

        return blocks

    def chunk_document(self, doc: RawDocument) -> List[DocumentChunk]:
        """Split a RawDocument into technical chunks with hierarchical header breadcrumbs."""
        blocks = self._extract_blocks(doc.content)
        chunks: List[DocumentChunk] = []

        # Maintain header stack for breadcrumbs: [ (level, title) ]
        header_stack: List[tuple] = []
        current_chunk_parts: List[str] = []
        current_chunk_len = 0
        has_code = False
        chunk_idx = 0

        def current_header_path() -> str:
            if not header_stack:
                return doc.title
            return " > ".join([h[1] for h in header_stack])

        def flush_current_chunk(carry_overlap: bool = False):
            nonlocal current_chunk_parts, current_chunk_len, has_code, chunk_idx
            if not current_chunk_parts:
                return
            
            chunk_body = "\n\n".join(current_chunk_parts).strip()
            if not chunk_body:
                current_chunk_parts = []
                current_chunk_len = 0
                return

            h_path = current_header_path()
            # Prefix header path to chunk content for strong semantic retrieval
            prefixed_content = f"[{h_path}]\n\n{chunk_body}" if h_path != doc.title else chunk_body

            chunk_id = f"{doc.id}_chunk_{chunk_idx}"
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc.id,
                chunk_index=chunk_idx,
                title=doc.title,
                source=doc.source,
                url=doc.url,
                header_path=h_path,
                content=prefixed_content,
                has_code=has_code or ("```" in chunk_body),
                metadata={
                    **doc.metadata,
                    "char_count": len(prefixed_content),
                },
            )
            chunks.append(chunk)
            chunk_idx += 1

            if carry_overlap and self.overlap_size > 0 and len(chunk_body) > self.overlap_size:
                overlap_text = chunk_body[-self.overlap_size:]
                current_chunk_parts = [overlap_text]
                current_chunk_len = len(overlap_text)
                has_code = "```" in overlap_text
            else:
                current_chunk_parts = []
                current_chunk_len = 0
                has_code = False

        for block in blocks:
            b_type = block["type"]

            if b_type == "header":
                level = block["level"]
                title = block["title"]

                # Update header stack
                while header_stack and header_stack[-1][0] >= level:
                    header_stack.pop()
                header_stack.append((level, title))

                # Flush chunk if we accumulated enough text
                if current_chunk_len >= self.min_chunk_size:
                    flush_current_chunk(carry_overlap=False)
                
                # Append header to current part
                current_chunk_parts.append(block["content"])
                current_chunk_len += len(block["content"])

            elif b_type == "code":
                code_len = len(block["content"])
                # If code block alone causes overflow, flush preceding text first
                if current_chunk_len > 0 and (current_chunk_len + code_len > self.max_chunk_size):
                    flush_current_chunk(carry_overlap=True)
                
                current_chunk_parts.append(block["content"])
                current_chunk_len += code_len
                has_code = True

                if current_chunk_len >= self.max_chunk_size:
                    flush_current_chunk(carry_overlap=True)

            else:  # Text block
                text_len = len(block["content"])
                if current_chunk_len > 0 and (current_chunk_len + text_len > self.max_chunk_size):
                    flush_current_chunk(carry_overlap=True)

                current_chunk_parts.append(block["content"])
                current_chunk_len += text_len

                if current_chunk_len >= self.max_chunk_size:
                    flush_current_chunk(carry_overlap=True)

        # Flush any remaining parts
        flush_current_chunk(carry_overlap=False)

        return chunks


def chunk_all_documents(
    docs: List[RawDocument],
    max_chunk_size: int = 1500,
    min_chunk_size: int = 200,
    overlap_size: int = 150,
) -> List[DocumentChunk]:
    """Helper to chunk a list of RawDocuments."""
    chunker = TechnicalMarkdownChunker(
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
        overlap_size=overlap_size,
    )
    all_chunks: List[DocumentChunk] = []
    for doc in docs:
        doc_chunks = chunker.chunk_document(doc)
        all_chunks.extend(doc_chunks)
    return all_chunks
