# Chunking: Badi file ko 500-500 chars ke tukdo me todna (Overlap se fact beech me nahi kat-ta)
import re
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Fixed-size: Simple sliding window se characters count karke todna
def fixed_size_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    step = max(1, chunk_size - overlap)
    return [text[i:i + chunk_size].strip() for i in range(0, len(text), step) if text[i:i + chunk_size].strip()]


# 2. Semantic: Paragraphs (\n\n) ke hisab se todna taaki complete meaning safe rahe
def semantic_chunks(text: str, max_chars: int = 600) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text.strip()) if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) + 2 <= max_chars:
            current = (current + "\n\n" + p) if current else p
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks


# 3. Recursive (LangChain): Industry standard - paragraph, line, aur words ke priority order me todna
def recursive_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    if not text or not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)
