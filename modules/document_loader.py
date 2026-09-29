# Document Loader: PDF ya TXT file se text nikalna
from typing import Dict, Any

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


def load_text_from_file(uploaded_file) -> str:
    if not uploaded_file:
        return ""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        if not PYPDF_AVAILABLE:
            raise ImportError("PDF support ke liye 'pip install pypdf' run karein.")
        reader = pypdf.PdfReader(uploaded_file)
        return "\n\n".join([page.extract_text() or "" for page in reader.pages])
    return uploaded_file.read().decode("utf-8", errors="ignore")


def get_document_stats(text: str) -> Dict[str, int]:
    return {"chars": len(text), "words": len(text.split()), "lines": len(text.splitlines())}
