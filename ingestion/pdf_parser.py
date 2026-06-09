# ingestion/pdf_parser.py
import fitz  # PyMuPDF
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import hashlib, base64

@dataclass
class ParsedElement:
    doc_id: str
    source_file: str
    page: int
    content_type: str          # "text" | "image" | "table"
    content: str
    raw_image_b64: Optional[str] = None
    bbox: Optional[tuple] = None
    metadata: dict = field(default_factory=dict)

class PDFParser:
    def __init__(self, ocr_fallback: bool = True):
        self.ocr_fallback = ocr_fallback

    def parse(self, pdf_path: str) -> List[ParsedElement]:
        path = Path(pdf_path)
        doc_id = hashlib.md5(path.read_bytes()).hexdigest()[:12]
        elements = []

        fitz_doc = fitz.open(pdf_path)
        for page_num, page in enumerate(fitz_doc, start=1):

            # Extract text blocks with layout awareness
            blocks = page.get_text("dict")["blocks"]
            page_text_parts = []
            for block in blocks:
                if block["type"] == 0:  # text block
                    for line in block["lines"]:
                        line_text = " ".join(span["text"] for span in line["spans"])
                        page_text_parts.append(line_text)

            page_text = "\n".join(page_text_parts).strip()

            if page_text:
                elements.append(ParsedElement(
                    doc_id=doc_id,
                    source_file=path.name,
                    page=page_num,
                    content_type="text",
                    content=page_text,
                    metadata={"total_pages": len(fitz_doc)}
                ))

            # Extract embedded images
            image_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = fitz_doc.extract_image(xref)
                img_bytes = base_image["image"]
                img_b64 = base_image["image"]
                img_b64 = base64.b64encode(img_bytes).decode()

                elements.append(ParsedElement(
                    doc_id=doc_id,
                    source_file=path.name,
                    page=page_num,
                    content_type="image",
                    content="",  # Filled later by ImageDescriber
                    raw_image_b64=img_b64,
                    metadata={"img_idx": img_idx, "ext": base_image["ext"]}
                ))

        fitz_doc.close()
        return elements