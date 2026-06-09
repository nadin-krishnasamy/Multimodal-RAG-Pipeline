# ingestion/table_parser.py
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import hashlib
from ingestion.pdf_parser import ParsedElement

class TableParser:
    def parse_tables(self, pdf_path: str) -> List[ParsedElement]:
        import camelot
        path = Path(pdf_path)
        doc_id = hashlib.md5(path.read_bytes()).hexdigest()[:12]
        elements = []

        try:
            # Try Lattice mode first (ideal for tables with clear grid lines)
            tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
            if len(tables) == 0:
                # Fall back to Stream mode (ideal for tables with whitespace margins instead of lines)
                tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")

            for i, table in enumerate(tables):
                df = table.df
                md_table = df.to_markdown(index=False)
                elements.append(ParsedElement(
                    doc_id=doc_id,
                    source_file=path.name,
                    page=table.page,
                    content_type="table",
                    content=md_table,
                    metadata={
                        "table_idx": i,
                        "accuracy": table.accuracy,
                        "rows": df.shape[0],
                        "cols": df.shape[1]
                    }
                ))
        except Exception as e:
            print(f"[WARN] Table extraction failed for {path.name}: {e}")

        return elements
