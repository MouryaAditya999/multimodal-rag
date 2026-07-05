import os
import pandas as pd


def extract_tables_from_pdf(pdf_path):
    """Extract tables from PDF using PyMuPDF's built-in table detection."""
    import fitz

    tables_data = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        found_tables = page.find_tables()
        for table_idx, table in enumerate(found_tables.tables):
            df = table.to_pandas()
            if df.empty:
                continue
            text_repr = _serialize_table(df)
            tables_data.append(
                {
                    "page": page_num + 1,
                    "table_index": table_idx,
                    "text": text_repr,
                    "source": os.path.basename(pdf_path),
                    "format": "pdf",
                    "modality": "table",
                    "rows": len(df),
                    "cols": len(df.columns),
                }
            )
    doc.close()
    return tables_data


def extract_tables_from_csv(csv_path):
    """Extract table from CSV file."""
    df = pd.read_csv(csv_path)
    text_repr = _serialize_table(df)
    return [
        {
            "page": 1,
            "table_index": 0,
            "text": text_repr,
            "source": os.path.basename(csv_path),
            "format": "csv",
            "modality": "table",
            "rows": len(df),
            "cols": len(df.columns),
        }
    ]


def extract_tables_from_xlsx(xlsx_path):
    """Extract tables from XLSX file (one per sheet)."""
    tables = []
    xl = pd.ExcelFile(xlsx_path)
    for sheet_name in xl.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
        if df.empty:
            continue
        text_repr = _serialize_table(df)
        tables.append(
            {
                "page": 1,
                "table_index": 0,
                "text": text_repr,
                "source": os.path.basename(xlsx_path),
                "format": "xlsx",
                "modality": "table",
                "rows": len(df),
                "cols": len(df.columns),
            }
        )
    return tables


def _serialize_table(df):
    """Convert a pandas DataFrame into a readable text representation."""
    lines = []
    header = " | ".join(str(c) for c in df.columns)
    lines.append(f"Columns: {header}")
    lines.append(f"Rows: {len(df)}")
    lines.append("")
    for idx, row in df.iterrows():
        row_str = " | ".join(str(v) for v in row.values)
        lines.append(f"Row {idx + 1}: {row_str}")
    return "\n".join(lines)


def extract_tables(file_path):
    """Detect file type and extract tables."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_tables_from_pdf(file_path)
    elif ext == ".csv":
        return extract_tables_from_csv(file_path)
    elif ext == ".xlsx":
        return extract_tables_from_xlsx(file_path)
    else:
        return []
