"""Render PDF pages with a portable optional PDFium backend."""

from pathlib import Path

from imagedt._backends import require


def pdf2image(pdf_path, save_out=None, dpi=300):
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    pdfium = require("pypdfium2", "pdf")
    output = (
        Path(save_out) if save_out is not None else Path(pdf_path).with_suffix(".jpg")
    )
    with pdfium.PdfDocument(str(pdf_path)) as document:
        for index in range(len(document)):
            destination = (
                output
                if len(document) == 1
                else output.with_name(f"{output.stem}-{index}{output.suffix}")
            )
            page = document[index]
            try:
                bitmap = page.render(scale=dpi / 72)
                try:
                    image = bitmap.to_pil()
                    try:
                        image.convert("RGB").save(destination)
                    finally:
                        image.close()
                finally:
                    bitmap.close()
            finally:
                page.close()
