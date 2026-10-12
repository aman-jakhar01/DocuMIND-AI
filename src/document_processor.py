import os
import shutil
import io
from pathlib import Path

import fitz
import pytesseract
from PIL import Image


def configure_tesseract():
    """
    Configure Tesseract OCR for local Windows or Streamlit Cloud/Linux.

    You can optionally set TESSERACT_CMD to the full executable path
    in your environment if automatic detection does not find it.
    """
    configured_path = os.getenv("TESSERACT_CMD")

    if configured_path:
        executable = Path(configured_path)
        if executable.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(executable)
            return
        raise RuntimeError(
            "TESSERACT_CMD is set, but the executable was not found at: "
            f"{configured_path}"
        )

    detected_path = shutil.which("tesseract")
    if detected_path:
        pytesseract.pytesseract.tesseract_cmd = detected_path
        return

    # Common Windows installation path.
    windows_path = Path(
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
    if windows_path.is_file():
        pytesseract.pytesseract.tesseract_cmd = str(windows_path)
        return

    raise RuntimeError(
        "Tesseract OCR was not found. On Streamlit Cloud, add a "
        "packages.txt file at the repository root containing "
        "'tesseract-ocr'. On Windows, install Tesseract OCR and ensure "
        "tesseract.exe is on PATH, or set TESSERACT_CMD to its full path."
    )


class DocumentProcessor:
    def __init__(self, output_dir="data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def pdf_to_images(self, pdf_path):
        """Render each PDF page to a PNG image."""
        images = []

        with fitz.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image_bytes = pix.tobytes("png")
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                # Include the PDF filename to avoid collisions between uploads.
                safe_stem = Path(pdf_path).stem
                image_path = self.output_dir / (
                    f"{safe_stem}_page_{page_number}.png"
                )
                image.save(image_path)
                images.append(str(image_path))

        return images

    def extract_text_from_image(self, image_path):
        """Extract text from a rendered page using Tesseract OCR."""
        try:
            configure_tesseract()
            with Image.open(image_path) as image:
                return pytesseract.image_to_string(image)
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError(
                "Tesseract OCR is not installed or cannot be found. "
                "Install it and ensure its executable is available."
            ) from exc

    def process_pdf(self, pdf_path):
        """Convert a PDF to page images and return OCR text per page."""
        image_paths = self.pdf_to_images(pdf_path)
        documents = []

        for image_path in image_paths:
            text = self.extract_text_from_image(image_path)
            documents.append({
                "image": image_path,
                "text": text,
            })

        return documents
