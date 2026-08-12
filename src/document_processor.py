import fitz
from pathlib import Path
from PIL import Image
import io
import pytesseract


# Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


class DocumentProcessor:

    def __init__(self):
        self.output_dir = Path(
            "data/processed"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def pdf_to_images(self, pdf_path):

        pdf = fitz.open(pdf_path)

        images = []

        for page_number, page in enumerate(pdf):

            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2)
            )

            image_bytes = pix.tobytes(
                "png"
            )

            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

            image_path = (
                self.output_dir
                / f"page_{page_number + 1}.png"
            )

            image.save(image_path)

            images.append(
                str(image_path)
            )

        pdf.close()

        return images

    def extract_text_from_image(
        self,
        image_path
    ):

        image = Image.open(
            image_path
        )

        text = pytesseract.image_to_string(
            image
        )

        return text

    def process_pdf(self, pdf_path):

        image_paths = self.pdf_to_images(
            pdf_path
        )

        documents = []

        for image_path in image_paths:

            text = self.extract_text_from_image(
                image_path
            )

            documents.append({
                "image": image_path,
                "text": text
            })

        return documents