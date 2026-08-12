from pathlib import Path
import time

from src.document_processor import DocumentProcessor
from src.predict import DocumentPredictor
from src.rag import DocumentRAG
from src.groq_client import ask_document
from src.database import save_document


class DocuMindPipeline:

    def __init__(self):

        self.processor = DocumentProcessor()
        self.predictor = DocumentPredictor()
        self.rag = DocumentRAG()

    def process_document(self, file_path):

        start_time = time.time()

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        # --------------------------------
        # 1. PDF → Images → OCR
        # --------------------------------

        documents = self.processor.process_pdf(
            str(file_path)
        )

        if not documents:
            raise ValueError(
                "No pages found in document."
            )

        # --------------------------------
        # 2. PyTorch Document Classification
        # --------------------------------

        first_page = documents[0]["image"]

        prediction = self.predictor.predict(
            first_page
        )

        # --------------------------------
        # 3. Combine OCR text
        # --------------------------------

        full_text = "\n\n".join(
            document["text"]
            for document in documents
            if document["text"].strip()
        )

        if not full_text.strip():
            full_text = (
                "No readable text was extracted "
                "from this document."
            )

        # --------------------------------
        # 4. Add text to RAG
        # --------------------------------

        chunks = self.rag.add_document(
            full_text,
            source=file_path.name
        )

        # --------------------------------
        # 5. Processing time
        # --------------------------------

        processing_time = round(
            time.time() - start_time,
            2
        )

        # --------------------------------
        # 6. Save analysis to SQL
        # --------------------------------

        save_document(
            filename=file_path.name,
            document_type=prediction[
                "document_type"
            ],
            confidence=prediction[
                "confidence"
            ],
            pages=len(documents),
            chunks=chunks,
            processing_time=processing_time,
        )

        # --------------------------------
        # 7. Return complete result
        # --------------------------------

        return {
            "filename": file_path.name,

            "document_type": prediction[
                "document_type"
            ],

            "confidence": prediction[
                "confidence"
            ],

            "top_predictions": prediction[
                "top_predictions"
            ],

            "pages": len(documents),

            "chunks": chunks,

            "processing_time": processing_time,

            "text": full_text,
        }

    def ask(self, question):

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        # --------------------------------
        # 1. Retrieve relevant chunks
        # --------------------------------

        results = self.rag.search(
            question,
            k=4
        )

        if not results:
            return (
                "I couldn't find relevant "
                "information in the document."
            )

        # --------------------------------
        # 2. Build context
        # --------------------------------

        context = "\n\n".join(
            result.page_content
            for result in results
        )

        # --------------------------------
        # 3. Send context to Groq
        # --------------------------------

        answer = ask_document(
            question,
            context
        )

        return answer