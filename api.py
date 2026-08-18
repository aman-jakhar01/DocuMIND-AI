from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from src.pipeline import DocuMindPipeline
from src.predict import DocumentPredictor


app = FastAPI(
    title="DocuMind AI API",
    description="AI-powered document intelligence API",
    version="1.0.0",
)


# Load models/pipeline once when API starts
pipeline = DocuMindPipeline()
predictor = pipeline.predictor


# ============================================================
# REQUEST MODELS
# ============================================================

class QuestionRequest(BaseModel):
    question: str


# ============================================================
# BASIC ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "message": "DocuMind AI API is running",
        "status": "success",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "ResNet18 V2",
        "device": str(predictor.device),
    }


# ============================================================
# IMAGE PREDICTION
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/jpg",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Use JPG or PNG for /predict. Use /analyze for PDFs.",
        )

    suffix = Path(
        file.filename or "upload"
    ).suffix

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            shutil.copyfileobj(
                file.file,
                temp_file,
            )

            temp_path = temp_file.name

        result = predictor.predict(
            temp_path
        )

        return {
            "filename": file.filename,
            **result,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )

    finally:

        if temp_path:
            Path(temp_path).unlink(
                missing_ok=True
            )


# ============================================================
# COMPLETE PDF ANALYSIS
# ============================================================

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    if file.content_type != "application/pdf":

        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:

            shutil.copyfileobj(
                file.file,
                temp_file,
            )

            temp_path = temp_file.name

        # Run your existing complete pipeline
        result = pipeline.process_document(
            temp_path
        )

        return {
            "filename": file.filename,
            "document_type": result["document_type"],
            "confidence": result["confidence"],
            "top_predictions": result["top_predictions"],
            "pages": result["pages"],
            "chunks": result["chunks"],
            "processing_time": result["processing_time"],
            "text": result["text"],
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Document analysis failed: {str(e)}",
        )

    finally:

        if temp_path:
            Path(temp_path).unlink(
                missing_ok=True
            )


# ============================================================
# DOCUMENT Q&A
# ============================================================

@app.post("/ask")
def ask(request: QuestionRequest):

    if not request.question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:

        answer = pipeline.ask(
            request.question
        )

        return {
            "question": request.question,
            "answer": answer,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Question answering failed: {str(e)}",
        )