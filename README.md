# 🧠 DocuMind AI

![DocuMind AI Banner](assets/banner.png)

> **Intelligent Document Analysis Platform powered by Deep Learning, OCR, RAG, and Generative AI.**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12.1-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76b900?logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.61-red?logo=streamlit)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Groq-LLM-orange)](https://groq.com/)

**16 Document Categories • 89.82% Accuracy • 90.08% Macro F1**

---

## 📌 Overview

DocuMind AI is an end-to-end document intelligence platform that combines deep learning, OCR, semantic search, Retrieval-Augmented Generation (RAG), and LLMs to analyze documents intelligently.

The platform can:

- Upload and process PDF documents
- Classify documents into 16 categories using fine-tuned ResNet18
- Return top-3 predictions with confidence scores
- Extract text using Tesseract OCR
- Generate semantic embeddings and perform vector search
- Generate AI-powered document summaries
- Answer questions using document-grounded RAG
- Store processing history and analytics in SQLite
- Provide an interactive Streamlit dashboard

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 Deep Learning | ResNet18 transfer learning for 16 document categories |
| 🔍 OCR | Text extraction using Tesseract |
| 📄 PDF Processing | PDF page processing using PyMuPDF |
| 🧩 Semantic Search | Sentence-transformer embeddings |
| 📚 RAG | Context-aware document retrieval |
| 🤖 Generative AI | Groq-powered summaries and Q&A |
| 🗄️ Database | SQLite document history and analytics |
| 📊 Dashboard | Interactive Streamlit application |
| 🎯 Predictions | Top-3 classes with confidence scores |

---

## 🏗️ Architecture

![DocuMind AI Architecture](assets/architecture.png)

### Processing Pipeline

```text
PDF Upload
    ↓
PDF Processing + OCR
    ↓
┌───────────────────────┬───────────────────────┐
│                       │                       │
▼                       ▼                       │
ResNet18 V2          Extracted Text              │
│                       ↓                       │
│                  Text Chunking                 │
│                       ↓                       │
│                  Embeddings                    │
│                       ↓                       │
│                    Chroma                      │
│                       ↓                       │
│                      RAG                        │
└───────────────┬───────┘                        │
                ↓                                │
             Groq LLM                            │
          ┌─────┴─────┐                          │
          ▼           ▼                          │
       Summary       Q&A                         │
          └─────┬─────┘                          │
                ↓                                │
           SQLite Analytics                       │
                ↓                                │
         Streamlit Dashboard                      │
```

---

## 📂 Supported Document Categories

1. Letter
2. Form
3. Email
4. Handwritten
5. Advertisement
6. Scientific Report
7. Scientific Publication
8. Specification
9. File Folder
10. News Article
11. Budget
12. Invoice
13. Presentation
14. Questionnaire
15. Resume
16. Memo

---

## 🧠 Deep Learning Model

DocuMind uses a ResNet18 transfer-learning model for document image classification.

### V1 → V2 Improvement

| Metric | V1 | V2 |
|---|---:|---:|
| Architecture | ResNet18 | ResNet18 |
| Training Images | 8,000 | 8,000 |
| Epochs | 3 | 5 |
| Accuracy | 84.07% | **89.82%** |
| Improvement | — | **+5.75 percentage points** |

### V2 Improvements

- Data augmentation
- Random resized cropping
- Random rotation
- Brightness and contrast augmentation
- Dropout
- Label smoothing
- AdamW optimizer
- Weight decay
- Cosine learning-rate scheduling
- Best-validation-model checkpointing
- CUDA GPU training

---

## 📊 Model Evaluation

### Overall Results

| Metric | Score |
|---|---:|
| Accuracy | **89.82%** |
| Macro Precision | **90.86%** |
| Macro Recall | **89.82%** |
| Macro F1 | **90.08%** |

### Per-Class Performance

| Document Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Letter | 80.28% | 91.94% | 85.71% |
| Form | 85.71% | 87.10% | 86.40% |
| Email | 98.31% | 93.55% | 95.87% |
| Handwritten | 100.00% | 96.77% | 98.36% |
| Advertisement | 96.49% | 88.71% | 92.44% |
| Scientific Report | 72.22% | 83.87% | 77.61% |
| Scientific Publication | 90.48% | 91.94% | 91.20% |
| Specification | 100.00% | 91.94% | 95.80% |
| File Folder | 96.67% | 93.55% | 95.08% |
| News Article | 93.10% | 87.10% | 90.00% |
| Budget | 87.93% | 82.26% | 85.00% |
| Invoice | 95.08% | 93.55% | 94.31% |
| Presentation | 68.24% | 93.55% | 78.91% |
| Questionnaire | 92.86% | 83.87% | 88.14% |
| Resume | 98.28% | 91.94% | 95.00% |
| Memo | 98.15% | 85.48% | 91.38% |

### Confusion Matrix

![DocuMind AI Confusion Matrix](models/evaluation/confusion_matrix.png)

---

## 🖥️ Application Screenshots

### Dashboard

![DocuMind AI Dashboard](assets/screenshots/dashboard.png)

### Document Classification

![Document Classification](assets/screenshots/classification.png)

### OCR and AI Analysis

![OCR and AI Analysis](assets/screenshots/analysis.png)

### Document Q&A

![Document Q&A](assets/screenshots/qa.png)

### Analytics

![DocuMind Analytics](assets/screenshots/analytics.png)

---

## 🛠️ Tech Stack

- **Python 3.12**
- **PyTorch**
- **ResNet18**
- **CUDA**
- **PyMuPDF**
- **Tesseract OCR**
- **Transformers / Sentence Transformers**
- **ChromaDB**
- **LangChain**
- **Groq**
- **SQLite**
- **Streamlit**
- **uv**
- **Git / GitHub**

---

## 📁 Project Structure

```text
DocuMind-AI/
│
├── app.py
├── init_db.py
├── test_pipeline.py
├── test_prediction.py
├── pyproject.toml
├── uv.lock
├── README.md
├── .gitignore
│
├── assets/
│   ├── banner.png
│   ├── architecture.png
│   └── screenshots/
│       ├── dashboard.png
│       ├── classification.png
│       ├── analysis.png
│       ├── qa.png
│       └── analytics.png
│
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── classifier.py
│   ├── pipeline.py
│   ├── document_processor.py
│   ├── rag.py
│   ├── groq_client.py
│   └── database.py
│
└── models/
    ├── document_classifier_v2.pth
    ├── class_names.json
    └── evaluation/
        ├── confusion_matrix.png
        └── classification_report.txt
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DocuMind-AI.git
cd DocuMind-AI
```

### Install dependencies

This project uses `uv`.

```bash
uv sync
```

### Configure Groq

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

Never commit your `.env` file.

### Install Tesseract OCR

Install Tesseract OCR and make sure it is available in your system PATH.

Verify:

```bash
tesseract --version
```

### Initialize the database

```bash
uv run python init_db.py
```

---

## 🚀 Run the Application

```bash
uv run streamlit run app.py
```

Open the Streamlit URL shown in the terminal.

---

## 🔄 How It Works

1. Upload a PDF through Streamlit.
2. Process PDF pages using PyMuPDF.
3. Classify the document using ResNet18 V2.
4. Extract text using Tesseract OCR.
5. Split the text into chunks.
6. Generate semantic embeddings.
7. Store and retrieve chunks using ChromaDB.
8. Use RAG to provide relevant context to the Groq LLM.
9. Generate summaries and document-grounded answers.
10. Store document history and analytics in SQLite.
11. Display results through the Streamlit dashboard.

---

## 🔐 Security

Keep secrets and generated data out of Git:

```gitignore
.env
.venv/
__pycache__/
data/uploads/
data/processed/
data/chroma_db/
data/documind.db
```

Never publish your Groq API key.

---

## ⚠️ Limitations

- Real-world PDFs can differ from the training distribution.
- Classification confidence is not a calibrated probability.
- OCR accuracy depends on document quality.
- Complex layouts can reduce OCR and classification performance.
- LLM responses depend on the quality of retrieved text.

---

## 🔮 Future Improvements

- Page-level document classification
- Confidence calibration
- Improved OCR preprocessing
- Layout-aware document models
- Larger and more diverse training data
- Advanced table extraction
- Cloud deployment
- Authentication and multi-user support
- Automated model monitoring

---

## 👨‍💻 Author

**Aman Jakhar**

End-to-end Machine Learning and Generative AI portfolio project.

---

⭐ **If you find DocuMind AI useful, consider starring the repository.**
