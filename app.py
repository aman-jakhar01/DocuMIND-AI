import tempfile

import pandas as pd
import streamlit as st

from src.pipeline import DocuMindPipeline
from src.database import get_analytics, get_documents


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.70;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 700;
    }

    .stMetric {
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 12px;
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧠 DocuMind AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Deep Learning Powered Intelligent Document Analysis Platform"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# LOAD PIPELINE
# ============================================================

@st.cache_resource
def load_pipeline():
    return DocuMindPipeline()


pipeline = load_pipeline()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System Status")

    st.success("🧠 PyTorch Model — Ready")
    st.success("👁️ OCR Engine — Ready")
    st.success("🔎 RAG System — Ready")
    st.success("🤖 Groq AI — Ready")
    st.success("🗄️ SQL Database — Ready")

    st.divider()

    st.subheader("Technology Stack")

    st.write(
        """
        - Python
        - PyTorch
        - ResNet18
        - CUDA
        - OCR
        - Transformers
        - RAG
        - Chroma
        - Groq
        - SQLite
        - Streamlit
        """
    )

    st.divider()

    st.caption(
        "DocuMind AI — Intelligent Document Analysis"
    )


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

st.header("📄 Document Analysis")

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="Upload a PDF for AI-powered classification, OCR, analysis and Q&A.",
)


if uploaded_file is not None:

    st.info(
        f"Selected document: **{uploaded_file.name}**"
    )

    if st.button(
        "🚀 Analyze Document",
        type="primary",
        use_container_width=True,
    ):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_path = temp_file.name

        with st.spinner(
            "Processing document with DocuMind AI..."
        ):

            try:

                result = pipeline.process_document(
                    temp_path
                )

                st.session_state["result"] = result
                st.session_state["document_processed"] = True

                # Reset question-related state
                st.session_state.pop(
                    "last_answer",
                    None,
                )

                st.success(
                    "✅ Document processed successfully!"
                )

            except Exception as error:

                st.error(
                    f"❌ Document processing failed: {error}"
                )


# ============================================================
# DOCUMENT RESULTS
# ============================================================

if st.session_state.get(
    "document_processed",
    False,
):

    result = st.session_state["result"]

    st.divider()

    st.header("📊 Document Analysis")


    # --------------------------------------------------------
    # MAIN METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Document Type",
            result["document_type"].title(),
        )

    with col2:

        st.metric(
            "Confidence",
            f"{result['confidence']}%",
        )

    with col3:

        st.metric(
            "Pages",
            result["pages"],
        )

    with col4:

        st.metric(
            "Text Chunks",
            result["chunks"],
        )


    # --------------------------------------------------------
    # PROCESSING TIME
    # --------------------------------------------------------

    st.caption(
        f"⏱️ Processing time: "
        f"{result.get('processing_time', 'N/A')} seconds"
    )


    # --------------------------------------------------------
    # CONFIDENCE STATUS
    # --------------------------------------------------------

    confidence = result["confidence"]

    if confidence >= 75:

        st.success(
            "🟢 High-confidence document classification."
        )

    elif confidence >= 50:

        st.info(
            "🟡 Moderate-confidence classification. "
            "Consider verifying the predicted type."
        )

    else:

        st.warning(
            "🔴 Low-confidence classification. "
            "Please verify the predicted document type."
        )


    # ========================================================
    # TOP 3 PREDICTIONS
    # ========================================================

    st.subheader("🧠 Top 3 Predictions")

    predictions = result.get(
        "top_predictions",
        [],
    )

    if predictions:

        prediction_col1, prediction_col2, prediction_col3 = (
            st.columns(3)
        )

        columns = [
            prediction_col1,
            prediction_col2,
            prediction_col3,
        ]

        for index, prediction in enumerate(
            predictions[:3]
        ):

            with columns[index]:

                st.metric(
                    f"#{index + 1} "
                    f"{prediction['document_type'].title()}",
                    f"{prediction['confidence']}%",
                )


    # ========================================================
    # OCR TEXT
    # ========================================================

    st.subheader("📝 Extracted Text")

    with st.expander(
        "View OCR Output",
        expanded=False,
    ):

        extracted_text = result.get(
            "text",
            "",
        )

        if extracted_text.strip():

            st.text_area(
                "OCR Text",
                extracted_text,
                height=350,
                label_visibility="collapsed",
            )

        else:

            st.warning(
                "No readable text was extracted."
            )


    # ========================================================
    # AI SUMMARY
    # ========================================================

    st.subheader("🤖 AI Document Summary")

    if st.button(
        "Generate AI Summary",
        use_container_width=True,
    ):

        with st.spinner(
            "Groq is analyzing the document..."
        ):

            try:

                summary = pipeline.ask(
                    """
                    Summarize this document.

                    Provide:
                    1. A short overview
                    2. The most important information
                    3. Important entities, numbers or dates
                    4. Any important conclusions

                    Keep the answer clear and structured.
                    """
                )

                st.session_state[
                    "summary"
                ] = summary

            except Exception as error:

                st.error(
                    f"Summary generation failed: {error}"
                )


    if "summary" in st.session_state:

        st.markdown(
            st.session_state["summary"]
        )


    # ========================================================
    # DOCUMENT Q&A
    # ========================================================

    st.divider()

    st.header("💬 Ask Questions About Your Document")

    st.caption(
        "Ask questions and DocuMind will retrieve relevant "
        "information from the document before sending it to Groq."
    )

    question = st.text_input(
        "Your question",
        placeholder="e.g. What is this document about?",
    )

    if st.button(
        "🔍 Ask DocuMind",
        type="primary",
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching the document and generating an answer..."
            ):

                try:

                    answer = pipeline.ask(
                        question
                    )

                    st.session_state[
                        "last_answer"
                    ] = answer

                except Exception as error:

                    st.error(
                        f"Question answering failed: {error}"
                    )


    if "last_answer" in st.session_state:

        st.markdown("### 🤖 DocuMind Answer")

        st.info(
            st.session_state["last_answer"]
        )


# ============================================================
# ANALYTICS DASHBOARD
# ============================================================

st.divider()

st.header("📊 DocuMind Analytics")

try:

    analytics = get_analytics()

    # --------------------------------------------------------
    # ANALYTICS METRICS
    # --------------------------------------------------------

    analytics_col1, analytics_col2, analytics_col3 = (
        st.columns(3)
    )

    with analytics_col1:

        st.metric(
            "📄 Documents Processed",
            analytics["total_documents"],
        )

    with analytics_col2:

        st.metric(
            "🎯 Average Confidence",
            f"{analytics['avg_confidence']}%",
        )

    with analytics_col3:

        st.metric(
            "⏱️ Average Processing Time",
            f"{analytics['avg_processing_time']} sec",
        )


    # --------------------------------------------------------
    # DOCUMENT DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "📑 Document Type Distribution"
    )

    distribution = analytics.get(
        "distribution",
        [],
    )

    if distribution:

        distribution_df = pd.DataFrame(
            distribution,
            columns=[
                "Document Type",
                "Count",
            ],
        )

        chart_col1, chart_col2 = st.columns(
            [2, 1]
        )

        with chart_col1:

            st.bar_chart(
                distribution_df.set_index(
                    "Document Type"
                )
            )

        with chart_col2:

            st.dataframe(
                distribution_df,
                use_container_width=True,
                hide_index=True,
            )

    else:

        st.info(
            "No document analytics available yet."
        )


    # --------------------------------------------------------
    # PROCESSING HISTORY
    # --------------------------------------------------------

    st.subheader(
        "🕒 Processing History"
    )

    documents = get_documents()

    if documents:

        history_df = pd.DataFrame(
            documents,
            columns=[
                "ID",
                "Filename",
                "Document Type",
                "Confidence",
                "Pages",
                "Chunks",
                "Processing Time",
                "Created At",
            ],
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No processing history available yet."
        )

except Exception as error:

    st.warning(
        f"Analytics unavailable: {error}"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DocuMind AI • PyTorch + OCR + RAG + Groq + SQLite"
)