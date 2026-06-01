"""
Sentiment Transformers — Web UI
Run: streamlit run app.py
"""

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from sentiment_engine import analyze_dataframe, analyze_text

APP_DIR = Path(__file__).resolve().parent
SAMPLE_CSV = APP_DIR / "sample_texts.csv"

st.set_page_config(
    page_title="Sentiment Transformers",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.25);
    }

    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.05rem;
    }

    .result-positive {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid #28a745;
        padding: 1.25rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }

    .result-negative {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 4px solid #dc3545;
        padding: 1.25rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }

    .badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .badge-positive { background: #28a745; color: white; }
    .badge-negative { background: #dc3545; color: white; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fc 0%, #eef1f8 100%);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-header">
        <h1>💬 Sentiment Transformers</h1>
        <p>AI-powered sentiment analysis using Hugging Face DistilBERT — 100% free &amp; open source</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### About")
    st.markdown(
        """
        This app uses **DistilBERT** fine-tuned on SST-2 to classify text as
        **POSITIVE** or **NEGATIVE** with confidence scores.

        Built with Hugging Face Transformers, Streamlit, and PyTorch.
        """
    )
    st.markdown("---")
    st.markdown("### Model")
    model_name = st.text_input(
        "Model ID",
        value="distilbert-base-uncased-finetuned-sst-2-english",
        help="Hugging Face model identifier",
    )
    st.markdown("---")
    st.markdown(
        "[View on GitHub](https://github.com/albertraj163/sentiment-transformers)"
    )

tab_single, tab_batch, tab_sample = st.tabs(
    ["✏️ Analyze Text", "📁 Batch CSV", "📋 Sample Data"]
)


def render_result(result: dict) -> None:
    label = result["label"]
    score = result["score"]
    text = result.get("text", "")

    css_class = "result-positive" if label == "POSITIVE" else "result-negative"
    badge_class = "badge-positive" if label == "POSITIVE" else "badge-negative"
    emoji = "😊" if label == "POSITIVE" else "😞"

    st.markdown(
        f"""
        <div class="{css_class}">
            <span class="badge {badge_class}">{emoji} {label}</span>
            <p style="margin: 0.75rem 0 0.5rem 0; font-size: 1.1rem;">{text}</p>
            <p style="margin: 0; opacity: 0.85;">Confidence: <strong>{score * 100:.1f}%</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


with tab_single:
    st.markdown("#### Enter text to analyze sentiment")
    user_text = st.text_area(
        "Your text",
        placeholder="Type or paste text here… e.g. 'I love this product!'",
        height=120,
        label_visibility="collapsed",
    )

    col1, _ = st.columns([1, 4])
    with col1:
        analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

    if analyze_btn:
        if not user_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing sentiment…"):
                result = analyze_text(user_text, model_name=model_name)
            render_result(result)
            st.progress(result["score"], text=f"Confidence: {result['score'] * 100:.1f}%")


with tab_batch:
    st.markdown("#### Upload a CSV file with a `text` column")
    uploaded = st.file_uploader("Choose CSV", type=["csv"], label_visibility="collapsed")

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.dataframe(df.head(10), use_container_width=True)

            if st.button("Analyze all rows", type="primary", key="batch_analyze"):
                with st.spinner(f"Analyzing {len(df)} rows…"):
                    results_df = analyze_dataframe(df, model_name=model_name)

                st.success(f"Analyzed {len(results_df)} texts successfully.")

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Positive", (results_df["label"] == "POSITIVE").sum())
                col_b.metric("Negative", (results_df["label"] == "NEGATIVE").sum())
                col_c.metric("Avg confidence", f"{results_df['score'].mean() * 100:.1f}%")

                st.dataframe(results_df, use_container_width=True)

                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="⬇️ Download results CSV",
                    data=csv_buffer.getvalue(),
                    file_name="sentiment_results.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Could not process file: {exc}")
    else:
        st.info("Upload a CSV file with at least one column named `text`.")


with tab_sample:
    st.markdown("#### Run analysis on bundled sample texts")
    sample_df = pd.read_csv(SAMPLE_CSV)
    st.dataframe(sample_df, use_container_width=True)

    if st.button("Analyze samples", type="primary", key="sample_analyze"):
        with st.spinner("Analyzing sample texts…"):
            sample_results = analyze_dataframe(sample_df, model_name=model_name)

        for _, row in sample_results.iterrows():
            render_result(
                {"text": row["text"], "label": row["label"], "score": row["score"]}
            )

        st.download_button(
            label="⬇️ Download sample results",
            data=sample_results.to_csv(index=False),
            file_name="sample_sentiment_results.csv",
            mime="text/csv",
        )
