"""
Sentiment Transformers — Web UI
Run: streamlit run app.py
"""

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from sentiment_engine import DEFAULT_MODEL, analyze_dataframe, analyze_text

APP_DIR = Path(__file__).resolve().parent
SAMPLE_CSV = APP_DIR / "sample_texts.csv"
GITHUB_URL = "https://github.com/albertraj163/sentiment-transformers"
LOCAL_APP_URL = "http://localhost:8501"
MODEL_URL = "https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english"
HF_DOCS_URL = "https://huggingface.co/docs/transformers/main/en/quicktour#pipeline-usage"

st.set_page_config(
    page_title="Sentiment Transformers | DistilBERT",
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

    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #9333ea 100%);
        padding: 2rem 2.25rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 18px 45px rgba(79, 70, 229, 0.22);
    }

    .hero h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
    }

    .hero p {
        margin: 0.65rem 0 0;
        opacity: 0.92;
        font-size: 1rem;
        max-width: 760px;
        line-height: 1.55;
    }

    .hero-tags {
        margin-top: 1rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }

    .hero-tag {
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.22);
        border-radius: 999px;
        padding: 0.3rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .panel {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.25rem 1.35rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    .panel-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.35rem;
    }

    .panel-subtitle {
        color: #6b7280;
        font-size: 0.88rem;
        margin-bottom: 1rem;
    }

    .result-positive,
    .result-negative {
        padding: 1.1rem 1.25rem;
        border-radius: 12px;
        margin: 0.85rem 0;
        border: 1px solid transparent;
    }

    .result-positive {
        background: #ecfdf5;
        border-color: #a7f3d0;
        border-left: 4px solid #059669;
    }

    .result-negative {
        background: #fef2f2;
        border-color: #fecaca;
        border-left: 4px solid #dc2626;
    }

    .badge {
        display: inline-block;
        padding: 0.32rem 0.8rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .badge-positive { background: #059669; color: white; }
    .badge-negative { background: #dc2626; color: white; }

    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
        color: #6b7280;
        font-size: 0.85rem;
        text-align: center;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fafafa 0%, #f3f4f6 100%);
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 0.75rem 1rem;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        border: none;
        border-radius: 10px;
        font-weight: 600;
    }

    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Sentiment Transformers</h1>
        <p>Professional sentiment analysis powered by Hugging Face DistilBERT.
        Classify customer feedback, reviews, and messages as positive or negative with confidence scores.</p>
        <div class="hero-tags">
            <span class="hero-tag">DistilBERT SST-2</span>
            <span class="hero-tag">100% Free & Open Source</span>
            <span class="hero-tag">Local + Batch CSV</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Project")
    st.markdown(
        f"""
        - **Local app:** [{LOCAL_APP_URL.replace('http://', '')}]({LOCAL_APP_URL})
        - **GitHub:** [{GITHUB_URL.replace('https://', '')}]({GITHUB_URL})
        - **Model:** [DistilBERT SST-2]({MODEL_URL})
        - **Docs:** [Transformers Pipeline]({HF_DOCS_URL})
        """
    )
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        """
        The model reads your text and returns:
        - **Label:** POSITIVE or NEGATIVE
        - **Score:** confidence between 0 and 1

        First run downloads the model from Hugging Face (~250 MB).
        """
    )
    st.markdown("---")
    st.markdown("### Settings")
    model_name = st.text_input(
        "Hugging Face model ID",
        value=DEFAULT_MODEL,
        help="Default model is optimized for English sentiment.",
    )
    st.caption(f"Default: `{DEFAULT_MODEL}`")

info_col1, info_col2, info_col3 = st.columns(3)
info_col1.metric("Engine", "DistilBERT")
info_col2.metric("Task", "Sentiment")
info_col3.metric("Cost", "Free")

tab_single, tab_batch, tab_sample = st.tabs(
    ["Single Text", "Batch Upload", "Sample Dataset"]
)


def render_result(result: dict) -> None:
    label = result["label"]
    score = result["score"]
    text = result.get("text", "")

    css_class = "result-positive" if label == "POSITIVE" else "result-negative"
    badge_class = "badge-positive" if label == "POSITIVE" else "badge-negative"

    st.markdown(
        f"""
        <div class="{css_class}">
            <span class="badge {badge_class}">{label}</span>
            <p style="margin: 0.85rem 0 0.45rem; font-size: 1.02rem; color: #111827;">{text}</p>
            <p style="margin: 0; color: #4b5563;">Confidence: <strong>{score * 100:.1f}%</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


with tab_single:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Analyze a single message</div>
            <div class="panel-subtitle">Paste product reviews, tweets, support tickets, or any English text.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    user_text = st.text_area(
        "Input text",
        placeholder="Example: The onboarding experience was smooth and the support team was excellent.",
        height=130,
        label_visibility="collapsed",
    )

    analyze_col, _ = st.columns([1, 3])
    with analyze_col:
        analyze_btn = st.button("Run analysis", type="primary", width="stretch")

    if analyze_btn:
        if not user_text.strip():
            st.warning("Enter some text before running analysis.")
        else:
            with st.spinner("Running DistilBERT inference…"):
                result = analyze_text(user_text, model_name=model_name)
            render_result(result)
            st.progress(result["score"], text=f"Model confidence: {result['score'] * 100:.1f}%")


with tab_batch:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Batch CSV analysis</div>
            <div class="panel-subtitle">Upload a CSV file containing a column named <code>text</code>.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "CSV file",
        type=["csv"],
        help="Required column: text",
        label_visibility="collapsed",
    )

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.caption(f"Previewing first 10 of {len(df)} rows")
            st.dataframe(df.head(10), width="stretch")

            if st.button("Analyze uploaded CSV", type="primary", key="batch_analyze"):
                with st.spinner(f"Analyzing {len(df)} rows…"):
                    results_df = analyze_dataframe(df, model_name=model_name)

                st.success(f"Completed analysis for {len(results_df)} rows.")

                metric_a, metric_b, metric_c = st.columns(3)
                metric_a.metric("Positive", int((results_df["label"] == "POSITIVE").sum()))
                metric_b.metric("Negative", int((results_df["label"] == "NEGATIVE").sum()))
                metric_c.metric("Avg confidence", f"{results_df['score'].mean() * 100:.1f}%")

                st.dataframe(results_df, width="stretch")

                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download results CSV",
                    data=csv_buffer.getvalue(),
                    file_name="sentiment_results.csv",
                    mime="text/csv",
                    width="stretch",
                )
        except Exception as exc:
            st.error(f"Unable to process CSV: {exc}")
    else:
        st.info("Upload a CSV with a `text` column to begin batch analysis.")


with tab_sample:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Built-in sample dataset</div>
            <div class="panel-subtitle">Try the app instantly using bundled examples from <code>sample_texts.csv</code>.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sample_df = pd.read_csv(SAMPLE_CSV)
    st.dataframe(sample_df, width="stretch")

    if st.button("Analyze sample texts", type="primary", key="sample_analyze"):
        with st.spinner("Analyzing sample texts…"):
            sample_results = analyze_dataframe(sample_df, model_name=model_name)

        for _, row in sample_results.iterrows():
            render_result(
                {"text": row["text"], "label": row["label"], "score": row["score"]}
            )

        st.download_button(
            label="Download sample results",
            data=sample_results.to_csv(index=False),
            file_name="sample_sentiment_results.csv",
            mime="text/csv",
            width="stretch",
        )

st.markdown(
    f"""
    <div class="footer">
        Sentiment Transformers ·
        <a href="{GITHUB_URL}" target="_blank">GitHub Repository</a> ·
        <a href="{MODEL_URL}" target="_blank">Hugging Face Model</a> ·
        Built with Streamlit + PyTorch
    </div>
    """,
    unsafe_allow_html=True,
)
