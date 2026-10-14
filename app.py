"""
SentimentScope - Interactive Streamlit Web Application
Real-time sentiment classification & Attention XAI using a custom Transformer built in PyTorch.
"""
import html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.infer import SentimentPredictor
from src.config import DEFAULT_CONFIG

# Page configuration
st.set_page_config(
    page_title="SentimentScope | PyTorch Transformer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling to optimize layout and reduce dead whitespace
st.markdown("""
<style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #4a6fa5;
        margin-bottom: 10px;
    }
    .token-box {
        display: inline-block;
        padding: 3px 8px;
        margin: 3px 2px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 0.92rem;
        transition: transform 0.1s ease;
    }
    .token-box:hover {
        transform: scale(1.06);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading PyTorch Transformer checkpoint & BERT tokenizer...")
def load_predictor():
    """Caches the predictor instance in memory for instantaneous inference."""
    return SentimentPredictor(checkpoint_path="sentiment_transformer.pt")


# Initialize session state for review text
if "review_input" not in st.session_state:
    st.session_state.review_input = "The cinematography was breathtaking, with a gripping narrative and powerhouse performances from the entire cast!"

# Pre-defined sample reviews
SAMPLE_REVIEWS = {
    "🌟 Positive (Rave Review)": (
        "An absolute masterpiece of modern cinema! The directing was sharp, the score was mesmerizing, "
        "and every single scene delivered authentic emotional resonance. A must-watch experience."
    ),
    "👎 Negative (Critical Pan)": (
        "A dull, completely predictable mess with zero character development. The pacing dragged endlessly "
        "and the dialogue felt completely robotic and uninspired. Save your money and skip this."
    ),
    "🎭 Nuanced (Mixed Sentiment)": (
        "The visual effects and lead actors were undeniably impressive, but the chaotic script and abrupt ending "
        "ultimately prevented it from reaching its full potential."
    )
}

def set_sample_text(text: str):
    st.session_state.review_input = text


# --- SIDEBAR: Architecture & Specs ---
with st.sidebar:
    st.title("🧠 Architecture Specs")
    st.caption("Custom PyTorch Transformer from scratch")

    st.markdown("---")
    st.subheader("Model Configuration")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.metric("Layers", DEFAULT_CONFIG["layers_num"])
        st.metric("Heads", DEFAULT_CONFIG["heads_num"])
        st.metric("Context Size", f"{DEFAULT_CONFIG['context_size']} tokens")
    with col_sb2:
        st.metric("Embedding Dim", DEFAULT_CONFIG["d_embed"])
        st.metric("Head Dim", DEFAULT_CONFIG["head_size"])
        st.metric("Vocab Size", "30,522")

    st.markdown("---")
    st.subheader("Evaluation Benchmarks")
    st.markdown("""
    - **Test Accuracy**: **76.88%** *(IMDB 25,000 test set)*
    - **Macro F1-Score**: **0.7688**
    - **Official Review**: **Passed on Attempt 1** (100% rubric compliance)
    - **Loss Progression**: `0.566` (Ep 1) ➔ `0.428` (Ep 3)
    """)

    st.markdown("---")
    st.subheader("🚀 Cloud Deployment")
    st.markdown("""
    This app is **100% ready for Hugging Face Spaces**.
    - Free CPU hosting via Hugging Face.
    - Zero external infrastructure needed.
    """)

    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[📁 GitHub Repository](https://github.com/JeffersonConza/sentiment-analysis-transformer)")
    st.markdown("[📜 Review Report](docs/REVIEW_FEEDBACK.md)")
    st.markdown("[📓 Notebook Walkthrough](docs/NOTEBOOK_WALKTHROUGH.md)")


# --- MAIN HEADER ---
st.title("🎬 SentimentScope")
st.subheader("Explainable Sentiment Analysis with a From-Scratch PyTorch Transformer")
st.markdown("""
An interactive NLP system implementing a **custom 4-layer Transformer** in pure PyTorch for binary movie review classification.
Featuring real-time prediction and **Multi-Head Self-Attention Explainability (XAI)**.
""")

st.markdown("---")

# Balanced 2-Column Layout to eliminate dead space
col_main, col_results = st.columns([1.1, 0.9], gap="large")

with col_main:
    st.markdown("### ✍️ Input Movie Review")
    
    st.caption("Quick Load Sample Reviews:")
    sample_cols = st.columns(len(SAMPLE_REVIEWS))
    for i, (label, text) in enumerate(SAMPLE_REVIEWS.items()):
        if sample_cols[i].button(label, use_container_width=True):
            set_sample_text(text)

    review_text = st.text_area(
        "Enter or paste movie review text below:",
        value=st.session_state.review_input,
        height=135,
        placeholder="Type your movie review here..."
    )

    analyze_clicked = st.button("🚀 Analyze Sentiment & Explain Attention", type="primary", use_container_width=True)

    # Execute Prediction and Attention Extraction
    result = None
    if review_text.strip():
        try:
            predictor = load_predictor()
            with st.spinner("Analyzing sequence through multi-head self-attention layers..."):
                result = predictor.predict(review_text, return_attention=True)
        except Exception as e:
            st.exception(f"Error performing inference: {e}")

    # Directly beneath the input: Explainable Attention Diagnostics
    if result is not None:
        is_positive = result["label"] == "Positive"
        
        st.markdown("---")
        st.markdown("#### 🔬 Explainable Attention Diagnostics")
        tab_xai, tab_matrix, tab_diag = st.tabs([
            "🔍 Attention Heatmap (XAI)",
            "🗺️ 2D Attention Matrix",
            "📐 Sequence Diagnostics"
        ])

        with tab_xai:
            st.caption(
                "Each word is highlighted by the attention weight received from all positions in Layer 4. "
                "Darker shades indicate key tokens guiding the classification."
            )
            if "tokens" in result and result["tokens"]:
                html_tokens = []
                for t in result["tokens"]:
                    tok_str = t["token"]
                    norm = t["normalized"]
                    
                    # Opacity scaling [0.15, 0.85]
                    alpha = 0.15 + (0.70 * norm)
                    
                    if is_positive:
                        bg = f"rgba(46, 204, 113, {alpha:.2f})"
                        border = "rgba(39, 174, 96, 0.5)"
                        text_col = "#0e4b24" if norm > 0.6 else "#1b3823"
                    else:
                        bg = f"rgba(231, 76, 60, {alpha:.2f})"
                        border = "rgba(192, 57, 43, 0.5)"
                        text_col = "#5a130b" if norm > 0.6 else "#381c18"

                    clean_tok = html.escape(tok_str)
                    score_val = t["score"]
                    html_tokens.append(
                        f'<span class="token-box" style="background-color: {bg}; border: 1px solid {border}; '
                        f'color: {text_col};" title="Attention Score: {score_val:.4f}">{clean_tok}</span>'
                    )

                st.markdown(
                    f'<div style="background-color: #fdfdfd; padding: 12px; border-radius: 8px; '
                    f'border: 1px solid #e0e0e0; line-height: 2.1;">{" ".join(html_tokens)}</div>',
                    unsafe_allow_html=True
                )

        with tab_matrix:
            st.caption("Pairwise query-to-key attention weights across sequence positions in Layer 4.")
            if "attention_matrix" in result and result["attention_matrix"]:
                matrix = np.array(result["attention_matrix"])
                limit = min(len(result["tokens"]), 14)
                sub_matrix = matrix[:limit, :limit]
                labels = [t["token"] for t in result["tokens"][:limit]]

                fig, ax = plt.subplots(figsize=(5.5, 4.5))
                im = ax.imshow(sub_matrix, cmap="Blues", interpolation="nearest")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                ax.set_xticks(range(limit))
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
                ax.set_yticks(range(limit))
                ax.set_yticklabels(labels, fontsize=8)
                ax.set_title("Scaled Dot-Product Attention (Layer 4)", fontsize=10, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        with tab_diag:
            col_d1, col_d2 = st.columns(2)
            col_d1.metric("Input Tokens", result["token_count"])
            col_d2.metric("Tokens Processed", f"{result['tokens_used']} / {DEFAULT_CONFIG['context_size']}")
            
            if result["token_count"] > DEFAULT_CONFIG["context_size"]:
                st.warning(
                    f"⚠️ Review length ({result['token_count']} tokens) exceeds maximum context window "
                    f"({DEFAULT_CONFIG['context_size']} tokens). Input was automatically truncated to bound "
                    r"$\mathcal{O}(N^2)$ self-attention memory overhead."
                )
            else:
                st.info(
                    f"ℹ️ Input length fits within the {DEFAULT_CONFIG['context_size']}-token context window "
                    "and was padded to standard dimension."
                )
    elif not review_text.strip():
        st.info("👈 Please enter review text to inspect attention.")


with col_results:
    st.markdown("### 📊 Real-Time Classification")
    
    if result is not None:
        is_positive = result["label"] == "Positive"
        confidence = result["confidence"]
        prob_pos = result["probabilities"]["Positive"]
        prob_neg = result["probabilities"]["Negative"]

        # Sentiment Verdict Banner
        if is_positive:
            st.success(f"### 🟢 Sentiment: **Positive** ({confidence}%)")
        else:
            st.error(f"### 🔴 Sentiment: **Negative** ({confidence}%)")

        # Probability Distribution Bars
        st.markdown("#### Confidence Breakdown")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.metric("Positive Probability", f"{prob_pos}%")
            st.progress(prob_pos / 100.0)
        with col_p2:
            st.metric("Negative Probability", f"{prob_neg}%")
            st.progress(prob_neg / 100.0)

        # Prominent Top Influential Words Chart
        content_tokens = [
            t for t in result["tokens"]
            if t["token"] not in ["[CLS]", "[SEP]", "[PAD]"] and not t["token"].startswith("##")
        ]
        if content_tokens:
            st.markdown("---")
            st.markdown("##### 🏆 Most Influential Words in Review")
            top_sorted = sorted(content_tokens, key=lambda x: x["score"], reverse=True)[:6]
            df_chart = pd.DataFrame({
                "Word": [t["token"] for t in top_sorted],
                "Attention Weight": [t["score"] for t in top_sorted]
            }).set_index("Word")
            
            chart_color = "#2ecc71" if is_positive else "#e74c3c"
            st.bar_chart(df_chart, horizontal=True, color=chart_color)

        # Quick Architecture Specs
        st.markdown("---")
        st.markdown("##### ⚡ Architecture Quick-Glance")
        col_q1, col_q2, col_q3 = st.columns(3)
        col_q1.metric("Layers", DEFAULT_CONFIG["layers_num"])
        col_q2.metric("Heads", DEFAULT_CONFIG["heads_num"])
        col_q3.metric("Context", f"{DEFAULT_CONFIG['context_size']} tok")
    else:
        st.info("👈 Enter a review on the left and click **Analyze Sentiment** to view predictions.")

st.markdown("---")
st.caption("Built with PyTorch & Streamlit • Developed by Jefferson Conza • AWS AI & ML Scholars")
