"""
SentimentScope - Interactive Streamlit Web Application
Single-Screen Viewport Dashboard with Dedicated Attention Heatmap Hero Showcase.
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

# Custom High-Density CSS
st.markdown("""
<style>
    /* Full viewport utilization */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1.8rem !important;
        padding-right: 1.8rem !important;
        max-width: 100% !important;
    }
    h1, h2, h3, h4, h5, p {
        margin-top: 0.1rem !important;
        margin-bottom: 0.25rem !important;
    }
    /* Token badge formatting */
    .token-box {
        display: inline-block;
        padding: 3px 7px;
        margin: 2px 2px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 0.90rem;
        transition: transform 0.1s ease, box-shadow 0.1s ease;
        cursor: default;
    }
    .token-box:hover {
        transform: scale(1.08);
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
    }
    .token-special {
        background-color: #f1f5f9 !important;
        color: #64748b !important;
        border: 1px dashed #cbd5e1 !important;
        font-style: italic;
    }
    /* Hero container for the Attention Heatmap */
    .heatmap-hero-container {
        background: #ffffff;
        border: 1.5px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 14px;
        min-height: 135px;
        max-height: 200px;
        overflow-y: auto;
        line-height: 2.1;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    /* Streamlit tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 3px 8px !important;
        font-size: 0.82rem !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading PyTorch Transformer & BERT tokenizer...")
def load_predictor():
    """Caches the predictor instance in memory for instantaneous inference."""
    return SentimentPredictor(checkpoint_path="sentiment_transformer.pt")


# Initialize session state for review text
if "review_input" not in st.session_state:
    st.session_state.review_input = "An absolute masterpiece! A truly brilliant and fantastic film with wonderful performances and exceptional directing."

# Pre-defined sample reviews
SAMPLE_REVIEWS = {
    "🌟 Positive": (
        "An absolute masterpiece! A truly brilliant and fantastic film with wonderful performances and exceptional directing."
    ),
    "👎 Negative": (
        "A dull, completely predictable mess with zero character development. The pacing dragged endlessly "
        "and the dialogue felt completely robotic and uninspired."
    ),
    "🎭 Nuanced": (
        "The visual effects and lead actors were undeniably impressive, but the chaotic script and abrupt ending "
        "prevented it from reaching its potential."
    )
}

def set_sample_text(text: str):
    st.session_state.review_input = text


# Stopwords list to surface genuine semantic driver words in the influence chart
COMMON_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "it", "its",
    "this", "that", "these", "those", "i", "you", "he", "she", "we", "they", "my",
    "your", "his", "her", "our", "their", "[cls]", "[sep]", "[pad]"
}


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
    - **Official Review**: **Passed on Attempt 1** (100% rubric)
    - **Loss Progression**: `0.566` (Ep 1) ➔ `0.428` (Ep 3)
    """)

    st.markdown("---")
    st.subheader("🚀 Cloud Deployment")
    st.markdown("Ready for **Hugging Face Spaces** free CPU hosting.")

    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[📁 GitHub Repository](https://github.com/JeffersonConza/sentiment-analysis-transformer)")
    st.markdown("[📜 Review Report](docs/REVIEW_FEEDBACK.md)")
    st.markdown("[📓 Notebook Walkthrough](docs/NOTEBOOK_WALKTHROUGH.md)")


# --- ULTRA-COMPACT 1-LINE HEADER ---
st.markdown("""
<div style="display: flex; align-items: baseline; justify-content: space-between; border-bottom: 2px solid #f1f5f9; padding-bottom: 4px; margin-bottom: 8px;">
    <div>
        <span style="font-size: 1.40rem; font-weight: 800; color: #0f172a;">🎬 SentimentScope</span>
        <span style="font-size: 0.88rem; color: #64748b; margin-left: 10px;">From-Scratch PyTorch Transformer & Explainable AI (XAI)</span>
    </div>
    <div>
        <span style="background: #e0e7ff; color: #3730a3; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: 600;">Test Acc: 76.88%</span>
        <span style="background: #dcfce7; color: #166534; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: 600; margin-left: 4px;">Passed Attempt 1</span>
    </div>
</div>
""", unsafe_allow_html=True)


# Execute prediction immediately so all sections have data ready
result = None
if st.session_state.review_input.strip():
    try:
        predictor = load_predictor()
        result = predictor.predict(st.session_state.review_input, return_attention=True)
    except Exception as e:
        st.error(f"Inference error: {e}")


# ==============================================================================
# ROW 1: CONTROL & VERDICT HUB (Top Half)
# ==============================================================================
col_top_left, col_top_right = st.columns([1.1, 0.9], gap="medium")

with col_top_left:
    # Header & Sample Buttons in a single aligned row
    col_hdr, col_s1, col_s2, col_s3 = st.columns([1.1, 0.9, 0.9, 0.9])
    with col_hdr:
        st.markdown("<div style='font-weight: 700; font-size: 0.92rem; margin-top: 4px;'>✍️ Input Review</div>", unsafe_allow_html=True)
    with col_s1:
        if st.button("🌟 Positive", use_container_width=True):
            set_sample_text(SAMPLE_REVIEWS["🌟 Positive"])
            st.rerun()
    with col_s2:
        if st.button("👎 Negative", use_container_width=True):
            set_sample_text(SAMPLE_REVIEWS["👎 Negative"])
            st.rerun()
    with col_s3:
        if st.button("🎭 Nuanced", use_container_width=True):
            set_sample_text(SAMPLE_REVIEWS["🎭 Nuanced"])
            st.rerun()

    # Compact text area (74px = ~3 lines)
    review_text = st.text_area(
        label="Review Text",
        value=st.session_state.review_input,
        height=74,
        label_visibility="collapsed",
        placeholder="Type or paste your movie review here..."
    )

    # Analyze Button
    if st.button("🚀 Analyze Sentiment & Extract Attention", type="primary", use_container_width=True):
        st.session_state.review_input = review_text
        st.rerun()


with col_top_right:
    if result is not None:
        is_positive = result["label"] == "Positive"
        confidence = result["confidence"]
        prob_pos = result["probabilities"]["Positive"]
        prob_neg = result["probabilities"]["Negative"]

        verdict_color = "#16a34a" if is_positive else "#dc2626"
        verdict_bg = "#f0fdf4" if is_positive else "#fef2f2"
        verdict_border = "#86efac" if is_positive else "#fca5a5"
        verdict_label = "POSITIVE" if is_positive else "NEGATIVE"
        verdict_icon = "🟢" if is_positive else "🔴"

        # Consolidated Verdict Card + Architecture Badges
        st.markdown(f"""
        <div style="background-color: {verdict_bg}; border: 1.5px solid {verdict_border}; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Sentiment Verdict</span>
                    <div style="font-size: 1.30rem; font-weight: 800; color: {verdict_color};">{verdict_icon} {verdict_label}</div>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Confidence</span>
                    <div style="font-size: 1.30rem; font-weight: 800; color: #0f172a;">{confidence}%</div>
                </div>
            </div>
            <div style="margin-top: 6px;">
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 600; margin-bottom: 2px;">
                    <span style="color: #16a34a;">Positive: {prob_pos}%</span>
                    <span style="color: #dc2626;">Negative: {prob_neg}%</span>
                </div>
                <div style="background-color: #fee2e2; border-radius: 4px; height: 7px; overflow: hidden; display: flex;">
                    <div style="background-color: #22c55e; width: {prob_pos}%; height: 100%;"></div>
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 6px; padding-top: 5px; border-top: 1px solid rgba(0,0,0,0.05); font-size: 0.72rem; color: #64748b;">
                <span>Context: <b>128 tokens</b></span>
                <span>Layers: <b>4 Blocks (4 Heads)</b></span>
                <span>Checkpoint: <b>Verified PT</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👈 Enter a review on the left and click **Analyze Sentiment**.")


# ==============================================================================
# ROW 2: THE SPECIAL ATTENTION & XAI HERO SHOWCASE (Bottom Half)
# ==============================================================================
st.markdown("<div style='margin-top: 6px; margin-bottom: 6px; border-top: 1px solid #f1f5f9;'></div>", unsafe_allow_html=True)

if result is not None:
    is_positive = result["label"] == "Positive"
    col_xai_left, col_xai_right = st.columns([1.12, 0.88], gap="medium")

    # --- LEFT HERO: THE SPECIAL ATTENTION HEATMAP PLACE ---
    with col_xai_left:
        # Title + Interactive Color Legend Bar
        legend_gradient = (
            "linear-gradient(to right, rgba(46, 204, 113, 0.15), rgba(46, 204, 113, 0.85))"
            if is_positive
            else "linear-gradient(to right, rgba(231, 76, 60, 0.15), rgba(231, 76, 60, 0.85))"
        )
        legend_label = "Positive Attention" if is_positive else "Negative Attention"

        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">
            <div style="font-weight: 700; font-size: 0.90rem; color: #1e293b;">
                🔬 Layer 4 Self-Attention Heatmap (XAI)
            </div>
            <div style="display: flex; align-items: center; font-size: 0.72rem; color: #64748b; gap: 5px;">
                <span>Low</span>
                <span style="display: inline-block; width: 55px; height: 7px; border-radius: 4px; background: {legend_gradient};"></span>
                <span>High ({legend_label})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if "tokens" in result and result["tokens"]:
            html_tokens = []
            for t in result["tokens"]:
                tok_str = t["token"]
                norm = t["normalized"]
                score_val = t["score"]
                
                # Check for special tokens
                if tok_str in ["[CLS]", "[SEP]", "[PAD]"]:
                    clean_tok = html.escape(tok_str)
                    html_tokens.append(f'<span class="token-box token-special" title="Special Token">{clean_tok}</span>')
                    continue

                # Opacity scaling [0.15, 0.85]
                alpha = 0.15 + (0.70 * norm)
                
                if is_positive:
                    bg = f"rgba(46, 204, 113, {alpha:.2f})"
                    border = "rgba(39, 174, 96, 0.40)"
                    text_col = "#0e4b24" if norm > 0.6 else "#1b3823"
                else:
                    bg = f"rgba(231, 76, 60, {alpha:.2f})"
                    border = "rgba(192, 57, 43, 0.40)"
                    text_col = "#5a130b" if norm > 0.6 else "#381c18"

                clean_tok = html.escape(tok_str)
                is_subword = tok_str.startswith("##")
                tooltip = (
                    f"WordPiece Subword: '{clean_tok}' (continuation of preceding word) | Attention: {score_val:.4f}"
                    if is_subword
                    else f"Token: '{clean_tok}' | Attention Score: {score_val:.4f}"
                )
                html_tokens.append(
                    f'<span class="token-box" style="background-color: {bg}; border: 1px solid {border}; '
                    f'color: {text_col};" title="{tooltip}">{clean_tok}</span>'
                )

            st.markdown(
                f'<div class="heatmap-hero-container">{" ".join(html_tokens)}</div>',
                unsafe_allow_html=True
            )
            st.caption(
                "Hover tokens to view Layer 4 weights. "
                "💡 *Note*: Tokens with `##` (e.g. `##zing`) are BERT WordPiece subwords used to tokenize compound/rare words."
            )

    # --- RIGHT PANEL: ATTENTION ANALYTICS & DIAGNOSTICS ---
    with col_xai_right:
        tab_chart, tab_matrix, tab_diag = st.tabs([
            "🏆 Top Influential Words",
            "🗺️ 2D Attention Matrix",
            "📐 Context Usage"
        ])

        with tab_chart:
            # Filter stopwords to surface real sentiment-carrying keywords
            all_content = [
                t for t in result["tokens"]
                if t["token"].lower() not in COMMON_STOPWORDS and not t["token"].startswith("##")
            ]
            if not all_content:
                all_content = [t for t in result["tokens"] if t["token"] not in ["[CLS]", "[SEP]", "[PAD]"]]

            if all_content:
                top_sorted = sorted(all_content, key=lambda x: x["score"], reverse=True)[:5]
                df_chart = pd.DataFrame({
                    "Word": [t["token"] for t in top_sorted],
                    "Attention Weight": [t["score"] for t in top_sorted]
                }).set_index("Word")
                
                chart_color = "#22c55e" if is_positive else "#ef4444"
                st.bar_chart(df_chart, horizontal=True, color=chart_color, height=160)
            else:
                st.caption("No content tokens to display.")

        with tab_matrix:
            if "attention_matrix" in result and result["attention_matrix"]:
                matrix = np.array(result["attention_matrix"])
                limit = min(len(result["tokens"]), 12)
                sub_matrix = matrix[:limit, :limit]
                labels = [t["token"] for t in result["tokens"][:limit]]

                fig, ax = plt.subplots(figsize=(4.5, 3.0))
                im = ax.imshow(sub_matrix, cmap="Blues", interpolation="nearest")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                ax.set_xticks(range(limit))
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
                ax.set_yticks(range(limit))
                ax.set_yticklabels(labels, fontsize=7)
                ax.set_title("Pairwise Self-Attention (Layer 4)", fontsize=8, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        with tab_diag:
            col_d1, col_d2 = st.columns(2)
            col_d1.metric("Input Tokens", result["token_count"])
            col_d2.metric("Processed Tokens", f"{result['tokens_used']} / {DEFAULT_CONFIG['context_size']}")
            
            if result["token_count"] > DEFAULT_CONFIG["context_size"]:
                st.warning(
                    f"Truncated: {result['token_count']} > {DEFAULT_CONFIG['context_size']} tokens "
                    r"to bound $\mathcal{O}(N^2)$ attention memory."
                )
            else:
                st.info(f"Fits within the {DEFAULT_CONFIG['context_size']}-token context window.")
