"""
SentimentScope - Interactive Streamlit Web Application
Real-time sentiment classification using a custom Transformer built in PyTorch.
"""
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

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #4a6fa5;
        margin-bottom: 10px;
    }
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 85%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
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
    st.markdown("### 🔗 Links")
    st.markdown("[📁 GitHub Repository](https://github.com/JeffersonConza/sentiment-analysis-transformer)")
    st.markdown("[📜 Review Report](docs/REVIEW_FEEDBACK.md)")


# --- MAIN CONTENT ---
st.title("🎬 SentimentScope")
st.subheader("Transformer-based Movie Review Sentiment Classifier in PyTorch")
st.markdown("""
This demo runs inference using a **custom 4-layer Transformer** implemented from scratch in pure PyTorch.
It combines multi-head self-attention, positional embeddings, and sequence mean pooling for binary sentiment classification.
""")

st.markdown("---")

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
        height=180,
        placeholder="Type your movie review here..."
    )

    analyze_clicked = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)

with col_results:
    st.markdown("### 📊 Classification Output")
    
    if review_text.strip():
        try:
            predictor = load_predictor()
            with st.spinner("Analyzing sequence through multi-head self-attention layers..."):
                result = predictor.predict(review_text)

            is_positive = result["label"] == "Positive"
            confidence = result["confidence"]
            prob_pos = result["probabilities"]["Positive"]
            prob_neg = result["probabilities"]["Negative"]

            # Sentiment Banner Card
            if is_positive:
                st.success(f"### 🟢 Sentiment: **Positive** ({confidence}%)")
            else:
                st.error(f"### 🔴 Sentiment: **Negative** ({confidence}%)")

            # Probability Breakdown
            st.markdown("#### Confidence Probabilities")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("Positive Probability", f"{prob_pos}%")
                st.progress(prob_pos / 100.0)
            with col_p2:
                st.metric("Negative Probability", f"{prob_neg}%")
                st.progress(prob_neg / 100.0)

            # Sequence & Context Window Diagnostics
            with st.expander("🔍 Sequence & Token Diagnostics", expanded=True):
                col_d1, col_d2 = st.columns(2)
                col_d1.metric("Input Tokens", result["token_count"])
                col_d2.metric("Tokens Processed", f"{result['tokens_used']} / {DEFAULT_CONFIG['context_size']}")
                
                if result["token_count"] > DEFAULT_CONFIG["context_size"]:
                    st.warning(
                        f"⚠️ Sequence length ({result['token_count']} tokens) exceeds max context window "
                        f"({DEFAULT_CONFIG['context_size']} tokens). Input was automatically truncated to preserve "
                        r"$\mathcal{O}(N^2)$ memory efficiency."
                    )
                else:
                    st.info(
                        f"ℹ️ Input fits within the {DEFAULT_CONFIG['context_size']}-token context window "
                        "and was padded to standard dimension."
                    )

        except Exception as e:
            st.exception(f"Error performing inference: {e}")
    else:
        st.info("👈 Please enter a review or choose a sample to analyze.")

st.markdown("---")
st.caption("Built with PyTorch & Streamlit • Developed by Jefferson Conza • AWS AI & ML Scholars")
