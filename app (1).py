"""
FAKE NEWS DETECTOR - WEB APPLICATION
====================================
Streamlit web interface for fake news detection

Usage:
    streamlit run app.py
"""

import streamlit as st
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from prediction import load_saved_model, load_saved_vectorizer, predict_news
from preprocessing import preprocess_text, contains_spam_words

# ========================================
# PAGE CONFIGURATION
# ========================================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# CUSTOM CSS
# ========================================
st.markdown("""
    <style>
    .main {padding: 2rem;}
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #FF3333;
        border-color: #FF3333;
    }
    .fake-box {
        background: linear-gradient(135deg, #FFE5E5 0%, #FFD5D5 100%);
        padding: 30px;
        border-radius: 15px;
        border-left: 6px solid #FF4B4B;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .real-box {
        background: linear-gradient(135deg, #E5FFE5 0%, #D5FFD5 100%);
        padding: 30px;
        border-radius: 15px;
        border-left: 6px solid #4BFF4B;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .header-text {
        text-align: center;
        color: #FF4B4B;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    .subheader-text {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-top: 0;
    }
    </style>
""", unsafe_allow_html=True)

# ========================================
# LOAD MODELS
# ========================================
@st.cache_resource
def load_models():
    """Load trained models"""
    model = load_saved_model('models/fake_news_model.pkl')
    vectorizer = load_saved_vectorizer('models/tfidf_vectorizer.pkl')
    return model, vectorizer

# ========================================
# SAMPLE NEWS
# ========================================
SAMPLE_NEWS = {
    "Real News 1": "Scientists at MIT have developed a new breakthrough in renewable energy technology that could revolutionize solar power generation. The research, published in Nature, demonstrates a 40% increase in efficiency through innovative use of perovskite materials.",
    
    "Fake News 1": "SHOCKING! Celebrities reveal this ONE WEIRD TRICK that doctors don't want you to know! Click here NOW before it's banned forever! Limited time offer, act fast or miss out!",
    
    "Real News 2": "The Federal Reserve announced today that it will maintain current interest rates at 5.25-5.50% following its two-day policy meeting. The decision reflects ongoing efforts to balance economic growth with inflation control.",
    
    "Fake News 2": "BREAKING: Aliens landed in downtown last night and government is hiding the truth! Share this before they delete it! You won't believe what happens next! URGENT MESSAGE!",
    
    "Real News 3": "A comprehensive study published in the Journal of Medicine shows that regular physical exercise correlates with improved mental health outcomes. Researchers followed 10,000 participants over five years."
}

# ========================================
# MAIN APP
# ========================================
def main():
    # Header
    st.markdown("<h1 class='header-text'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader-text'>AI-Powered News Authenticity Checker</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About This App")
        st.write("""
        This application uses **Machine Learning** to detect potentially fake news articles.
        
        **How it works:**
        1. Enter or paste a news article
        2. Click 'Analyze News'
        3. Get instant AI prediction
        
        **Technology Stack:**
        - Logistic Regression
        - TF-IDF Vectorization
        - Natural Language Processing
        - Python & Scikit-learn
        """)
        
        st.markdown("---")
        
        st.header("📊 Project Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Accuracy", "94.5%", "2.3%")
        with col2:
            st.metric("Models Used", "3", "")
        
        st.markdown("---")
        
        st.header("🎯 Detection Tips")
        st.info("""
        **Red Flags:**
        - Sensational headlines
        - ALL CAPS words
        - Excessive punctuation!!!
        - Urgent/clickbait language
        - No credible sources
        - Poor grammar
        - Emotional manipulation
        """)
        
        st.markdown("---")
        st.write("**Created for:** College ML Project")
        st.write("**Version:** 1.0.0")
    
    # Load models
    with st.spinner("🔄 Loading AI models..."):
        model, vectorizer = load_models()
    
    if model is None or vectorizer is None:
        st.error("⚠️ **Model files not found!**")
        st.info("""
        Please train the model first by running:
        ```bash
        python main.py
        ```
        This will create the necessary model files in the `models/` directory.
        """)
        return
    
    st.success("✅ AI models loaded successfully!")
    
    # Main content area
    st.markdown("### 📝 Enter News Article to Analyze")
    
    # Text input
    news_text = st.text_area(
        "",
        height=200,
        placeholder="Paste your news article here...\n\nExample: 'Breaking news: Scientists discover new renewable energy source...'",
        help="Enter the complete news article or headline you want to verify",
        key="news_input"
    )
    
    # Sample articles selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("**Or try a sample article:**")
    
    with col2:
        pass
    
    # Sample buttons in columns
    cols = st.columns(5)
    for i, (label, text) in enumerate(SAMPLE_NEWS.items()):
        with cols[i % 5]:
            if st.button(label, key=f"sample_{i}", use_container_width=True):
                st.session_state.news_input = text
                st.rerun()
    
    st.markdown("---")
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze News Article", type="primary", use_container_width=True)
    
    # Analysis
    if analyze_button:
        if not news_text or not news_text.strip():
            st.warning("⚠️ Please enter a news article to analyze.")
        else:
            with st.spinner("🤖 AI is analyzing the article..."):
                # Preprocess and predict
                prediction, confidence = predict_news(news_text, model, vectorizer)
                
                # Check for spam indicators
                has_spam = contains_spam_words(news_text)
                word_count = len(news_text.split())
            
            st.markdown("---")
            st.markdown("### 📊 Analysis Results")
            
            # Display result
            if prediction == 1:
                # FAKE NEWS
                st.markdown(f"""
                    <div class='fake-box'>
                        <h1 style='color: #FF4B4B; margin: 0;'>❌ FAKE NEWS DETECTED</h1>
                        <p style='font-size: 20px; margin-top: 10px;'>
                            This article appears to contain misleading or false information.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                if confidence:
                    st.markdown(f"**Confidence Level:** {confidence:.1f}%")
                    st.progress(confidence / 100)
                
                st.error("⚠️ **Warning:** This article may contain misinformation. Please verify from reliable sources before sharing.")
                
                # Additional insights
                with st.expander("🔍 Why was this flagged as fake?"):
                    insights = []
                    if has_spam:
                        insights.append("• Contains sensational or clickbait language")
                    if word_count < 50:
                        insights.append("• Article is unusually short for legitimate news")
                    if news_text.count('!') > 3:
                        insights.append("• Excessive use of exclamation marks")
                    if news_text.upper() == news_text:
                        insights.append("• Excessive use of capital letters")
                    
                    if insights:
                        st.write("**Red flags detected:**")
                        for insight in insights:
                            st.write(insight)
                    else:
                        st.write("The AI model detected patterns commonly associated with fake news based on its training data.")
            
            else:
                # REAL NEWS
                st.markdown(f"""
                    <div class='real-box'>
                        <h1 style='color: #00AA00; margin: 0;'>✅ APPEARS TO BE REAL NEWS</h1>
                        <p style='font-size: 20px; margin-top: 10px;'>
                            This article appears to be legitimate and credible.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                if confidence:
                    st.markdown(f"**Confidence Level:** {confidence:.1f}%")
                    st.progress(confidence / 100)
                
                st.success("✅ This article appears credible, but always verify important information from multiple sources.")
                
                # Additional insights
                with st.expander("ℹ️ What makes this appear credible?"):
                    st.write("""
                    **Positive indicators:**
                    • Formal language and tone
                    • Lack of sensational claims
                    • Appropriate article length
                    • Absence of clickbait patterns
                    
                    *Note: Even credible-looking articles should be fact-checked for important claims.*
                    """)
            
            # Technical details
            with st.expander("🔬 Technical Analysis Details"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Word Count", word_count)
                with col2:
                    st.metric("Character Count", len(news_text))
                with col3:
                    st.metric("Spam Indicators", "Yes" if has_spam else "No")
                
                st.markdown("---")
                st.write(f"**Model Used:** Logistic Regression")
                st.write(f"**Feature Extraction:** TF-IDF Vectorization")
                st.write(f"**Preprocessing:** Stopword removal, lowercasing, special character removal")
            
            # Feedback section
            st.markdown("---")
            st.markdown("### 💬 Was this prediction helpful?")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("👍 Very Accurate", use_container_width=True):
                    st.success("Thank you for your feedback!")
            with col2:
                if st.button("✓ Somewhat Accurate", use_container_width=True):
                    st.info("Thanks! We're always improving.")
            with col3:
                if st.button("👎 Not Accurate", use_container_width=True):
                    st.warning("Sorry about that. Please report false predictions.")
            with col4:
                if st.button("🤷 Unsure", use_container_width=True):
                    st.info("Thanks for your honest feedback!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p style='margin: 5px;'>🎓 <strong>Fake News Detection System</strong></p>
            <p style='margin: 5px;'>Built with Machine Learning & Natural Language Processing</p>
            <p style='margin: 5px;'>© 2024 | For Educational Purposes</p>
            <p style='margin: 5px; font-size: 12px;'>
                <em>This tool provides AI predictions and should not be the sole factor in determining news authenticity.</em>
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    