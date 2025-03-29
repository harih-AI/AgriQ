import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import traceback
import os
# 1. Define the original class exactly as it was when pickled
class EfficientTamilRAG:
    def __init__(self, data_path='data\dataset_KissanVanni_tamil.csv', sample_size=20000):
        self.df = pd.read_csv(data_path).head(sample_size)
        self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self._precompute_embeddings()
    
    def _precompute_embeddings(self, batch_size=256):
        self.question_embeddings = []
        for i in range(0, len(self.df), batch_size):
            batch = self.df['question'].iloc[i:i+batch_size].tolist()
            self.question_embeddings.extend(self.embedder.encode(batch))
        self.question_embeddings = np.array(self.question_embeddings)
    
    def encode(self, texts):
        return self.embedder.encode(texts)

# 2. Load the model
@st.cache_resource
def load_model():
    try:
        
        model_path = os.path.join('QuantumRAG', 'rag_system.pkl')
        model = joblib.load(model_path)
        st.success("மாதிரி வெற்றிகரமாக ஏற்றப்பட்டது!")  # Model loaded successfully
        return model
    except Exception as e:
        st.error(f"மாதிரியை ஏற்ற முடியவில்லை: {str(e)}")  # Failed to load model
        st.text(traceback.format_exc())
        return None

# Sample data (replace with your actual data)
questions = ["விவசாயத்தில் பயிர் சுழற்சி ஏன் முக்கியம்?", 
            "மண் அரிப்பைத் தடுக்கும் விவசாயம் என்ன?"]
answers = ["இது மண் அரிப்பு மற்றும் குறைவதைத் தடுக்க உதவுகிறது", 
          "பயிர் சுழற்சி முறை"]

# Streamlit UI with Tamil text
st.set_page_config(page_title="தமிழ் விவசாய உதவியாளர்", page_icon="🌱")

# Custom CSS for Tamil font
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Tamil&display=swap');
            
body, input, button, textarea {
    font-family: 'Noto Sans Tamil', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("🌱 தமிழ் விவசாய உதவியாளர்")
st.write("விவசாயம் சார்ந்த உங்கள் கேள்விகளை இங்கே உள்ளிடவும்")

# Load model
model = load_model()

if model is None:
    st.error("தொடர முடியவில்லை. மாதிரி ஏற்றப்படவில்லை.")  # Cannot continue, model not loaded
    st.stop()

# Input section with Tamil labels
with st.form("question_form"):
    user_question = st.text_input("உங்கள் கேள்வி:", placeholder="விவசாயம் பற்றிய கேள்வியை இங்கே தட்டச்சு செய்க...")
    submit_button = st.form_submit_button("பதிலைப் பெறுக")  # Get Answer

if submit_button and user_question:
    try:
        # Generate embeddings
        question_emb = model.encode([user_question])
        corpus_emb = model.encode(questions)
        
        # Find most similar question
        sim_scores = cosine_similarity(question_emb, corpus_emb)[0]
        best_idx = np.argmax(sim_scores)
        
        # Display answer in Tamil
        st.subheader("பதில்:")
        st.success(answers[best_idx])
        
        # Show confidence in Tamil
        confidence = float(sim_scores[best_idx])
        st.write("பொருத்தம்:")  # Match
        st.progress(confidence)
        st.caption(f"நம்பகத்தன்மை: {confidence:.2f}")  # Confidence
        
        # Additional Tamil prompts
        st.write("---")
        st.write("இந்த பதில் உதவியாக இருந்ததா?")
        col1, col2 = st.columns(2)
        with col1:
            st.button("ஆம்", help="பதில் பயனுள்ளதாக இருந்தது")  # Yes
        with col2:
            st.button("இல்லை", help="பதில் பயனுள்ளதாக இல்லை")  # No
        
    except Exception as e:
        st.error("ஒரு பிழை ஏற்பட்டது")  # An error occurred
        st.error(f"விவரம்: {str(e)}")  # Details
        st.text(traceback.format_exc())
elif submit_button and not user_question:
    st.warning("தயவு செய்து ஒரு கேள்வியை உள்ளிடவும்")  # Please enter a question