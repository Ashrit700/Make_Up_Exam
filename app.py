import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import json
import pickle
import os

st.set_page_config(page_title="Next Word Predictor", layout="wide")

st.title("🔮 Next Word Prediction with LSTM")
st.markdown("Complete sentences and predict next words using a trained LSTM model.")

# Load model and artifacts
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("saved_model/next_word_lstm.keras")
    with open("saved_model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("saved_model/metadata.json", "r") as f:
        metadata = json.load(f)
    return model, tokenizer, metadata

@st.cache_data
def load_training_history():
    with open("saved_model/history.json", "r") as f:
        history = json.load(f)
    return history

# Prediction function
def predict_next_words(model, tokenizer, metadata, text, top_k=3):
    max_sequence_len = metadata["max_sequence_len"]
    try:
        token_list = tokenizer.texts_to_sequences([text])[0]
        if not token_list:  # Handle unknown words
            return []
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        
        # Filter out low-probability predictions and padding token (index 0)
        min_prob = 0.01
        valid_indices = np.where((predicted_probs > min_prob) & (np.arange(len(predicted_probs)) != 0))[0]
        
        if len(valid_indices) == 0:
            top_indices = np.argsort(predicted_probs)[-top_k:][::-1]
        else:
            top_indices = valid_indices[np.argsort(predicted_probs[valid_indices])[-top_k:][::-1]]
        
        results = []
        for idx in top_indices:
            if idx == 0:  # Skip padding token
                continue
            word = next((w for w, i in tokenizer.word_index.items() if i == idx), None)
            if word and word.strip():  # Only valid, non-empty words
                results.append((word, float(predicted_probs[idx])))
        return results
    except Exception as e:
        st.warning(f"Could not generate predictions: {str(e)}")
        return []

# Autocomplete function
def autocomplete_sentence(model, tokenizer, metadata, text, words_to_add=3):
    max_sequence_len = metadata["max_sequence_len"]
    completed = text
    for _ in range(words_to_add):
        try:
            token_list = tokenizer.texts_to_sequences([completed])[0]
            if not token_list:  # Handle unknown words
                break
            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
            predicted_probs = model.predict(token_list, verbose=0)[0]
            
            # Find highest probability that's not padding token (index 0)
            valid_mask = np.arange(len(predicted_probs)) != 0
            valid_probs = predicted_probs.copy()
            valid_probs[~valid_mask] = -1
            
            predicted_idx = np.argmax(valid_probs)
            if predicted_probs[predicted_idx] < 0.01:  # Skip if probability too low
                break
                
            word = next((w for w, i in tokenizer.word_index.items() if i == predicted_idx), None)
            if word and word.strip():  # Only valid, non-empty words
                completed += " " + word
            else:
                break
        except Exception as e:
            break
    return completed

# Load model and data
try:
    model, tokenizer, metadata = load_model_and_tokenizer()
    history = load_training_history()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📝 Predict Next Word", "✍️ Complete Sentence", "📊 Model Evaluation"])
    
    with tab1:
        st.header("Predict Next Word")
        st.write("Enter a sentence fragment and get the top predicted next words with confidence scores.")
        
        user_text = st.text_input("Enter text:", placeholder="e.g., 'artificial intelligence is'", key="predict_input")
        
        if user_text:
            try:
                predictions = predict_next_words(model, tokenizer, metadata, user_text, top_k=5)
                if predictions:
                    st.success(f"✓ Top predictions for: **{user_text}**")
                    cols = st.columns(len(predictions))
                    for i, (word, prob) in enumerate(predictions):
                        with cols[i]:
                            st.metric(label=f"#{i+1}", value=word, delta=f"{prob*100:.1f}%")
                else:
                    st.warning("No predictions available.")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
    
    with tab2:
        st.header("Complete Sentence")
        st.write("Start typing a sentence and let the model complete it automatically.")
        
        user_input = st.text_input("Enter incomplete sentence:", placeholder="e.g., 'machine learning'", key="complete_input")
        num_words = st.slider("Words to add:", 1, 10, 3)
        
        if user_input:
            try:
                completed = autocomplete_sentence(model, tokenizer, metadata, user_input, words_to_add=num_words)
                st.info(f"**Original:** {user_input}")
                st.success(f"**Completed:** {completed}")
            except Exception as e:
                st.error(f"Error during completion: {e}")
    
    with tab3:
        st.header("Model Evaluation Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Epochs", metadata["epochs"])
            st.metric("Vocabulary Size", metadata["total_words"])
            st.metric("Max Sequence Length", metadata["max_sequence_len"])
        
        with col2:
            st.metric("Training Sequences", metadata["num_sequences"])
            st.metric("Dataset Sentences", metadata["num_sentences"])
            final_accuracy = history["accuracy"][-1]
            st.metric("Final Accuracy", f"{final_accuracy*100:.2f}%")
        
        st.subheader("Training History")
        
        # Create figure with loss and accuracy
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1.plot(history["loss"], linewidth=2, color='#e74c3c')
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title("Training Loss Over Epochs", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(history["accuracy"], linewidth=2, color='#2ecc71')
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.set_title("Training Accuracy Over Epochs", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
except FileNotFoundError:
    st.error("⚠️ Model files not found. Please run `train_model.py` first to train and save the model.")
