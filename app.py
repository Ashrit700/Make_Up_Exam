import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Next Word Predictor", layout="wide")

st.title("Next Word Prediction using LSTM")
st.markdown("Developed for the Next Word Prediction project.")

# Utility functions to cache the dataset loading and preprocessing
@st.cache_data
def load_dataset(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    sentences = text.split("\n")
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

@st.cache_data
def preprocess_text(sentences):
    tokenizer = Tokenizer()
    # Fit on texts (lowercasing and tokenization happens here automatically)
    tokenizer.fit_on_texts(sentences)
    total_words = len(tokenizer.word_index) + 1
    
    # Sequence generation
    input_sequences = []
    for line in sentences:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
            
    # Padding sequences
    max_sequence_len = max([len(x) for x in input_sequences]) if input_sequences else 0
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    X, y = input_sequences[:,:-1], input_sequences[:,-1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)
    
    return tokenizer, max_sequence_len, total_words, X, y, input_sequences

# Model building and training
def build_and_train_model(X, y, max_sequence_len, total_words, epochs=50):
    model = Sequential()
    model.add(Embedding(input_dim=total_words, output_dim=100, input_length=max_sequence_len-1))
    model.add(LSTM(150))
    model.add(Dense(total_words, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Create an empty placeholder for the progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Custom callback to update Streamlit UI during training
    class StreamlitCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {logs['loss']:.4f} - Accuracy: {logs['accuracy']:.4f}")

    history = model.fit(X, y, epochs=epochs, verbose=0, callbacks=[StreamlitCallback()])
    
    return model, history

# Prediction
def predict_next_words(model, tokenizer, max_sequence_len, text, top_k=3):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)[0]
    
    # Get top k indices
    top_indices = np.argsort(predicted_probs)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        word = next((w for w, i in tokenizer.word_index.items() if i == idx), None)
        if word:
            results.append((word, predicted_probs[idx]))
            
    return results

# Autocomplete
def autocomplete_sentence(model, tokenizer, max_sequence_len, text, words_to_add=3):
    for _ in range(words_to_add):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_idx = np.argmax(predicted_probs)
        word = next((w for w, i in tokenizer.word_index.items() if i == predicted_idx), None)
        if word:
            text += " " + word
        else:
            break
    return text

# UI
tab1, tab2, tab3, tab4 = st.tabs(["Task 1 & 2: Dataset & Preprocessing", "Task 3: LSTM Model", "Task 4: Next Word Prediction", "Task 5: Performance Visualization"])

dataset_path = "dataset.txt"
sentences = load_dataset(dataset_path)

if sentences:
    tokenizer, max_seq_len, total_vocab, X, y, input_seqs = preprocess_text(sentences)
    
    with tab1:
        st.header("Task 1: Dataset Creation & Task 2: NLP Preprocessing")
        st.write(f"Loaded **{len(sentences)}** meaningful sentences from `{dataset_path}`.")
        
        st.subheader("NLP Preprocessing Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Vocabulary Size", total_vocab)
        col2.metric("Maximum Sequence Length", max_seq_len)
        col3.metric("Total Sequences Generated", len(input_seqs))
        
        st.subheader("Sample Processed Sequences (Padded)")
        st.write("Below are the numerical representations of the generated n-gram sequences, padded to the maximum sequence length:")
        st.code(input_seqs[:5])
        
        st.subheader("Vocabulary Mapping (Sample)")
        st.write("A peek into the tokenized vocabulary (Word -> Index Mapping):")
        sample_vocab = dict(list(tokenizer.word_index.items())[:20])
        st.json(sample_vocab)
        
    with tab2:
        st.header("Task 3: LSTM Model Development")
        st.write("This tab allows you to train the deep learning model. The architecture is as follows:")
        st.code('''
model = Sequential()
model.add(Embedding(input_dim=total_vocab, output_dim=100, input_length=max_sequence_len-1))
model.add(LSTM(150))
model.add(Dense(total_vocab, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        ''', language='python')
        
        epochs = st.slider("Select Epochs for Training", min_value=10, max_value=200, value=50, step=10)
        
        if st.button("Train Model"):
            with st.spinner("Initializing Training..."):
                model, history = build_and_train_model(X, y, max_seq_len, total_vocab, epochs=epochs)
                st.session_state['model'] = model
                st.session_state['history'] = history.history
            st.success("Model trained successfully! You can now predict words and view visualizations.")

    with tab3:
        st.header("Task 4: Next Word Prediction")
        
        if 'model' not in st.session_state:
            st.warning("⚠️ Please train the model first in the 'Task 3: LSTM Model' tab.")
        else:
            model = st.session_state['model']
            
            st.write("Enter a sentence fragment (e.g., 'Artificial Intelligence is', 'Machine learning can')")
            user_input = st.text_input("Sentence Fragment", "Artificial Intelligence is")
            
            if st.button("Predict Next Word", type="primary"):
                if user_input.strip() == "":
                    st.error("Please enter some text.")
                else:
                    predictions = predict_next_words(model, tokenizer, max_seq_len, user_input, top_k=3)
                    
                    if predictions:
                        top_word = predictions[0][0]
                        st.success(f"**Predicted Next Word:** `{top_word}`")
                        
                        st.subheader("Top-3 Predicted Words (Optional Task 5)")
                        for i, (word, prob) in enumerate(predictions):
                            st.write(f"**{i+1}. {word}** ({prob*100:.2f}%)")
                            st.progress(float(prob))
                            
                        st.subheader("Sentence Auto-completion (Optional Task 5)")
                        words_to_add = st.slider("Number of words to auto-complete", 1, 10, 3)
                        completed = autocomplete_sentence(model, tokenizer, max_seq_len, user_input, words_to_add=words_to_add)
                        st.info(f"**Auto-completed Sentence:** {completed}")
                    else:
                        st.error("Could not predict the next word. The word might not be in the vocabulary.")
                        
    with tab4:
        st.header("Task 5: Performance Visualization")
        if 'history' not in st.session_state:
            st.warning("⚠️ Please train the model first in the 'Task 3: LSTM Model' tab to view visualizations.")
        else:
            history = st.session_state['history']
            
            st.write("Below are the Training Accuracy and Loss graphs recorded during the model training phase.")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Accuracy plot
            ax1.plot(history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Loss plot
            ax2.plot(history['loss'], label='Training Loss', color='red', linewidth=2)
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            st.pyplot(fig)
else:
    st.error(f"Dataset not found at `{dataset_path}`. Please ensure the file exists.")
