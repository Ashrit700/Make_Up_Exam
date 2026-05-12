# 🔮 Next Word Prediction with LSTM

A machine learning application that predicts the next word in a sentence using an LSTM neural network model trained on a curated dataset of English sentences.

## 📋 Project Overview

This project implements a complete Next Word Prediction system with three main components:
- **Model Training**: LSTM neural network trained on 401+ sentences
- **Prediction Engine**: Real-time next word prediction with confidence scores
- **Interactive UI**: Streamlit-based user interface for predictions and visualization

## 🎯 Features

### 1. Predict Next Word
- Enter any sentence fragment
- Get top 5 predicted next words with confidence scores
- Visualized in metric cards with probability percentages

### 2. Complete Sentence
- Automatically complete partial sentences
- Add 1-10 words progressively
- Adjustable word count with interactive slider

### 3. Model Evaluation
- Training metrics (epochs, vocabulary, accuracy)
- Interactive graphs showing:
  - Training loss over 100 epochs
  - Training accuracy progression
- Final model statistics

## 📁 Project Structure

```
Make_Up_Exam/
├── app.py                 # Streamlit web application
├── train_model.py         # Model training script
├── dataset.txt            # Training dataset (401+ sentences)
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── saved_model/
    ├── next_word_lstm.keras    # Trained LSTM model
    ├── tokenizer.pkl           # Text tokenizer
    ├── history.json            # Training history
    └── metadata.json           # Model metadata
```

## 🛠 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone/Download the project:**
   ```bash
   cd c:\check\Make_Up_Exam
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Option 1: Run the Web Application
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`

### Option 2: Retrain the Model
```bash
python train_model.py
```
This will:
- Load the dataset from `dataset.txt`
- Preprocess text (tokenization, padding)
- Build and train the LSTM model
- Save all artifacts to `saved_model/`

## 📊 Model Architecture

```
Input Layer
    ↓
Embedding Layer (100 dimensions)
    ↓
LSTM Layer (150 units)
    ↓
Dense Layer (Softmax activation)
    ↓
Output Predictions
```

**Model Specifications:**
- Type: Sequential LSTM
- Embedding Dimension: 100
- LSTM Units: 150
- Loss Function: Categorical Crossentropy
- Optimizer: Adam
- Epochs: 100
- Activation: Softmax

## 📈 Training Results

- **Dataset**: 401+ sentences
- **Vocabulary Size**: 1,846 unique words
- **Training Sequences**: 2,619 n-gram sequences
- **Final Accuracy**: ~96%
- **Training Loss**: Converged to ~0.14

## 🎨 UI Interface

The Streamlit app provides three interactive tabs:

1. **📝 Predict Next Word**
   - Input text field for sentence fragments
   - Real-time predictions with confidence metrics
   - Top 5 predictions displayed

2. **✍️ Complete Sentence**
   - Incomplete sentence input
   - Adjustable word count slider (1-10 words)
   - Shows original and completed sentences

3. **📊 Model Evaluation**
   - Training metrics dashboard
   - Loss and accuracy graphs
   - Model performance statistics

## 🔧 Customization

### Modify Training Parameters
Edit `train_model.py`:
```python
EPOCHS = 100  # Change number of training epochs
```

### Adjust Prediction Threshold
Edit `app.py`:
```python
min_prob = 0.01  # Minimum probability threshold (default: 1%)
```

### Add More Training Data
1. Add sentences to `dataset.txt`
2. Run `python train_model.py` to retrain
3. Restart the Streamlit app

## 📦 Dependencies

- `streamlit` - Web application framework
- `tensorflow` - Deep learning framework
- `numpy` - Numerical computing
- `matplotlib` - Data visualization
- `pandas` - Data manipulation

See `requirements.txt` for versions.

## 🐛 Troubleshooting

### Issue: "Model files not found"
**Solution**: Run `python train_model.py` to train and save the model first.

### Issue: Random predictions without context
**Solution**: This is fixed by filtering low-probability predictions. The model automatically filters predictions below 1% probability.

### Issue: Streamlit app crashes
**Solution**: 
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Clear cache: `streamlit cache clear`
- Restart the app

## 📝 Example Usage

### Predict Next Word
```
Input: "artificial intelligence is"
Output: 
  #1: powerful (87.3%)
  #2: important (8.2%)
  #3: useful (4.5%)
```

### Complete Sentence
```
Input: "machine learning"
Output: "machine learning helps business improve productivity"
```

## 🔬 Technical Details

### NLP Preprocessing Pipeline
1. **Text Lowercasing**: Convert all text to lowercase
2. **Tokenization**: Convert words to integer indices
3. **Sequence Generation**: Create n-gram sequences
4. **Padding**: Pad sequences to uniform length
5. **Encoding**: Convert labels to categorical format

### Prediction Algorithm
1. Convert input text to token sequence
2. Pad sequence to model's expected length
3. Pass through LSTM model
4. Get output probability distribution
5. Select top-K predictions by probability
6. Filter invalid/low-probability tokens
7. Return results with confidence scores

## 📄 License

This project is created for educational purposes.

## 👨‍💻 Development

Last Updated: May 2026
Model Version: 1.1 (Retrained with expanded dataset)

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Verify all dependencies are installed
3. Ensure dataset.txt exists and is properly formatted

---

**Enjoy predicting next words! 🚀**
