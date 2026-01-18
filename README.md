# 🎯 Sentiment Analysis with DeBERTa-v3

A modern Gradio interface for sentiment analysis using a fine-tuned DeBERTa-v3-base model.

![Sentiment Analysis](https://img.shields.io/badge/Model-DeBERTa--v3--base-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Gradio](https://img.shields.io/badge/Interface-Gradio-orange)

## 📋 Overview

This project provides:
- **Training Notebook** (`train_sentiment_deberta.ipynb`) - Train a 3-class sentiment classifier on Google Colab
- **Gradio Web Interface** (`app.py`) - Interactive demo to test the trained model

### Sentiment Classes
| Label | Emoji | Description |
|-------|-------|-------------|
| Negative | 😠 | Negative sentiment |
| Neutral | 😐 | Neutral/mixed sentiment |
| Positive | 😊 | Positive sentiment |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Trained model file (`sentiment_model.pt`)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the App

1. **Place your trained model** (`sentiment_model.pt`) in the project root directory

2. **Start the Gradio server**:
   ```bash
   python app.py
   ```

3. **Open your browser** to [http://127.0.0.1:7860](http://127.0.0.1:7860)

---

## 🏋️ Training the Model

Use the provided Jupyter notebook to train your own model:

1. Open `train_sentiment_deberta.ipynb` in Google Colab
2. Mount your Google Drive
3. Update `DATASET_PATH` to point to your CSV file
4. Run all cells to train the model
5. Download `sentiment_model.pt` from Google Drive

### Dataset Format
Your CSV file should have these columns:
| Column | Description |
|--------|-------------|
| `text` | The text to classify |
| `sentiment` | Label: `negative`, `neutral`, or `positive` |

---

## 📁 Project Structure

```
sentiment analysis/
├── app.py                          # Gradio web interface
├── requirements.txt                # Python dependencies
├── train_sentiment_deberta.ipynb   # Training notebook (Colab)
├── sentiment_model.pt              # Trained model (after training)
└── archive/                        # Dataset files
    ├── train.csv
    └── test.csv
```

---

## 🔧 Model Export Formats

The training notebook exports the model in multiple formats:

| Format | File | Use Case |
|--------|------|----------|
| PyTorch | `sentiment_model.pt` | **Used by this app** |
| ONNX | `sentiment_model.onnx` | Fast inference, edge deployment |
| HuggingFace | `model.safetensors` | HuggingFace ecosystem |

---

## ⚙️ Configuration

Edit these variables in `app.py` to customize:

```python
PT_MODEL_PATH = "./sentiment_model.pt"  # Path to your trained model
MAX_LENGTH = 256                         # Max input text length
```

---

## 📊 Features

- ✅ **Real-time sentiment analysis**
- ✅ **Confidence score visualization**
- ✅ **Pre-built example texts**
- ✅ **Modern, responsive UI**
- ✅ **GPU acceleration** (when available)

---

## 📝 License

MIT License - Feel free to use and modify for your projects.
