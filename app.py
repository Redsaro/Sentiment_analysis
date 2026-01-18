"""
🎯 Sentiment Analysis Interface
================================
A modern Gradio interface to showcase the DeBERTa-v3 sentiment analysis model.
Supports 3-class classification: Negative, Neutral, Positive
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os

# ============================================================================
# Configuration
# ============================================================================

# Path to the trained .pt model file
PT_MODEL_PATH = "./sentiment_model.pt"
HUGGINGFACE_MODEL = "microsoft/deberta-v3-base"
MAX_LENGTH = 256

# Default label mappings (will be overridden from .pt file if available)
LABEL_MAP = {'negative': 0, 'neutral': 1, 'positive': 2}
ID_TO_LABEL = {0: 'Negative 😠', 1: 'Neutral 😐', 2: 'Positive 😊'}
LABEL_COLORS = {0: '#ef4444', 1: '#6b7280', 2: '#22c55e'}  # red, gray, green

# ============================================================================
# Model Loading
# ============================================================================

def load_model():
    """Load the sentiment analysis model from .pt file and tokenizer."""
    global ID_TO_LABEL, LABEL_MAP
    
    # Check if the .pt file exists
    if not os.path.exists(PT_MODEL_PATH):
        raise FileNotFoundError(
            f"❌ Model file not found: {PT_MODEL_PATH}\n"
            f"Please ensure 'sentiment_model.pt' is in the current directory.\n"
            f"You can obtain this file by running the training notebook."
        )
    
    print(f"📥 Loading model from: {PT_MODEL_PATH}")
    
    # Load the checkpoint
    checkpoint = torch.load(PT_MODEL_PATH, map_location='cpu', weights_only=False)
    
    # Extract label mappings from checkpoint
    if 'label_map' in checkpoint:
        LABEL_MAP = checkpoint['label_map']
        print(f"✅ Loaded label_map: {LABEL_MAP}")
    
    if 'id_to_label' in checkpoint:
        # Convert keys to integers and add emojis
        id_to_label_raw = checkpoint['id_to_label']
        emoji_map = {0: '😠', 1: '😐', 2: '😊'}
        ID_TO_LABEL = {
            int(k): f"{v.capitalize()} {emoji_map.get(int(k), '')}" 
            for k, v in id_to_label_raw.items()
        }
        print(f"✅ Loaded id_to_label: {ID_TO_LABEL}")
    
    # Load tokenizer from HuggingFace (same tokenizer used for training)
    print(f"📥 Loading tokenizer from: {HUGGINGFACE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL)
    
    # Reconstruct model configuration
    if 'config' in checkpoint:
        from transformers import DebertaV2Config
        config = DebertaV2Config.from_dict(checkpoint['config'])
        model = AutoModelForSequenceClassification.from_config(config)
        print(f"✅ Model config loaded from checkpoint")
    else:
        # Fallback: create model with default config
        model = AutoModelForSequenceClassification.from_pretrained(
            HUGGINGFACE_MODEL,
            num_labels=3,
            id2label={0: 'negative', 1: 'neutral', 2: 'positive'},
            label2id=LABEL_MAP,
            ignore_mismatched_sizes=True
        )
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model weights loaded successfully!")
    
    return model, tokenizer


# Load model and tokenizer globally
print("🚀 Initializing Sentiment Analysis Model...")
model, tokenizer = load_model()
model.eval()

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✅ Model loaded on: {device}")


# ============================================================================
# Prediction Function
# ============================================================================

def predict_sentiment(text: str) -> tuple:
    """
    Analyze the sentiment of the input text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Tuple of (sentiment_label, confidence_dict)
    """
    if not text or not text.strip():
        return "Please enter some text to analyze.", None
    
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
    
    # Extract results
    probs = probs.cpu().numpy()[0]
    pred_id = probs.argmax()
    confidence = probs[pred_id]
    
    # Format sentiment result with emoji
    sentiment = ID_TO_LABEL[pred_id]
    result_text = f"**{sentiment}** ({confidence:.1%} confidence)"
    
    # Create data for bar chart
    chart_data = {
        "Negative 😠": float(probs[0]),
        "Neutral 😐": float(probs[1]),
        "Positive 😊": float(probs[2])
    }
    
    return result_text, chart_data


# ============================================================================
# Gradio Interface
# ============================================================================

# Custom CSS for modern dark theme
custom_css = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.main-header {
    text-align: center;
    padding: 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    margin-bottom: 1rem;
}

.main-header h1 {
    color: white;
    margin: 0;
    font-size: 2rem;
}

.main-header p {
    color: rgba(255, 255, 255, 0.9);
    margin: 0.5rem 0 0 0;
}

textarea {
    font-size: 1.1rem !important;
}

.prose h2 {
    color: #667eea;
}
"""

# Example texts for quick testing
examples = [
    ["I absolutely love this product! It exceeded all my expectations and I couldn't be happier with my purchase."],
    ["This is the worst customer service I have ever experienced. Extremely disappointed and frustrated."],
    ["The package arrived on time. The product matches the description."],
    ["The movie was okay, nothing special but not terrible either."],
    ["Outstanding work! The team delivered exceptional results beyond what we imagined possible."],
    ["I'm not sure how I feel about this. It has some good features but also some drawbacks."],
]

# Build the interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.HTML("""
        <div class="main-header">
            <h1>🎯 Sentiment Analysis</h1>
            <p>Powered by DeBERTa-v3 • Analyze the sentiment of any text</p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            # Input
            text_input = gr.Textbox(
                label="📝 Enter Text to Analyze",
                placeholder="Type or paste your text here...",
                lines=5,
                max_lines=10
            )
            
            # Submit button
            submit_btn = gr.Button("🔍 Analyze Sentiment", variant="primary", size="lg")
            
            # Examples
            gr.Examples(
                examples=examples,
                inputs=text_input,
                label="💡 Try These Examples"
            )
        
        with gr.Column(scale=1):
            # Output
            sentiment_output = gr.Markdown(
                label="Result",
                value="*Enter text and click Analyze*"
            )
            
            # Confidence chart
            confidence_chart = gr.Label(
                label="📊 Confidence Distribution",
                num_top_classes=3
            )
    
    # Footer
    gr.Markdown("""
    ---
    **Model Info**: DeBERTa-v3-base fine-tuned for 3-class sentiment classification
    
    **Labels**: 😠 Negative | 😐 Neutral | 😊 Positive
    """)
    
    # Connect button to function
    submit_btn.click(
        fn=predict_sentiment,
        inputs=text_input,
        outputs=[sentiment_output, confidence_chart]
    )
    
    # Also trigger on Enter key
    text_input.submit(
        fn=predict_sentiment,
        inputs=text_input,
        outputs=[sentiment_output, confidence_chart]
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎯 Starting Sentiment Analysis Interface")
    print("="*60)
    print("\nOpen your browser to: http://127.0.0.1:7860")
    print("Press Ctrl+C to stop the server\n")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # Set to True for public URL
        show_error=True
    )
