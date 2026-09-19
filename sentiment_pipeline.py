import os
import re
import torch
import spacy
import joblib
import html
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---- Load model & tokenizer ----
# Path to the folder containing config.json, model.safetensors, tokenizer.json, tokenizer_config.json
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "Transformer_model"))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_model = None
_tokenizer = None
_nlp = None

# Classical ML artifacts (TF-IDF + three trained classifiers).
CLASSICAL_DIR = os.environ.get("CLASSICAL_MODEL_DIR", os.path.join(os.path.dirname(__file__), "Classical_models"))
_tfidf = None
_classical_models = {}
_label_encoder = None


def load_resources():
    """Load model, tokenizer, and spaCy pipeline once. Call this at app startup."""
    global _model, _tokenizer, _nlp, _tfidf, _classical_models, _label_encoder
    if _model is None:
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        _model.to(device)
        _model.eval()
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")

    # Classical artifacts are optional at startup so the app can still run
    # with DistilBERT alone. When present, all three baseline models become
    # available for live side-by-side prediction.
    # These filenames match the user's actual saved training artifacts.
    if _tfidf is None:
        vectorizer_path = os.path.join(CLASSICAL_DIR, "tfidf_vectorizer.pkl")
        if os.path.exists(vectorizer_path):
            _tfidf = joblib.load(vectorizer_path)

    if _label_encoder is None:
        encoder_path = os.path.join(CLASSICAL_DIR, "label_encoder.pkl")
        if os.path.exists(encoder_path):
            _label_encoder = joblib.load(encoder_path)

    for name, filename in {
        "Logistic Regression": "log_reg.pkl",
        "Naive Bayes": "Mul_NB.pkl",
        "XGBoost": "xgb_balanced_model.pkl",
    }.items():
        if name not in _classical_models:
            path = os.path.join(CLASSICAL_DIR, filename)
            if os.path.exists(path):
                _classical_models[name] = joblib.load(path)
    return _model, _tokenizer, _nlp


id_to_label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# ---- Aspect keywords ----
aspect_keywords = {
    "effectiveness": ["work", "worked", "works", "effective", "effectiveness", "helped", "help", "relief",
                       "improve", "improved", "improvement", "cured", "reduce", "reduced"],
    "side_effects": ["side effect", "side effects", "nausea", "dizziness", "dizzy", "headache", "vomiting",
                      "drowsy", "drowsiness", "rash", "weight gain", "weight loss", "insomnia", "fatigue",
                      "reaction"],
    "dosage": ["dose", "dosage", "mg", "pill", "pills", "tablet", "tablets", "schedule", "taking", "took",
               "prescribed"],
    "cost": ["price", "expensive", "cheap", "cost", "insurance", "afford", "copay"]
}


# ---- Helper functions ----
def get_sentences(text):
    doc = _nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def detect_aspects(sentence):
    sentence_lower = sentence.lower()
    return [aspect for aspect, kws in aspect_keywords.items() if any(kw in sentence_lower for kw in kws)]


def split_on_contrast(sentence):
    parts = re.split(r'\b(but|however|although|though|yet)\b', sentence, flags=re.IGNORECASE)
    clauses = []
    current = parts[0]
    for i in range(1, len(parts), 2):
        clauses.append(current.strip())
        current = parts[i + 1] if i + 1 < len(parts) else ''
    if current.strip():
        clauses.append(current.strip())
    return [c for c in clauses if c]


def predict_sentiment(text):
    inputs = _tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=256)
    # DistilBERT does not accept token_type_ids (unlike full BERT) — drop it if the tokenizer added it
    inputs = {k: v.to(device) for k, v in inputs.items() if k != 'token_type_ids'}
    with torch.no_grad():
        outputs = _model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred_id = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred_id].item()
    return id_to_label[pred_id], confidence



def clean_for_classical(text):
    """Match the heavy-cleaning pipeline used by the TF-IDF baselines."""
    text = html.unescape(str(text))
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.strip('"').strip("'")
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z'\s]", " ", text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    try:
        doc = _nlp(text)
        tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and t.lemma_.strip()]
        return " ".join(tokens)
    except Exception:
        return text


def predict_classical_models(text):
    """Run every available TF-IDF baseline on the same review."""
    if _tfidf is None or not _classical_models:
        return []
    features = _tfidf.transform([clean_for_classical(text)])
    results = []
    for name in ["Logistic Regression", "Naive Bayes", "XGBoost"]:
        model = _classical_models.get(name)
        if model is None:
            continue
        pred = model.predict(features)[0]
        if hasattr(pred, "item"):
            pred = pred.item()

        # Prefer the exact LabelEncoder saved with the models. Fall back to the
        # documented 0/1/2 mapping if the encoder is unavailable.
        if _label_encoder is not None:
            try:
                label = str(_label_encoder.inverse_transform([pred])[0])
            except Exception:
                label = id_to_label.get(int(pred), str(pred)) if isinstance(pred, (int, float)) else str(pred)
        else:
            label = id_to_label.get(int(pred), str(pred)) if isinstance(pred, (int, float)) else str(pred)
        confidence = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            confidence = float(max(probs))
        results.append({"model": name, "sentiment": label, "confidence": round(confidence, 3) if confidence is not None else None})
    return results

# ---- Main entry point ----
def analyze_review(text: str, include_model_comparison: bool = True) -> dict:
    if _model is None or _tokenizer is None or _nlp is None:
        load_resources()

    overall_sentiment, overall_confidence = predict_sentiment(text)
    model_comparison = []
    if include_model_comparison:
        model_comparison = predict_classical_models(text)
        model_comparison.append({'model': 'DistilBERT', 'sentiment': overall_sentiment, 'confidence': round(overall_confidence, 3), 'production': True})
    sentences = get_sentences(text)
    aspect_sentiments = {}

    for sentence in sentences:
        for clause in split_on_contrast(sentence):
            aspects_in_clause = detect_aspects(clause)
            if aspects_in_clause:
                clause_label, clause_conf = predict_sentiment(clause)
                for aspect in aspects_in_clause:
                    if aspect not in aspect_sentiments or clause_conf > aspect_sentiments[aspect]['confidence']:
                        aspect_sentiments[aspect] = {'sentiment': clause_label, 'confidence': round(clause_conf, 3)}

    return {
        'overall_sentiment': overall_sentiment,
        'overall_confidence': round(overall_confidence, 3),
        'aspects': [{'aspect': k, **v} for k, v in aspect_sentiments.items()],
        'model_comparison': model_comparison
    }


if __name__ == "__main__":
    load_resources()
    sample = ("This medication worked really well for my anxiety, but I experienced bad nausea "
               "and dizziness the first week.")
    import json
    print(json.dumps(analyze_review(sample), indent=2))
