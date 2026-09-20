# 💊 DrugSentinel

> An NLP-based drug review sentiment and aspect analysis platform using Classical Machine Learning, DistilBERT, and Aspect-Based Sentiment Analysis (ABSA).

DrugSentinel analyzes medication reviews and converts unstructured text into structured sentiment insights. The system identifies **overall sentiment, aspect-level sentiment, confidence scores, and model predictions** through an interactive web application and FastAPI backend.

### 🚀 Live Demo

[**Open DrugSentinel**](https://drugsentinel-573539539303.us-central1.run.app)

---

## 📌 Overview

Online medication reviews contain valuable information about users' experiences, but manually analyzing large collections of reviews can be time-consuming.

DrugSentinel applies **Natural Language Processing and Machine Learning** techniques to automatically analyze drug reviews and identify:

* Overall sentiment
* Prediction confidence
* Relevant review aspects
* Aspect-level sentiment
* Aspect-level confidence
* Predictions from multiple Machine Learning models
* Aggregate sentiment statistics for bulk reviews

The project combines **Classical Machine Learning models with a Transformer-based DistilBERT model** to provide both baseline predictions and contextual text classification.

---

## ✨ Features

### 🔹 Single Review Analysis

Analyze an individual medication review and obtain:

* Overall sentiment
* Confidence score
* Detected aspects
* Aspect-level sentiment
* Aspect-level confidence
* Predictions from multiple models

### 🔹 Bulk Review Analysis

Analyze multiple reviews in a single request and generate:

* Overall sentiment distribution
* Aspect-level sentiment distribution
* Individual review predictions
* Confidence scores
* Batch-level statistics

### 🔹 Aspect-Based Sentiment Analysis

DrugSentinel goes beyond classifying an entire review as positive, neutral, or negative.

It identifies sentiment associated with specific aspects of a review:

| Aspect           | Example                                   |
| ---------------- | ----------------------------------------- |
| 💊 Effectiveness | "The medication worked very well."        |
| ⚠️ Side Effects  | "It caused severe nausea."                |
| 💰 Cost          | "It is too expensive."                    |
| 📋 Dosage        | "The dosing schedule was easy to follow." |

For example:

> "The medication worked really well but caused severe dryness."

The system can identify:

```text
Effectiveness → Positive
Side Effects  → Negative
```

This provides more detailed information than simple document-level sentiment classification.

---

# 🧠 Machine Learning Approach

DrugSentinel uses both **Classical Machine Learning** and a **Transformer-based model**.

## Classical Machine Learning

The project includes multiple traditional classification models for baseline comparison:

* Logistic Regression
* Naive Bayes
* XGBoost

These models provide comparative predictions alongside the production sentiment model.

## Transformer-Based Classification

**DistilBERT** is used as the production sentiment classification model.

DistilBERT provides contextual text representations, allowing sentiment classification to consider the surrounding context of words within a review.

### Model Comparison

For individual reviews, DrugSentinel can compare predictions from multiple models.

Example:

| Model               | Prediction   | Confidence | Production |
| ------------------- | ------------ | ---------: | ---------- |
| Logistic Regression | Positive     |      51.7% | No         |
| Naive Bayes         | Positive     |      37.5% | No         |
| XGBoost             | Neutral      |      34.8% | No         |
| **DistilBERT**      | **Negative** |  **93.8%** | **Yes**    |

This makes it possible to examine how different approaches classify the same piece of text.

---

# 🔍 Aspect Analysis

The current aspect categories include:

```text
Effectiveness
Side Effects
Cost
Dosage
```

Each detected aspect can have its own sentiment and confidence.

Example:

```json
{
  "aspect": "cost",
  "sentiment": "Negative",
  "confidence": 0.901
}
```

A single review can therefore contain multiple aspect-level opinions.

---

# 📊 Example

### Input

```text
This drug worked wonders for my anxiety within just two weeks.
```

### Output

```json
{
  "overall_sentiment": "Positive",
  "overall_confidence": 0.991,
  "aspects": [
    {
      "aspect": "effectiveness",
      "sentiment": "Positive",
      "confidence": 0.991
    }
  ]
}
```

---

# 📦 Bulk Analysis

DrugSentinel can process multiple reviews together.

Example batch result:

```text
Total Reviews: 8

Positive: 4
Neutral: 1
Negative: 3
```

### Aspect-Level Distribution

```text
Effectiveness
├── Positive: 3
├── Neutral: 1
└── Negative: 2

Side Effects
└── Negative: 2

Cost
├── Positive: 1
└── Negative: 1

Dosage
├── Positive: 2
└── Negative: 1
```

This allows larger collections of reviews to be summarized without inspecting every review individually.

---

# 📚 Dataset

DrugSentinel uses the **Drug Review Dataset** associated with the UCI Machine Learning Repository.

The dataset was originally published through the **UCI Machine Learning Repository** and was used for the 2018 Kaggle University Club Hackathon. The Kaggle dataset page identifies the original UCI source and provides the publicly accessible version used by this project.

The original UCI-hosted version is not currently the accessible source used for this project, so the dataset was obtained from the following **Kaggle-hosted copy**:

### 🔗 Dataset

**[UCI ML Drug Review Dataset — Kaggle](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018)**

The Kaggle dataset contains **200,000+ patient drug reviews** and includes information such as drug names, medical conditions, review text, ratings, and other associated fields.

The dataset was originally associated with research on **aspect-based sentiment analysis of drug reviews**, including analysis of aspects such as effectiveness and side effects.

### Dataset Usage

The review data is used in DrugSentinel for:

* Text preprocessing
* Sentiment classification
* Classical Machine Learning
* Transformer-based classification
* Model evaluation
* Aspect-oriented analysis

> **Dataset attribution:** The dataset is not owned by this project. It is used as an external dataset for educational and research purposes. Please refer to the [original Kaggle dataset page](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018) for its attribution, acknowledgments, and usage conditions.

---

# 🏗️ System Architecture

```text
                       ┌───────────────────────┐
                       │     Web Interface     │
                       └───────────┬───────────┘
                                   │
                                   ▼
                       ┌───────────────────────┐
                       │     FastAPI Backend   │
                       └───────────┬───────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
          ┌──────────────────┐          ┌──────────────────┐
          │  Classical ML    │          │    DistilBERT    │
          │                  │          │                  │
          │ Logistic Reg.    │          │ Transformer      │
          │ Naive Bayes      │          │ Classification   │
          │ XGBoost          │          │                  │
          └────────┬─────────┘          └────────┬─────────┘
                   │                             │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Sentiment Analysis  │
                       │         +           │
                       │  Aspect Analysis    │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Results / History / │
                       │      Statistics     │
                       └─────────────────────┘
```

---

# 🔄 Analysis Workflow

```text
                Drug Review
                     │
                     ▼
             Text Processing
                     │
                     ▼
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
   Classical Models        DistilBERT
          │                     │
          └──────────┬──────────┘
                     │
                     ▼
          Sentiment Classification
                     │
                     ▼
             Aspect Analysis
                     │
                     ▼
          Confidence Calculation
                     │
                     ▼
             Structured Output
                     │
                     ▼
              Web Interface
```

---

# 🔌 API Endpoints

DrugSentinel exposes its functionality through a FastAPI REST API.

| Method | Endpoint            | Description                       |
| ------ | ------------------- | --------------------------------- |
| `GET`  | `/`                 | Web application                   |
| `GET`  | `/api/health`       | API health check                  |
| `POST` | `/api/analyze`      | Analyze a single review           |
| `POST` | `/api/analyze-bulk` | Analyze multiple reviews          |
| `GET`  | `/api/history`      | Retrieve analysis history         |
| `GET`  | `/api/stats`        | Retrieve analysis statistics      |
| `GET`  | `/docs`             | Interactive Swagger documentation |
| `GET`  | `/openapi.json`     | OpenAPI specification             |

---

# 🛠️ Tech Stack

## Programming

* Python

## Machine Learning

* Scikit-learn
* XGBoost
* Joblib

## Deep Learning & Transformers

* PyTorch
* Hugging Face Transformers
* DistilBERT

## Natural Language Processing

* Natural Language Processing
* Text Classification
* Aspect-Based Sentiment Analysis
* spaCy

## Backend

* FastAPI
* Uvicorn
* Pydantic

## Frontend

* HTML
* CSS
* JavaScript

## Development & Containerization

* Docker
* Git
* GitHub

---

# 📁 Project Structure

```text
DrugSentinel/
│
├── main.py
├── sentiment_pipeline.py
├── database.py
├── requirements.txt
├── Dockerfile
├── README.md
│
├── Transformer_model/
│   └── ...
│
├── Classical_models/
│   └── ...
│
├── static/
│   └── ...
│
└── ...
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/ashit0907/DrugSentinel.git
cd DrugSentinel
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Download the spaCy Model

```bash
python -m spacy download en_core_web_sm
```

## 5. Start the Application

```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```

The application will be available at:

```text
http://127.0.0.1:8080
```

---

# 📚 API Documentation

After starting the application, open:

```text
http://127.0.0.1:8080/docs
```

FastAPI provides an interactive Swagger interface where the available endpoints can be explored and tested.

---

# 🧪 Example Use Cases

DrugSentinel can be used for exploratory analysis of collections of medication reviews to identify:

* General sentiment trends
* Frequently discussed aspects
* Sentiment toward effectiveness
* Sentiment associated with side effects
* Cost-related sentiment
* Dosage-related sentiment
* Differences between classification approaches

The system is designed primarily as an **NLP and Machine Learning project for text analysis and experimentation**.

---

# 🎯 Project Objectives

The main objectives of DrugSentinel are:

1. Build an end-to-end NLP sentiment classification system.
2. Compare Classical Machine Learning models with a Transformer-based model.
3. Implement aspect-level sentiment analysis for medication reviews.
4. Support both single-review and bulk-review analysis.
5. Expose the models through a REST API.
6. Provide an interactive web interface.
7. Maintain analysis history and aggregated statistics.
8. Demonstrate a complete workflow from text input to structured ML output.

---

# 🔬 Why DistilBERT?

Traditional text classification approaches can provide strong baselines, but they may have limited contextual understanding.

DistilBERT uses contextual representations to capture relationships between words within a sentence.

For example:

```text
"The medication worked well, but the side effects were terrible."
```

A single overall sentiment label can hide the different opinions expressed in the sentence.

Aspect-level analysis provides a more detailed representation:

```text
Effectiveness → Positive
Side Effects  → Negative
```

This combination of **Transformer-based classification and aspect-level analysis** is one of the central components of DrugSentinel.

---

# 📈 Future Improvements

Potential future improvements include:

* More advanced aspect extraction
* Additional Transformer architectures
* Improved aspect classification
* Explainability and model interpretation
* Interactive analytics dashboards
* More comprehensive model evaluation
* Larger and more diverse review datasets
* Multilingual review analysis
* Model versioning and monitoring
* Authentication and user management
* Production deployment optimization

---

# ⚠️ Disclaimer

**DrugSentinel is an educational and research-oriented NLP project.**

The predictions generated by the system represent analysis of textual reviews and **must not be interpreted as medical advice, clinical recommendations, or evidence regarding the safety or effectiveness of any medication.**

Always consult a qualified healthcare professional for medical decisions.

---

# 👨‍💻 Author

**Ashit**

B.Tech Computer Science & Engineering — AI & ML

### Areas of Interest

* Machine Learning
* Natural Language Processing
* Deep Learning
* Transformers
* Data Science
* MLOps
* Python Development

---

## ⭐ Support

If you find the project interesting, feel free to explore the repository, experiment with the models, and build upon the implementation.

If you find DrugSentinel useful, consider giving the repository a ⭐.
