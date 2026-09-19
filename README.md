---
title: DrugSentinel
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# DrugSentinel

Aspect-based sentiment analysis for patient drug reviews. Classifies overall sentiment
(Positive/Neutral/Negative) using a fine-tuned DistilBERT model, and breaks sentiment
down by aspect — effectiveness, side effects, dosage, and cost — using clause-level
sentiment detection. Includes a live comparison against three classical ML baselines
(Logistic Regression, Naive Bayes, XGBoost) trained on the same data.

B.Tech CSE (AIML) Minor Project — Group 12.
