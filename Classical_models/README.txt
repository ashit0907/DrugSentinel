DrugSentinel V6 - classical model artifacts

Place these exact saved artifacts in this folder:

1. tfidf_vectorizer.pkl
2. log_reg.pkl
3. Mul_NB.pkl
4. xgb_balanced_model.pkl
5. label_encoder.pkl

These are the filenames used by sentiment_pipeline.py. No renaming is required.

The application sends the same review through all three TF-IDF classical models and
the fine-tuned DistilBERT model. DistilBERT remains the production model because
it was selected from the held-out test-set evaluation. The live comparison is a
prediction comparison for the same unseen review, not a replacement for test-set
model evaluation.

The fitted TF-IDF vectorizer must be the exact object used to train the three
classical models. Do not refit a new vectorizer on arbitrary data.
