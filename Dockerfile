# DrugSentinel — Docker image for Hugging Face Spaces
FROM python:3.11-slim

# System dependencies needed by torch/spacy build chains
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application
# This includes: main.py, sentiment_pipeline.py, database.py, static/,
# Transformer_model/, Classical_models/ (with your real .pkl files added)
COPY . .

# Hugging Face Spaces requires the app to listen on port 7860
ENV PORT=7860
EXPOSE 7860

# Create a non-root user (recommended by HF Spaces)
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
