import io
import uuid
import csv
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import sentiment_pipeline as sp
import database as db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the model once and initialize the database
    print("Loading model, tokenizer, and spaCy pipeline...")
    sp.load_resources()
    db.init_db()
    print("Ready.")
    yield
    # Shutdown: nothing to clean up


app = FastAPI(title="DrugSentinel API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewRequest(BaseModel):
    review: str
    drug_name: str | None = None


class AspectResult(BaseModel):
    aspect: str
    sentiment: str
    confidence: float


class ModelPrediction(BaseModel):
    model: str
    sentiment: str
    confidence: float | None = None
    production: bool = False


class AnalysisResult(BaseModel):
    overall_sentiment: str
    overall_confidence: float
    aspects: list[AspectResult]
    model_comparison: list[ModelPrediction]


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze", response_model=AnalysisResult)
def analyze_single(payload: ReviewRequest):
    review = payload.review.strip()
    if not review:
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")
    if len(review) > 5000:
        raise HTTPException(status_code=400, detail="Review text is too long (max 5000 characters).")

    result = sp.analyze_review(review)
    db.save_analysis(review, result, drug_name=payload.drug_name, source="single")
    return result


@app.post("/api/analyze-bulk")
async def analyze_bulk(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or "review" not in [f.lower() for f in reader.fieldnames]:
        raise HTTPException(
            status_code=400,
            detail="CSV must contain a 'review' column."
        )

    # find the actual column name regardless of case
    review_col = next(f for f in reader.fieldnames if f.lower() == "review")
    drug_col = next((f for f in reader.fieldnames if f.lower() in ("drugname", "drug_name", "drug")), None)

    rows = list(reader)
    if len(rows) == 0:
        raise HTTPException(status_code=400, detail="CSV file has no data rows.")
    if len(rows) > 500:
        raise HTTPException(status_code=400, detail="Bulk upload is limited to 500 rows per file.")

    batch_id = str(uuid.uuid4())
    results = []

    for row in rows:
        review_text = (row.get(review_col) or "").strip()
        if not review_text:
            continue
        drug_name = row.get(drug_col) if drug_col else None
        # Bulk analysis uses the production DistilBERT pipeline only.
        # The classical models remain available for single-review live comparison,
        # but skipping them here avoids 3 extra model inferences per uploaded row.
        result = sp.analyze_review(review_text, include_model_comparison=False)
        db.save_analysis(review_text, result, drug_name=drug_name, source="bulk", batch_id=batch_id)
        results.append({"review": review_text, "drug_name": drug_name, **result})

    summary = _summarize_batch(results)

    return {
        "batch_id": batch_id,
        "row_count": len(results),
        "summary": summary,
        "results": results
    }


def _summarize_batch(results: list[dict]) -> dict:
    sentiment_dist = {"Positive": 0, "Neutral": 0, "Negative": 0}
    aspect_dist: dict[str, dict[str, int]] = {}

    for r in results:
        sentiment_dist[r["overall_sentiment"]] += 1
        for a in r["aspects"]:
            aspect_dist.setdefault(a["aspect"], {"Positive": 0, "Neutral": 0, "Negative": 0})
            aspect_dist[a["aspect"]][a["sentiment"]] += 1

    return {
        "overall_sentiment_distribution": sentiment_dist,
        "aspect_sentiment_distribution": aspect_dist
    }


@app.get("/api/history")
def history(limit: int = 50):
    return db.get_recent_analyses(limit=limit)


@app.get("/api/stats")
def stats():
    return db.get_summary_stats()


# Serve the frontend dashboard as static files, mounted last so /api/* routes take priority
app.mount("/", StaticFiles(directory="static", html=True), name="static")
