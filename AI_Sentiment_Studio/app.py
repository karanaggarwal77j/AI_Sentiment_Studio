import os
from datetime import datetime, timezone

from flask import Flask, render_template, request, redirect, url_for
from transformers import pipeline

app = Flask(__name__)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
model = None

MAX_REVIEW_LENGTH = 2000
MAX_BULK_REVIEWS = 100


def get_model():
    """Load the sentiment model only when it is first needed."""
    global model
    if model is None:
        model = pipeline("sentiment-analysis", model=MODEL_NAME)
    return model


def clean_label(label):
    """Convert model labels into Positive, Neutral, or Negative."""
    label = str(label).lower()

    if "positive" in label or label == "label_2":
        return "Positive"
    if "negative" in label or label == "label_0":
        return "Negative"
    if "neutral" in label or label == "label_1":
        return "Neutral"

    return label.title()


def save_analysis(text, sentiment, score):
    """Save a result to MongoDB when MONGO_URI is configured.

    MongoDB is optional. If saving fails, analysis can still be displayed.
    """
    uri = os.getenv("MONGO_URI")
    if not uri:
        return

    client = None
    try:
        from pymongo import MongoClient

        client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        client["sentiment_analyzer"]["analyses"].insert_one({
            "text": text,
            "sentiment": sentiment,
            "score": score,
            "created_at": datetime.now(timezone.utc),
        })
    except Exception as exc:
        print("Optional MongoDB save skipped:", exc)
    finally:
        if client is not None:
            client.close()


def analyze(text):
    """Analyze one review and return its text, sentiment, and confidence score."""
    text = text.strip()
    if not text:
        raise ValueError("Review text cannot be empty.")

    # Avoid passing an excessively long string to the model.
    prediction = get_model()(text[:MAX_REVIEW_LENGTH])[0]
    sentiment = clean_label(prediction["label"])
    score = float(prediction["score"])

    result = {
        "text": text,
        "sentiment": sentiment,
        "score": score,
    }

    save_analysis(text, sentiment, score)
    return result


def model_error_message(exc):
    """Return a helpful message for model-loading or inference errors."""
    print("Sentiment model error:", repr(exc))
    return (
        "The AI model could not analyze the review. On the first run, "
        "check your internet connection so Transformers can download the model, "
        "then verify that your requirements are installed."
    )


def get_counts_and_percentages(results):
    labels = ["Positive", "Neutral", "Negative"]
    counts = {label: 0 for label in labels}

    for item in results:
        sentiment = item.get("sentiment")
        if sentiment in counts:
            counts[sentiment] += 1

    total = len(results)
    percentages = {
        label: round(count * 100 / total, 1) if total else 0
        for label, count in counts.items()
    }
    return counts, percentages, total


@app.route("/")
def home():
    """Send the home page to the single-review screen."""
    return redirect(url_for("single_review"))


@app.route("/single", methods=["GET", "POST"])
def single_review():
    single_result = None
    error = None
    review_text = ""

    if request.method == "POST":
        review_text = request.form.get("review", "").strip()

        if not review_text:
            error = "Please enter a review before analyzing."
        elif len(review_text) > MAX_REVIEW_LENGTH:
            error = f"Please keep the review under {MAX_REVIEW_LENGTH} characters."
        else:
            try:
                single_result = analyze(review_text)
            except Exception as exc:
                error = model_error_message(exc)

    return render_template(
        "single_review.html",
        single_result=single_result,
        review_text=review_text,
        error=error,
    )


@app.route("/bulk", methods=["GET", "POST"])
def bulk_reviews():
    bulk_results = []
    error = None
    reviews_text = ""

    if request.method == "POST":
        reviews_text = request.form.get("reviews", "")
        reviews = [
            line.strip()
            for line in reviews_text.splitlines()
            if line.strip()
        ]

        if not reviews:
            error = "Enter at least one review, with one review per line."
        elif len(reviews) > MAX_BULK_REVIEWS:
            error = f"Please analyze no more than {MAX_BULK_REVIEWS} reviews at a time."
        else:
            try:
                # Analyze each non-empty line separately so each result is listed.
                for review in reviews:
                    bulk_results.append(analyze(review))
            except Exception as exc:
                # Do not show a misleading partial chart if the batch failed.
                bulk_results = []
                error = model_error_message(exc)

    counts, percentages, total = get_counts_and_percentages(bulk_results)

    return render_template(
        "bulk_reviews.html",
        bulk_results=bulk_results,
        counts=counts,
        percentages=percentages,
        total=total,
        reviews_text=reviews_text,
        error=error,
    )


if __name__ == "__main__":
    # debug=True is useful during local development. Do not use it in production.
    app.run(debug=True)
