import os
import json
import time
import requests
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "openai/gpt-oss-120b:free"
INPUT_CSV = "reviews.csv"
OUTPUT_CSV = "reviews_classified.csv"
URL = "https://openrouter.ai/api/v1/chat/completions"

if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not found")

JSON_SCHEMA = {
    "name": "review_classification",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral", "mixed", "unknown"],
                "description": "Overall sentiment of the review."
            },
            "topic": {
                "type": "string",
                "enum": ["delivery", "product_quality", "fit", "seller", "price", "other", "unknown"],
                "description": "Main topic of the review."
            }
        },
        "required": ["sentiment", "topic"],
        "additionalProperties": False
    }
}

def parse_result(content: str) -> dict:
    if not content:
        return {"sentiment": "unknown", "topic": "unknown"}

    content = content.strip()
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {"sentiment": "unknown", "topic": "unknown"}

    try:
        obj = json.loads(content[start:end + 1])
        return {
            "sentiment": obj.get("sentiment", "unknown"),
            "topic": obj.get("topic", "unknown"),
        }
    except Exception:
        return {"sentiment": "unknown", "topic": "unknown"}

def classify_review(review_text: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You classify customer reviews. "
                    "Choose exactly one sentiment from: positive, negative, neutral, mixed. "
                    "Use unknown only when the text is too broken or the result cannot be determined. "
                    "Choose exactly one topic from: delivery, product_quality, fit, seller, price, other. "
                    "Use unknown only on errors or unclassifiable text. "
                    "Return only valid JSON that matches the schema."
                )
            },
            {
                "role": "user",
                "content": f"Review: {review_text}"
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": JSON_SCHEMA
        }
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    r = requests.post(URL, headers=headers, json=payload, timeout=90)
    r.raise_for_status()

    data = r.json()
    choices = data.get("choices", [])
    if not choices:
        return {"sentiment": "unknown", "topic": "unknown"}

    content = choices[0].get("message", {}).get("content", "")
    result = parse_result(content)

    return {
        "sentiment": result.get("sentiment", "unknown"),
        "topic": result.get("topic", "unknown"),
    }

def main():
    df = pd.read_csv(INPUT_CSV, sep=";", engine="python", on_bad_lines="warn")

    if "review" not in df.columns:
        raise RuntimeError("'review' column not found in input file")

    sentiments = []
    topics = []

    for i, row in df.iterrows():
        review = str(row.get("review", "")).strip()
        try:
            result = classify_review(review)
            sentiment = result.get("sentiment", "unknown")
            topic = result.get("topic", "unknown")
        except Exception as e:
            print(f"Row {i} failed: {e}")
            sentiment = "unknown"
            topic = "unknown"

        sentiments.append(sentiment)
        topics.append(topic)
        time.sleep(0.3)

    if len(sentiments) != len(df):
        raise RuntimeError(f"Length mismatch: df={len(df)}, sentiments={len(sentiments)}, topics={len(topics)}")

    df["sentiment"] = sentiments
    df["topic"] = topics
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig", sep=";")
    print(f"Saved: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()