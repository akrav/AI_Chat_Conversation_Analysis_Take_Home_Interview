from __future__ import annotations

import json
import os
import asyncio
import random
from typing import Iterable, List, Dict, Optional

import pandas as pd
from dotenv import load_dotenv

# Load .env
load_dotenv()

try:
    from openai import OpenAI, AsyncOpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
    AsyncOpenAI = None  # type: ignore


def _get_openai_client():
    if OpenAI is None:
        raise RuntimeError("openai package not available")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    return OpenAI(api_key=api_key)


def _get_async_client():
    if AsyncOpenAI is None:
        raise RuntimeError("openai package not available")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    return AsyncOpenAI(api_key=api_key)


def _parse_json_content(content: str) -> dict:
    try:
        return json.loads(content)
    except Exception:
        pass
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = content[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass
    return {"raw": content}


def _build_unified_prompt(text: str) -> str:
    return (
        "Analyze the following conversation text and return a SINGLE valid JSON object only, with this exact schema and rules.\n"
        "SCHEMA (all fields REQUIRED, no nulls):\n"
        "{\n"
        "  \"intent\": { \"intent\": string(one of: Technical Support, Creative Writing, Product Inquiry, General Chat, Jailbreaking, Other), \"confidence\": number in [0,1] },\n"
        "  \"sentiment\": { \"sentiment\": string(one of: positive, neutral, negative), \"confidence\": number in [0,1] },\n"
        "  \"entities\": { \"entities\": [ { \"text\": string, \"category\": string(one of: brand, store, product, person, software, organization, location, possible_brand, possible_store, possible_product, other), \"sentiment\": { \"sentiment\": string(one of: positive, neutral, negative), \"confidence\": number in [0,1] } } ] }\n"
        "}\n"
        "RULES:\n"
        "- Output ONLY JSON (no prose, no code fences).\n"
        "- Do NOT use null anywhere. If uncertain, choose: category=\"other\", entity.sentiment.sentiment=\"neutral\", and confidence=0.5.\n"
        "- entities.entities MUST contain at least one object. If no explicit entity is present, create one that captures the main subject.\n"
        "- All confidences MUST be numeric in [0,1].\n\n"
        f"Text: {text}"
    )


def _ensure_llm_object_shape(obj: dict) -> dict:
    intent = obj.get("intent") or {}
    sentiment = obj.get("sentiment") or {}
    entities = obj.get("entities") or {}
    if not isinstance(intent, dict):
        intent = {}
    if not isinstance(sentiment, dict):
        sentiment = {}
    if not isinstance(entities, dict):
        entities = {}
    raw_list = entities.get("entities") if isinstance(entities.get("entities"), list) else []
    norm_entities: List[Dict[str, Optional[str]]] = []
    for item in raw_list:
        if isinstance(item, dict):
            text = item.get("text")
            cat = item.get("category")
            es = item.get("sentiment") or {}
            if not isinstance(es, dict):
                es = {}
            e_sent = es.get("sentiment")
            e_conf = es.get("confidence")
            norm_entities.append({"text": text, "category": cat, "entity_sentiment": e_sent, "entity_sent_confidence": e_conf})
        else:
            norm_entities.append({"text": str(item), "category": None, "entity_sentiment": None, "entity_sent_confidence": None})
    entities = {"entities": norm_entities}
    if "intent" not in intent:
        intent["intent"] = None
    if "confidence" not in intent:
        intent["confidence"] = None
    if "sentiment" not in sentiment:
        sentiment["sentiment"] = None
    if "confidence" not in sentiment:
        sentiment["confidence"] = None
    return {"intent": intent, "sentiment": sentiment, "entities": entities}


def analyze_rows_with_llm(
    rows: Iterable[dict],
    tasks: List[str],  # kept for compatibility, ignored
    model: str = "gpt-4.1-nano",
    max_rows: Optional[int] = None,
) -> List[dict]:
    client = _get_openai_client()
    results: List[dict] = []
    count = 0
    for row in rows:
        if max_rows is not None and count >= max_rows:
            break
        conversation = row.get("conversation", [])
        text_parts = [t.get("content", "") for t in conversation if t.get("content")]
        text = "\n".join(text_parts)[:8000]
        prompt = _build_unified_prompt(text)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = resp.choices[0].message.content or "{}"
        parsed = _ensure_llm_object_shape(_parse_json_content(content))
        enriched = dict(row)
        enriched["llm_analysis"] = parsed
        results.append(enriched)
        count += 1
    return results


async def _analyze_single_async(client, row: dict, model: str, retries: int = 5) -> dict:
    conversation = row.get("conversation", [])
    text_parts = [t.get("content", "") for t in conversation if t.get("content")]
    text = "\n".join(text_parts)[:8000]
    prompt = _build_unified_prompt(text)
    attempt = 0
    while True:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = resp.choices[0].message.content or "{}"
            parsed = _ensure_llm_object_shape(_parse_json_content(content))
            enriched = dict(row)
            enriched["llm_analysis"] = parsed
            return enriched
        except Exception:
            attempt += 1
            if attempt > retries:
                enriched = dict(row)
                enriched["llm_analysis"] = {
                    "intent": {"intent": None, "confidence": None},
                    "sentiment": {"sentiment": None, "confidence": None},
                    "entities": {"entities": []},
                    "error": "request_failed",
                }
                return enriched
            await asyncio.sleep(min(2 ** attempt + random.random(), 10))


async def _analyze_rows_with_llm_async(
    rows: List[dict],
    model: str,
    concurrency: int = 6,
) -> List[dict]:
    client = _get_async_client()
    sem = asyncio.Semaphore(concurrency)

    async def bound_analyze(r):
        async with sem:
            return await _analyze_single_async(client, r, model)

    coros = [bound_analyze(r) for r in rows]
    results = await asyncio.gather(*coros)
    return list(results)


def run_llm_on_subset(
    subset_jsonl_path: str,
    output_path: str,
    tasks: Optional[List[str]] = None,  # kept for compatibility, ignored
    model: str = "gpt-4.1-nano",
    max_rows: Optional[int] = 1000,
    concurrency: int = 6,
) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(subset_jsonl_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if max_rows is not None:
        rows = rows[: max_rows]

    enriched: List[dict] = asyncio.run(_analyze_rows_with_llm_async(rows, model=model, concurrency=concurrency))

    with open(output_path, "w", encoding="utf-8") as f:
        for r in enriched:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return output_path


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM analysis on a subset JSONL")
    parser.add_argument("subset", help="Path to subset JSONL")
    parser.add_argument("output", help="Path to output JSONL with LLM analysis")
    parser.add_argument("--model", default="gpt-4.1-nano")
    parser.add_argument("--max_rows", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()

    out = run_llm_on_subset(args.subset, args.output, model=args.model, max_rows=args.max_rows, concurrency=args.concurrency)
    print(out)
