from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict
import subprocess

from src.hf_loader import load_filter_to_dataframe, save_dataframe_jsonl
from src.preprocess import preprocess_and_save_from_raw
from src.bertopic_pipeline import load_docs_from_jsonl, run_bertopic, save_outputs
from src.sentiment_rule_based import batch_process_subset
# defer llm_analysis import to runtime to avoid partial-import issues
# from src.llm_analysis import run_llm_on_subset
from src.synthesis import build_unified_table
from src.entity_clustering import cluster_entities
from src.visualizations import generate_all

ROOT = Path('/Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview')
DATA = ROOT / 'data'
RAW = DATA / '01_raw'
INTERIM = DATA / '02_interim'
PROC = DATA / '03_processed'
ANAL = DATA / '04_analysis'
REPORTS = ROOT / 'reports'
IMAGES = REPORTS / 'images'

for p in [RAW, INTERIM, PROC, ANAL, IMAGES, REPORTS]:
    p.mkdir(parents=True, exist_ok=True)


def _run_llm_safe(cleaned_jsonl: str, llm_out: str, model: str, max_rows: int, concurrency: int) -> None:
    try:
        # lazy import first
        from src.llm_analysis import run_llm_on_subset  # type: ignore
        run_llm_on_subset(cleaned_jsonl, llm_out, model=model, max_rows=max_rows, concurrency=concurrency)
        return
    except Exception:
        # fallback to CLI to avoid import-time issues
        cmd = [
            os.environ.get('PYTHON', 'python'), '-m', 'src.llm_analysis', cleaned_jsonl, llm_out,
            '--model', model, '--max_rows', str(max_rows), '--concurrency', str(concurrency)
        ]
        subprocess.run(cmd, check=True)


def run_pipeline(limit: int = 1000, language: str = 'English', llm_model: str = 'gpt-4.1-nano',
                 max_llm_rows: int = 1000, concurrency: int = 6, skip_llm: bool = False) -> Dict[str, str]:
    results: Dict[str, str] = {}

    # 1) Load + filter
    df = load_filter_to_dataframe(language=language, toxic=False, limit=limit)
    raw_path = RAW / f'wildchat_{language.lower()}_notoxic_{limit}.jsonl'
    save_dataframe_jsonl(df, str(raw_path))
    results['raw_jsonl'] = str(raw_path)

    # 2) Preprocess
    cleaned_jsonl = preprocess_and_save_from_raw(str(raw_path), output_dir=str(INTERIM))
    results['cleaned_jsonl'] = cleaned_jsonl

    # 3) BERTopic
    identities, docs = load_docs_from_jsonl(cleaned_jsonl)
    model, topics, probs = run_bertopic(docs)
    paths = save_outputs(model, docs, topics, identities, output_dir=str(REPORTS))
    results.update(paths)

    # 4) Rule-based sentiment
    rule_csv = ANAL / 'sentiment_rule_based_1000.csv'
    batch_process_subset(cleaned_jsonl, str(rule_csv))
    results['rule_based_csv'] = str(rule_csv)

    # 5) LLM analysis (use existing if present)
    llm_jsonl = PROC / 'llm_analysis_results_1000_v4.jsonl'
    if not llm_jsonl.exists() and not skip_llm:
        _run_llm_safe(cleaned_jsonl, str(llm_jsonl), model=llm_model, max_rows=max_llm_rows, concurrency=concurrency)
    results['llm_jsonl'] = str(llm_jsonl)

    # 6) Unified table (entity-level)
    unified_csv = ANAL / 'unified_table.csv'
    build_unified_table(
        cleaned_jsonl_path=cleaned_jsonl,
        doc_topics_csv_path=paths.get('doc_topics_csv', paths.get('doc_topics', str(REPORTS / 'bertopic_doc_topics.csv'))),
        topics_csv_path=paths.get('topics_csv', paths.get('topics', str(REPORTS / 'bertopic_topics.csv'))),
        rule_based_csv_path=str(rule_csv),
        llm_jsonl_path=str(llm_jsonl),
        output_csv_path=str(unified_csv),
    )
    results['unified_csv'] = str(unified_csv)

    # 7) Entity clustering
    cluster_entities(str(unified_csv), output_csv_path=str(unified_csv), min_cluster_size=4)

    # 8) Visualizations
    imgs = generate_all(str(unified_csv), out_dir=str(IMAGES))
    for k, v in imgs.items():
        results[f'img_{k}'] = v

    return results


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser(description='Run end-to-end analysis pipeline')
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--language', default='English')
    parser.add_argument('--llm_model', default='gpt-4.1-nano')
    parser.add_argument('--max_llm_rows', type=int, default=1000)
    parser.add_argument('--concurrency', type=int, default=6)
    parser.add_argument('--skip_llm', action='store_true')
    args = parser.parse_args()

    out = run_pipeline(limit=args.limit, language=args.language, llm_model=args.llm_model,
                       max_llm_rows=args.max_llm_rows, concurrency=args.concurrency, skip_llm=args.skip_llm)
    print(json.dumps(out, indent=2)) 