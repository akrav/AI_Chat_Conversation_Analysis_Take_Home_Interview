from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional
import subprocess
import logging
import time

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

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def _parse_max_arg(v: Optional[str]) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"all", "max", "*", "none", "-1"}:
        return None
    try:
        n = int(s)
        return None if n < 0 else n
    except Exception:
        return None


def _run_llm_safe(cleaned_jsonl: str, llm_out: str, model: str, max_rows: Optional[int], concurrency: int) -> None:
    try:
        # lazy import first
        from src.llm_analysis import run_llm_on_subset  # type: ignore
        run_llm_on_subset(cleaned_jsonl, llm_out, model=model, max_rows=max_rows, concurrency=concurrency)
        return
    except Exception:
        # fallback to CLI to avoid import-time issues
        cmd = [
            os.environ.get('PYTHON', 'python'), '-m', 'src.llm_analysis', cleaned_jsonl, llm_out,
            '--model', model, '--max_rows', str(max_rows if max_rows is not None else 'all'), '--concurrency', str(concurrency)
        ]
        subprocess.run(cmd, check=True)


def _count_lines(path: str) -> int:
    n = 0
    with open(path, 'r', encoding='utf-8') as f:
        for _ in f:
            n += 1
    return n


def run_pipeline(limit: int = 1000, language: str = 'English', llm_model: str = 'gpt-4.1-nano',
                 max_llm_rows: Optional[int] = None, concurrency: int = 6, skip_llm: bool = False) -> Dict[str, str]:
    results: Dict[str, str] = {}
    t0 = time.time()
    logger.info(f"Pipeline start: limit={limit}, language={language}, llm_model={llm_model}, max_llm_rows={max_llm_rows or 'all'}")

    # 1) Load + filter
    t = time.time()
    logger.info("Step 1/8: Loading and filtering dataset")
    df = load_filter_to_dataframe(language=language, toxic=False, limit=limit)
    raw_path = RAW / f'wildchat_{language.lower()}_notoxic_{limit}.jsonl'
    save_dataframe_jsonl(df, str(raw_path))
    results['raw_jsonl'] = str(raw_path)
    logger.info(f"Completed step 1 in {time.time()-t:.1f}s -> {raw_path}")

    # 2) Preprocess
    t = time.time()
    logger.info("Step 2/8: Preprocessing and cleaning conversations")
    cleaned_jsonl = preprocess_and_save_from_raw(str(raw_path), output_dir=str(INTERIM))
    results['cleaned_jsonl'] = cleaned_jsonl
    logger.info(f"Completed step 2 in {time.time()-t:.1f}s -> {cleaned_jsonl}")

    # 3) BERTopic
    t = time.time()
    logger.info("Step 3/8: Running BERTopic")
    identities, docs = load_docs_from_jsonl(cleaned_jsonl)
    model, topics, probs = run_bertopic(docs)
    paths = save_outputs(model, docs, topics, identities, output_dir=str(REPORTS))
    results.update(paths)
    logger.info(f"Completed step 3 in {time.time()-t:.1f}s -> topics/doc_topics CSVs")

    # 4) Rule-based sentiment
    t = time.time()
    logger.info("Step 4/8: Rule-based sentiment (VADER/TextBlob)")
    rule_csv = ANAL / 'sentiment_rule_based_1000.csv'
    batch_process_subset(cleaned_jsonl, str(rule_csv))
    results['rule_based_csv'] = str(rule_csv)
    logger.info(f"Completed step 4 in {time.time()-t:.1f}s -> {rule_csv}")

    # 5) LLM analysis (always overwrite unless skipped)
    t = time.time()
    llm_flat_jsonl = PROC / 'llm_analysis_results.jsonl'
    if not skip_llm:
        logger.info("Step 5/8: LLM analysis starting")
        _run_llm_safe(cleaned_jsonl, str(llm_flat_jsonl), model=llm_model, max_rows=max_llm_rows, concurrency=concurrency)
        # Verify expected nested lines count
        base = os.path.splitext(str(llm_flat_jsonl))[0]
        llm_nested_jsonl = base + '_nested.jsonl'
        verify_path = llm_nested_jsonl if os.path.exists(llm_nested_jsonl) else str(llm_flat_jsonl)
        expected = _count_lines(cleaned_jsonl)
        actual = _count_lines(verify_path)
        if max_llm_rows is not None:
            expected = min(expected, int(max_llm_rows))
        if actual != expected:
            raise RuntimeError(f"LLM output rows ({actual}) != expected ({expected}) in {verify_path}.")
        logger.info(f"Completed step 5 in {time.time()-t:.1f}s -> flat={llm_flat_jsonl} nested={llm_nested_jsonl} (nested_rows={actual})")
    else:
        logger.info("Step 5/8: LLM analysis skipped by flag")
    results['llm_jsonl'] = str(llm_flat_jsonl)

    # 6) Unified table (entity-level)
    t = time.time()
    logger.info("Step 6/8: Building unified table (CSV/XLSX)")
    unified_csv = ANAL / 'unified_table.csv'
    build_unified_table(
        cleaned_jsonl_path=cleaned_jsonl,
        doc_topics_csv_path=paths.get('doc_topics_csv', paths.get('doc_topics', str(REPORTS / 'bertopic_doc_topics.csv'))),
        topics_csv_path=paths.get('topics_csv', paths.get('topics', str(REPORTS / 'bertopic_topics.csv'))),
        rule_based_csv_path=str(rule_csv),
        llm_jsonl_path=str(llm_flat_jsonl),
        output_csv_path=str(unified_csv),
    )
    results['unified_csv'] = str(unified_csv)
    logger.info(f"Completed step 6 in {time.time()-t:.1f}s -> {unified_csv}")

    # 7) Entity clustering
    t = time.time()
    logger.info("Step 7/8: Entity clustering")
    cluster_entities(str(unified_csv), output_csv_path=str(unified_csv), min_cluster_size=4)
    logger.info(f"Completed step 7 in {time.time()-t:.1f}s -> cluster columns added")

    # 8) Visualizations
    t = time.time()
    logger.info("Step 8/8: Generating visualizations")
    imgs = generate_all(str(unified_csv), out_dir=str(IMAGES))
    for k, v in imgs.items():
        results[f'img_{k}'] = v
    logger.info(f"Completed step 8 in {time.time()-t:.1f}s -> {len(imgs)} images")

    logger.info(f"Pipeline completed in {time.time()-t0:.1f}s")
    return results


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser(description='Run end-to-end analysis pipeline')
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--language', default='English')
    parser.add_argument('--llm_model', default='gpt-4.1-nano')
    parser.add_argument('--max_llm_rows', default='all')
    parser.add_argument('--concurrency', type=int, default=6)
    parser.add_argument('--skip_llm', action='store_true')
    args = parser.parse_args()

    out = run_pipeline(limit=args.limit, language=args.language, llm_model=args.llm_model,
                       max_llm_rows=_parse_max_arg(args.max_llm_rows), concurrency=args.concurrency,
                       skip_llm=args.skip_llm)
    print(json.dumps(out, indent=2)) 