"""
5-Fold Cross-Validation evaluation of Qwen for AI-generated text detection.

For each fold:
  - 80% is ignored (no training — Qwen is zero-shot)
  - 20% is used as the test set
  - Qwen classifies each article in the test set
  - Metrics are computed per fold

Final output: mean ± std across all 5 folds.

Usage:
    python qwen_kfold.py --csv dataset_balanced.csv --model_path ~/models/Qwen2.5-7B-Instruct
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_FOLDS       = 5


# ── Logger setup ──────────────────────────────────────────────────────────────

def setup_logger(log_path: str) -> logging.Logger:
    """Write logs to both the terminal and a .log file simultaneously."""
    logger = logging.getLogger("qwen_kfold")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Wikipedia content auditor trained to detect AI-generated text.

Below are the OFFICIAL signs of AI writing documented by Wikipedia's WikiProject AI Cleanup (WP:AISIGNS). Use these as your detection criteria:

## CONTENT SIGNS
1. Undue emphasis on significance and legacy — phrases like "stands as a testament to", "marks a pivotal moment", "reflects broader trends", "shaping the landscape", "indelible mark", "deeply rooted", "setting the stage for", "key turning point", "evolving landscape".
2. Canned notability claims — phrases like "independent coverage", "profiled in national/regional media outlets", "maintains an active social media presence", "has been featured in".
3. Superficial analysis — sentences ending with present-participle phrases like "highlighting its importance", "underscoring its significance", "reflecting the broader", "contributing to the", "cultivating a sense of", "ensuring that".
4. Promotional language — over-positive, advertisement-like descriptions with no neutral tone.
5. Vague attributions — claims like "experts say", "many believe", "it is widely recognized" without citing specific sources.
6. Outline-like conclusions — generic paragraphs about "challenges and future prospects" that could apply to any topic.

## LANGUAGE SIGNS
7. High-density AI vocabulary — frequent use of: delve, testament, comprehensive, pivotal, notable, significant, crucial, vibrant, multifaceted, nuanced, intricate, commendable, invaluable, paramount, groundbreaking, revolutionary, beacon, tapestry, bustling, rich history, it is worth noting, in the realm of, stands out, not only X but also Y.
8. Negative parallelism — structures like "Not just X, but also Y" or "Not X, but Y".
9. Rule of three — listing exactly three items repeatedly ("efficiency, clarity, and impact").
10. Elegant variation — avoiding repeating a word by using unnecessary synonyms awkwardly.
11. Avoidance of simple "is/are" — replacing simple statements with complex nominalized forms.

## STYLE SIGNS
12. Overuse of em dashes — using — dashes far more than a human Wikipedia editor would.
13. Inline-header vertical lists — bold word at the start of a bullet point acting as a mini-heading.
14. Unusual title case — capitalizing words that should not be capitalized.

## SIGNS OF HUMAN WRITING
- Specific, unusual facts that are too niche to be statistically common
- Natural flow variation — some sentences short, some long, with genuine rhythm
- Hedged, cautious language backed by citations
- Content written clearly before ChatGPT's launch (November 2022)

---
Your task: Read the Wikipedia article excerpt and decide if it was written by a HUMAN or an AI.

Respond with ONLY one word: AI or Human
Do NOT explain. Do NOT add punctuation. Just one word."""

EXAMPLES_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Wikipedia article excerpt:\n\n"
            "The Battle of Hastings was fought on 14 October 1066 between the Norman-French army "
            "of William, the Duke of Normandy, and an English army under the Anglo-Saxon King "
            "Harold II, near the present-day town of Battle, East Sussex. Harold's forces "
            "occupied a ridge north of Hastings; the Normans attacked the following morning.\n\n"
            "Is this Human or AI-generated?"
        ),
    },
    {"role": "assistant", "content": "Human"},
    {
        "role": "user",
        "content": (
            "Wikipedia article excerpt:\n\n"
            "The Statistical Institute of Catalonia was officially established in 1989, marking "
            "a pivotal moment in the evolution of regional statistics in Spain. The founding of "
            "Idescat represented a significant shift toward regional statistical independence, "
            "enabling Catalonia to develop a comprehensive statistical system tailored to its "
            "unique socio-economic context. This initiative was part of a broader movement "
            "across Spain to decentralize administrative functions and enhance regional "
            "governance, reflecting the enduring importance of data-driven decision making.\n\n"
            "Is this Human or AI-generated?"
        ),
    },
    {"role": "assistant", "content": "AI"},
]


# ── GPU helpers ───────────────────────────────────────────────────────────────

def gpu_memory_str() -> str:
    if not torch.cuda.is_available():
        return "GPU: N/A"
    used  = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"GPU mem: {used:.1f}/{total:.1f} GB"


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_path: str, log: logging.Logger):
    log.info("=" * 55)
    log.info("LOADING MODEL")
    log.info("=" * 55)
    log.info(f"Model path : {model_path}")

    t0 = time.time()
    log.info("Step 1/2 — Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    log.info(f"  Tokenizer loaded  ({time.time()-t0:.1f}s)")

    log.info("Step 2/2 — Loading model weights into GPU memory...")
    t1 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    load_time = time.time() - t1
    log.info(f"  Model loaded ({load_time:.1f}s)")
    log.info(f"  Device     : {next(model.parameters()).device}")
    log.info(f"  {gpu_memory_str()}")
    log.info("=" * 55)
    return model, tokenizer


# ── Inference ─────────────────────────────────────────────────────────────────

def ask_qwen(model, tokenizer, text: str, max_chars: int = 2000) -> tuple[int, str]:
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    user_msg  = {
        "role": "user",
        "content": (
            f"Wikipedia article excerpt:\n\n{truncated}\n\n"
            "Is this Human or AI-generated?"
        ),
    }
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(EXAMPLES_MESSAGES)
    messages.append(user_msg)

    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw        = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    label      = 1 if "ai" in raw.lower() else 0
    return label, raw


# ── Fold runner ───────────────────────────────────────────────────────────────

def run_fold(fold_num, n_folds, texts, y_true, article_ids,
             model, tokenizer, checkpoint_path, log: logging.Logger):

    total       = len(texts)
    predictions = []
    raw_responses = []
    ai_count    = 0
    human_count = 0
    error_count = 0
    log_every   = max(1, total // 10)   # every 10%

    log.info("")
    log.info("=" * 55)
    log.info(f"FOLD {fold_num}/{n_folds}  —  {total} articles to classify")
    log.info("=" * 55)
    fold_start = time.time()

    for i, (text, art_id) in enumerate(zip(texts, article_ids)):
        t0 = time.time()
        try:
            label, raw = ask_qwen(model, tokenizer, text)
        except torch.cuda.OutOfMemoryError:
            log.warning(f"Article {i+1} (id={art_id}) OOM — retrying with shorter text...")
            torch.cuda.empty_cache()
            try:
                label, raw = ask_qwen(model, tokenizer, text, max_chars=500)
            except Exception as e2:
                log.warning(f"Article {i+1} retry also failed: {e2}")
                label, raw = 0, "ERROR"
                error_count += 1
        except Exception as e:
            log.warning(f"Article {i+1} (id={art_id}) FAILED: {e}")
            label, raw = 0, "ERROR"
            error_count += 1

        predictions.append(label)
        raw_responses.append(raw)
        label_str = "AI" if label == 1 else "Human"
        if label == 1:
            ai_count += 1
        else:
            human_count += 1

        done      = i + 1
        elapsed   = time.time() - fold_start
        avg_sec   = elapsed / done
        remaining = avg_sec * (total - done)

        # Log every article at DEBUG level (visible in log file)
        log.debug(
            f"  Article {done:>4}/{total} | id={art_id} | "
            f"pred={label_str:<5} | raw='{raw}' | {time.time()-t0:.1f}s"
        )

        # Log milestone summary at INFO level (visible in terminal too)
        if done % log_every == 0 or done == total:
            log.info(
                f"  Progress : {done}/{total} ({done/total*100:.0f}%) | "
                f"AI={ai_count} Human={human_count} Errors={error_count} | "
                f"Elapsed={int(elapsed//60)}m{int(elapsed%60):02d}s | "
                f"ETA={int(remaining//60)}m{int(remaining%60):02d}s | "
                f"Speed={avg_sec:.1f}s/article | {gpu_memory_str()}"
            )

    fold_time = time.time() - fold_start
    log.info(f"  Fold {fold_num} finished in {int(fold_time//60)}m {int(fold_time%60):02d}s")

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc  = accuracy_score(y_true, predictions)
    f1   = f1_score(y_true, predictions, average="binary", zero_division=0)
    prec = precision_score(y_true, predictions, average="binary", zero_division=0)
    rec  = recall_score(y_true, predictions, average="binary", zero_division=0)
    cm   = confusion_matrix(y_true, predictions)
    tn, fp, fn, tp = cm.ravel()
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    log.info("")
    log.info(f"  FOLD {fold_num} RESULTS")
    log.info(f"  {'Accuracy':<12}: {acc:.4f}")
    log.info(f"  {'F1':<12}: {f1:.4f}")
    log.info(f"  {'Precision':<12}: {prec:.4f}")
    log.info(f"  {'Recall':<12}: {rec:.4f}")
    log.info(f"  {'FPR':<12}: {fpr:.4f}  (false alarms on human articles)")
    log.info(f"  Confusion Matrix : TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    log.info("")
    log.info(classification_report(y_true, predictions,
                                   target_names=["Human (0)", "AI (1)"]))

    # ── Save checkpoint (so results aren't lost if job dies) ──────────────────
    ckpt_df = pd.DataFrame({
        "article_id": article_ids,
        "true_label": y_true,
        "qwen_pred":  predictions,
        "qwen_raw":   raw_responses,
        "fold":       fold_num,
    })
    ckpt_df.to_csv(checkpoint_path, index=False)
    log.info(f"  Checkpoint saved : {checkpoint_path}")

    return predictions, raw_responses, {
        "accuracy": acc, "f1": f1,
        "precision": prec, "recall": rec, "fpr": fpr,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        type=str, required=True)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--text_col",   type=str, default="content")
    parser.add_argument("--label_col",  type=str, default="is_ai_flagged")
    parser.add_argument("--id_col",     type=str, default="article_id")
    parser.add_argument("--n_folds",    type=int, default=N_FOLDS)
    parser.add_argument("--output",     type=str, default="qwen_kfold_results.csv")
    parser.add_argument("--log_dir",    type=str, default="logs")
    parser.add_argument("--seed",       type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Logger ────────────────────────────────────────────────────────────────
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"qwen_kfold_{run_id}.log")
    log      = setup_logger(log_file)

    log.info("=" * 55)
    log.info("  Qwen 5-Fold CV — AI Detection on Wikipedia")
    log.info("=" * 55)
    log.info(f"  Run ID     : {run_id}")
    log.info(f"  Log file   : {log_file}")
    log.info(f"  Model      : {args.model_path}")
    log.info(f"  Dataset    : {args.csv}")
    log.info(f"  Folds      : {args.n_folds}")
    log.info(f"  Output     : {args.output}")
    log.info(f"  Seed       : {args.seed}")
    log.info(f"  CUDA avail : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"  GPU        : {torch.cuda.get_device_name(0)}")
        log.info(f"  {gpu_memory_str()}")
    log.info("=" * 55)

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("")
    log.info("LOADING DATASET")
    t0 = time.time()
    df = pd.read_csv(args.csv)
    df = df.dropna(subset=[args.text_col, args.label_col])
    log.info(f"  File loaded in {time.time()-t0:.1f}s")
    log.info(f"  Total articles  : {len(df)}")
    log.info(f"  AI-generated    : {df[args.label_col].sum()}")
    log.info(f"  Human-written   : {(df[args.label_col] == 0).sum()}")
    log.info(f"  Columns         : {list(df.columns)}")

    texts  = df[args.text_col].fillna("").tolist()
    labels = df[args.label_col].astype(int).tolist()
    ids    = df[args.id_col].tolist() if args.id_col in df.columns else list(range(len(df)))

    # ── Load model ────────────────────────────────────────────────────────────
    log.info("")
    model, tokenizer = load_model(args.model_path, log)

    # ── K-Fold split ──────────────────────────────────────────────────────────
    log.info("")
    log.info(f"SPLITTING INTO {args.n_folds} STRATIFIED FOLDS (seed={args.seed})")
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    X   = np.array(texts)
    y   = np.array(labels)
    ID  = np.array(ids)

    all_metrics  = []
    all_preds_df = []
    job_start    = time.time()

    for fold_num, (_, test_idx) in enumerate(skf.split(X, y), start=1):
        ckpt_path = os.path.join(
            args.log_dir, f"checkpoint_fold{fold_num}_{run_id}.csv"
        )
        preds, raws, metrics = run_fold(
            fold_num     = fold_num,
            n_folds      = args.n_folds,
            texts        = X[test_idx].tolist(),
            y_true       = y[test_idx].tolist(),
            article_ids  = ID[test_idx].tolist(),
            model        = model,
            tokenizer    = tokenizer,
            checkpoint_path = ckpt_path,
            log          = log,
        )
        all_metrics.append(metrics)

        fold_df = df.iloc[test_idx].copy()
        fold_df["fold"]       = fold_num
        fold_df["qwen_pred"]  = preds
        fold_df["qwen_raw"]   = raws
        fold_df["qwen_label"] = ["AI-generated" if p == 1 else "Human-written" for p in preds]
        all_preds_df.append(fold_df)

        total_elapsed = time.time() - job_start
        remaining_folds = args.n_folds - fold_num
        avg_fold_time = total_elapsed / fold_num
        log.info(
            f"  Overall: {fold_num}/{args.n_folds} folds done | "
            f"Total elapsed: {int(total_elapsed//3600)}h{int((total_elapsed%3600)//60):02d}m | "
            f"Est. remaining: {int(avg_fold_time*remaining_folds//3600)}h"
            f"{int((avg_fold_time*remaining_folds%3600)//60):02d}m"
        )

    # ── Final summary ─────────────────────────────────────────────────────────
    total_time = time.time() - job_start
    log.info("")
    log.info("=" * 55)
    log.info("  FINAL SUMMARY — Mean ± Std across all folds")
    log.info("=" * 55)

    metric_names = ["accuracy", "f1", "precision", "recall", "fpr"]
    summary = {}
    for m in metric_names:
        vals      = [fold[m] for fold in all_metrics]
        mean, std = np.mean(vals), np.std(vals)
        summary[m] = (mean, std)
        per_fold  = "  ".join([f"{v:.4f}" for v in vals])
        log.info(f"  {m:<12}: {mean:.4f} ± {std:.4f}   [{per_fold}]")

    fpr_mean = summary["fpr"][0]
    log.info("")
    log.info(f"  FPR Assessment : {'GOOD (< 5%)' if fpr_mean < 0.05 else 'HIGH — review prompting'}")
    log.info(f"  Total run time : {int(total_time//3600)}h {int((total_time%3600)//60):02d}m")
    log.info("=" * 55)

    # ── Save outputs ──────────────────────────────────────────────────────────
    result_df = pd.concat(all_preds_df, ignore_index=True)
    result_df.to_csv(args.output, index=False)
    log.info(f"\n  All predictions : {args.output}")

    summary_path = args.output.replace(".csv", "_summary.csv")
    summary_df = pd.DataFrame([
        {"metric": m, "mean": v[0], "std": v[1]}
        for m, v in summary.items()
    ])
    summary_df.to_csv(summary_path, index=False)
    log.info(f"  Summary table   : {summary_path}")
    log.info(f"  Full log file   : {log_file}")
    log.info("")
    log.info("  DONE.")


if __name__ == "__main__":
    main()
