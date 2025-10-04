#!/usr/bin/env python3
# map_goemotions_csv_to_occ.py
# Convert a GoEmotions-style CSV (one column per label) into an OCC-labeled sample.
# - Works with the CSV you uploaded (columns: text, id, ... + one column per label).
# - Produces CSV + JSONL with strict and lenient OCC mappings and handy review flags.
#
# Usage:
#   python map_goemotions_csv_to_occ.py --input goemotions_1.csv --count 400 --out occ_sample.csv --jsonl occ_sample.jsonl
#
# Optional flags:
#   --lenient           Use lenient (proxy) mapping as `occ_labels_final` (default uses strict only)
#   --drop-unclear      Drop rows where `example_very_unclear == 1` if that column exists
#   --seed 13           Sampling seed
#
# Notes:
# - We *do not* force single-label; examples can carry multiple OCC labels. You can filter later if you want 1 label.
# - We cap "Neutral-only" rows during stratified sampling so the sample isn't dominated by Neutral.
# - Fields `has_proxy` and `needs_manual` help you triage what to eyeball.
#
# Author: ChatGPT

from __future__ import annotations
import argparse, csv, json, random
from typing import List, Dict, Any
import pandas as pd

# GoEmotions label names present in the public dataset (27 + neutral)
GO_LABELS = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval","disgust",
    "embarrassment","excitement","fear","gratitude","grief","joy","love",
    "nervousness","optimism","pride","realization","relief","remorse","sadness",
    "surprise","neutral"
]

# OCC label universe (as provided by the user)
OCC_LABELS = [
    "Neutral","Happy-for","Resentment","Gloating","Pity","Satisfaction","Fears-Confirmed","Relief",
    "Disappointment","Gratification","Remorse","Gratitude","Anger","Love","Hate","Hope","Fear",
    "Joy","Distress","Pride","Shame","Admiration","Reproach"
]

# STRICT mapping: confident one-to-one mappings
GO_TO_OCC_STRICT: Dict[str, str] = {
    "neutral": "Neutral",
    "admiration": "Admiration",
    "anger": "Anger",
    "disappointment": "Disappointment",
    "disapproval": "Reproach",
    "fear": "Fear",
    "gratitude": "Gratitude",
    "grief": "Distress",
    "joy": "Joy",
    "love": "Love",
    "optimism": "Hope",
    "pride": "Pride",
    "relief": "Relief",
    "remorse": "Remorse",
    "sadness": "Distress",
}

# LENIENT proxies (optional): broaden coverage but mark via `has_proxy=True`
GO_TO_OCC_LENIENT: Dict[str, str] = {
    "embarrassment": "Shame",
    "disgust": "Hate",
    "annoyance": "Anger",
    "amusement": "Joy",
    "excitement": "Joy",
    "nervousness": "Fear",
    "caring": "Love",
    "approval": "Admiration",
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to GoEmotions-style CSV (one column per label)")
    ap.add_argument("--out", default="occ_sample.csv", help="Output CSV path")
    ap.add_argument("--jsonl", default="occ_sample.jsonl", help="Optional JSONL path")
    ap.add_argument("--count", type=int, default=400, help="How many rows to sample")
    ap.add_argument("--seed", type=int, default=13, help="Sampling seed")
    ap.add_argument("--lenient", action="store_true", help="Use lenient (proxy) mapping as `occ_labels_final`")
    ap.add_argument("--drop-unclear", action="store_true", help="Drop rows where `example_very_unclear == 1` if present")
    return ap.parse_args()

def map_row_to_occ(row: pd.Series, label_cols: List[str]) -> Dict[str, Any]:
    go_present = [c for c in label_cols if row.get(c, 0) == 1]
    strict = sorted({GO_TO_OCC_STRICT[g] for g in go_present if g in GO_TO_OCC_STRICT})
    proxy = sorted({GO_TO_OCC_LENIENT[g] for g in go_present if g in GO_TO_OCC_LENIENT})
    unmapped = sorted([g for g in go_present if g not in GO_TO_OCC_STRICT and g not in GO_TO_OCC_LENIENT])
    has_proxy = bool(proxy)
    needs_manual = (len(strict) == 0) or has_proxy or bool(unmapped)
    return {
        "goemotions_labels": go_present,
        "occ_labels_strict": strict,
        "occ_labels_lenient": sorted(set(strict) | set(proxy)),
        "source_labels_unmapped": unmapped,
        "has_proxy": has_proxy,
        "needs_manual": needs_manual,
    }

def stratified_sample(df_mapped: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    random.seed(seed)
    # Buckets by first strict OCC label, excluding Neutral for balance
    non_neutral = df_mapped[df_mapped["occ_labels_strict"].map(lambda x: (len(x) > 0) and (len(set(x) - {"Neutral"}) > 0))].copy()
    neutral_only = df_mapped[df_mapped["occ_labels_strict"].map(lambda x: (x == ["Neutral"]) or (set(x) == {"Neutral"}))].copy()
    other = df_mapped[df_mapped["occ_labels_strict"].map(lambda x: len(x) == 0)].copy()

    # Build buckets for non-neutral strict labels
    import itertools, collections
    buckets = collections.defaultdict(list)
    for idx, row in non_neutral.iterrows():
        key = sorted(set(row["occ_labels_strict"]) - {"Neutral"})[0]
        buckets[key].append(idx)

    # target per-bucket for non-neutral labels
    k = max(1, count // (len(buckets) + 2)) if buckets else max(1, count // 3)

    chosen = []

    # Round 1: up to k per non-neutral bucket
    for key, idxs in buckets.items():
        random.shuffle(idxs)
        chosen.extend(idxs[:k])

    # Round 2: add some neutral-only (cap at ~20% of desired count)
    cap_neutral = max(1, int(0.2 * count))
    neu_idxs = neutral_only.index.tolist()
    random.shuffle(neu_idxs)
    chosen.extend(neu_idxs[:cap_neutral])

    # Round 3: fill remainder from "other" (no strict labels) then from remaining non-neutral/neutral
    def fill_from(pool_idxs):
        nonlocal chosen
        for i in pool_idxs:
            if len(chosen) >= count:
                break
            if i not in chosen:
                chosen.append(i)

    other_idxs = other.index.tolist()
    random.shuffle(other_idxs)
    fill_from(other_idxs)

    remaining = [i for i in non_neutral.index.tolist() + neu_idxs if i not in chosen]
    random.shuffle(remaining)
    fill_from(remaining)

    chosen = chosen[:count]
    return df_mapped.loc[chosen].copy()

def main():
    args = parse_args()
    df = pd.read_csv(args.input, low_memory=False)

    # Optional filter: drop "very unclear" examples if present
    if args.drop_unclear and "example_very_unclear" in df.columns:
        df = df[df["example_very_unclear"] != 1].copy()

    # Identify label columns present in the file
    label_cols = [c for c in df.columns if c in GO_LABELS]
    if not label_cols:
        raise SystemExit("Could not find any GoEmotions label columns in the CSV.")

    # Minimal required text/id columns (best effort)
    text_col = "text" if "text" in df.columns else None
    id_col = "id" if "id" in df.columns else None

    # Map each row to OCC
    mapped_records = []
    for _, row in df.iterrows():
        m = map_row_to_occ(row, label_cols)
        rec = {
            "id": row[id_col] if id_col else "",
            "text": row[text_col] if text_col else "",
            **m
        }
        rec["occ_labels_final"] = rec["occ_labels_lenient"] if args.lenient else rec["occ_labels_strict"]
        mapped_records.append(rec)
    out_df = pd.DataFrame(mapped_records)

    # Stratified sample
    sampled = stratified_sample(out_df, count=args.count, seed=args.seed)

    # Write CSV in GoEmotions format (one column per OCC label with 0/1)
    csv_df = sampled[["id", "text"]].copy()
    
    # Add one column per OCC label
    for occ_label in OCC_LABELS:
        csv_df[occ_label] = sampled["occ_labels_final"].map(lambda labels: 1 if occ_label in labels else 0)
    
    # Add readable mapped OCC labels column (as JSON list for easy scanning)
    def dumps_list(x):
        import json
        return json.dumps(x, ensure_ascii=False)
    
    csv_df["mapped_occ_labels"] = sampled["occ_labels_final"].map(dumps_list)
    
    # Add metadata columns for reference (as JSON strings)
    csv_df["goemotions_labels"] = sampled["goemotions_labels"].map(dumps_list)
    csv_df["source_labels_unmapped"] = sampled["source_labels_unmapped"].map(dumps_list)
    csv_df["has_proxy"] = sampled["has_proxy"]
    csv_df["needs_manual"] = sampled["needs_manual"]

    csv_df.to_csv(args.out, index=False, quoting=csv.QUOTE_MINIMAL)

    # JSONL
    if args.jsonl:
        with open(args.jsonl, "w", encoding="utf-8") as f:
            for _, r in sampled.iterrows():
                f.write(json.dumps({k: r[k] for k in sampled.columns}, ensure_ascii=False) + "\n")

    # Summary to stdout
    from collections import Counter
    counts_final = Counter(l for labels in sampled["occ_labels_final"] for l in labels)
    print(f"Wrote {len(sampled)} rows to {args.out}")
    if args.jsonl:
        print(f"Also wrote JSONL to {args.jsonl}")
    print(f"\nFormat: id, text, + one column per OCC label (0/1), + metadata columns")
    print(f"\nOCC label counts in the sample:")
    for k, v in counts_final.most_common():
        print(f"  {k:>18}: {v}")
    print(f"\nRows marked needs_manual=True: {sampled['needs_manual'].sum()}")
    print(f"Rows with proxy mappings: {sampled['has_proxy'].sum()}")

if __name__ == "__main__":
    main()
