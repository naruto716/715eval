#!/usr/bin/env python3
"""
Evaluate BART zero-shot classification on OCC emotions.
Usage: python evaluate_bart.py --input occ_sample.csv --output bart_results.csv
"""

import argparse
import pandas as pd
import torch
from transformers import pipeline
from sklearn.metrics import precision_recall_fscore_support, classification_report
from tqdm import tqdm

# OCC emotion labels
OCC_LABELS = [
    "Neutral", "Happy-for", "Resentment", "Gloating", "Pity", 
    "Satisfaction", "Fears-Confirmed", "Relief", "Disappointment", 
    "Gratification", "Remorse", "Gratitude", "Anger", "Love", 
    "Hate", "Hope", "Fear", "Joy", "Distress", "Pride", 
    "Shame", "Admiration", "Reproach"
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="occ_sample.csv", help="Input CSV with ground truth")
    parser.add_argument("--output", default="bart_results.csv", help="Output CSV with predictions")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for multi-label classification")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    # Initialize BART pipeline
    print("Loading BART model (facebook/bart-large-mnli)...")
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device
    )
    
    # Run inference
    print(f"Running inference on {len(df)} samples...")
    predictions = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text']
        
        # Get predictions
        result = classifier(
            text,
            candidate_labels=OCC_LABELS,
            multi_label=True
        )
        
        # Convert scores to binary predictions using threshold
        pred_dict = {label: 0 for label in OCC_LABELS}
        for label, score in zip(result['labels'], result['scores']):
            if score >= args.threshold:
                pred_dict[label] = 1
        
        predictions.append(pred_dict)
    
    # Create predictions dataframe
    pred_df = pd.DataFrame(predictions)
    
    # Add predictions to original dataframe
    result_df = df.copy()
    for label in OCC_LABELS:
        result_df[f"{label}_pred"] = pred_df[label]
    
    # Save results
    result_df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
    
    # Calculate metrics
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    # Prepare ground truth and predictions
    y_true = df[OCC_LABELS].values
    y_pred = pred_df[OCC_LABELS].values
    
    # Overall metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    print(f"\nOverall Metrics (Multi-label):")
    print(f"  Macro Precision: {precision:.4f}")
    print(f"  Macro Recall:    {recall:.4f}")
    print(f"  Macro F1:        {f1:.4f}")
    
    # Micro metrics
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    print(f"\n  Micro Precision: {precision_micro:.4f}")
    print(f"  Micro Recall:    {recall_micro:.4f}")
    print(f"  Micro F1:        {f1_micro:.4f}")
    
    # Per-label metrics
    print(f"\nPer-Label Metrics:")
    print(f"{'Label':<20} {'Support':>8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)
    
    for i, label in enumerate(OCC_LABELS):
        label_support = y_true[:, i].sum()
        if label_support > 0:
            p, r, f, _ = precision_recall_fscore_support(
                y_true[:, i], y_pred[:, i], average='binary', zero_division=0
            )
            print(f"{label:<20} {int(label_support):>8} {p:>10.4f} {r:>10.4f} {f:>10.4f}")
        else:
            print(f"{label:<20} {int(label_support):>8} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
    
    print("\n" + "="*60)
    print(f"Threshold used: {args.threshold}")
    print(f"Device: {'GPU' if device == 0 else 'CPU'}")

if __name__ == "__main__":
    main()

