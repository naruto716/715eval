#!/usr/bin/env python3
"""
Run BART on the validation dataset and evaluate against scored ground truth.
Ground truth: score >= 7 counts as positive label.

FOR SAGEMAKER - NOT FOR LOCAL EXECUTION
"""

import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

OCC_LABELS = [
    "Neutral", "Happy-for", "Resentment", "Gloating", "Pity", 
    "Satisfaction", "Fears-Confirmed", "Relief", "Disappointment", 
    "Gratification", "Remorse", "Gratitude", "Anger", "Love", 
    "Hate", "Hope", "Fear", "Joy", "Distress", "Pride", 
    "Shame", "Admiration", "Reproach"
]

def main():
    # Load validation data
    print("Loading validation data...")
    df = pd.read_csv('CS715 Model Validation (Responses) - Form responses 1.csv')
    
    # Skip first row (test data)
    df = df.iloc[1:].reset_index(drop=True)
    
    print(f"Found {len(df)} sentences to classify\n")
    
    # Initialize BART
    print("Loading BART model (facebook/bart-large-mnli)...")
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device
    )
    
    # Prepare ground truth (score >= 7 = positive)
    print("\nPreparing ground truth (threshold >= 7)...")
    y_true = []
    for _, row in df.iterrows():
        true_labels = []
        for label in OCC_LABELS:
            score = row.get(label, 0)
            try:
                score = float(score) if pd.notna(score) else 0
                true_labels.append(1 if score >= 7 else 0)
            except:
                true_labels.append(0)
        y_true.append(true_labels)
    
    y_true = np.array(y_true)
    
    # Run inference
    print(f"\nRunning BART inference on {len(df)} samples...")
    y_pred = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row['Paste Current Sentence in Here']).strip()
        
        # Get BART predictions
        result = classifier(
            text,
            candidate_labels=OCC_LABELS,
            multi_label=True
        )
        
        # Convert scores to binary (threshold 0.5 for BART)
        pred_labels = []
        for label in OCC_LABELS:
            # Find score for this label
            score_idx = result['labels'].index(label)
            score = result['scores'][score_idx]
            pred_labels.append(1 if score >= 0.5 else 0)
        
        y_pred.append(pred_labels)
    
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    print(f"\n{'='*60}")
    print("EVALUATION METRICS - BART vs Scored Ground Truth (>= 7)")
    print(f"{'='*60}\n")
    
    # Overall metrics
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    micro_prec, micro_rec, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    print("Overall Metrics (Multi-label):")
    print(f"  Macro Precision: {macro_prec:.4f}")
    print(f"  Macro Recall:    {macro_rec:.4f}")
    print(f"  Macro F1:        {macro_f1:.4f}")
    print()
    print(f"  Micro Precision: {micro_prec:.4f}")
    print(f"  Micro Recall:    {micro_rec:.4f}")
    print(f"  Micro F1:        {micro_f1:.4f}")
    
    # Per-label metrics
    print(f"\n{'Per-Label Metrics:'}")
    print(f"{'Label':<20} {'Support':>8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)
    
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    for i, label in enumerate(OCC_LABELS):
        if support[i] > 0:
            print(f"{label:<20} {int(support[i]):>8} {prec[i]:>10.4f} {rec[i]:>10.4f} {f1[i]:>10.4f}")
        else:
            print(f"{label:<20} {int(support[i]):>8} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
    
    # Save detailed results
    results_df = df.copy()
    for i, label in enumerate(OCC_LABELS):
        results_df[f"{label}_bart_pred"] = y_pred[:, i]
    
    results_df.to_csv('bart_validation_results.csv', index=False)
    print(f"\n{'='*60}")
    print(f"Detailed results saved to: bart_validation_results.csv")

if __name__ == "__main__":
    main()