#!/usr/bin/env python3
"""
Run BART on the validation dataset (38 sentences with scored labels).
This generates predictions that we'll later compare against the scores.
"""

import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

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
    
    # Run inference
    print(f"\nRunning BART inference on {len(df)} samples...")
    predictions = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row['Paste Current Sentence in Here']).strip()
        
        # Get BART predictions
        result = classifier(
            text,
            candidate_labels=OCC_LABELS,
            multi_label=True
        )
        
        # Convert scores to binary (threshold 0.5)
        pred_dict = {'text': text}
        for label, score in zip(result['labels'], result['scores']):
            pred_dict[f"{label}_pred"] = 1 if score >= 0.5 else 0
        
        predictions.append(pred_dict)
    
    # Save predictions
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv('bart_validation_predictions.csv', index=False)
    
    print(f"\n{'='*60}")
    print(f"Saved predictions to: bart_validation_predictions.csv")
    print(f"{'='*60}")
    print(f"\nNext: Run evaluation script to compare against scored ground truth")

if __name__ == "__main__":
    main()

