#!/usr/bin/env python3
"""
Analyze model predictions to find cases where models might be more accurate than the mapped ground truth.
"""

import pandas as pd
import json

# OCC labels that don't exist in GoEmotions (models can predict these but ground truth can't have them)
OCC_ONLY_LABELS = [
    "Happy-for", "Resentment", "Gloating", "Pity", 
    "Satisfaction", "Gratification", "Fears-Confirmed"
]

def load_labels(row, label_type="true"):
    """Extract active labels from a row."""
    if label_type == "true":
        # Ground truth from mapped_occ_labels
        return json.loads(row['mapped_occ_labels'])
    else:
        # Predictions from gpt4_predicted_emotions or calculate from _pred columns
        if 'gpt4_predicted_emotions' in row:
            return json.loads(row['gpt4_predicted_emotions'])
        else:
            # For BART, reconstruct from _pred columns
            from evaluate_bart import OCC_LABELS
            return [label for label in OCC_LABELS if row.get(f"{label}_pred", 0) == 1]

def analyze_bart():
    print("\n" + "="*80)
    print("BART ANALYSIS: Cases where model predicts OCC-specific emotions")
    print("="*80)
    
    df = pd.read_csv('bart_results.csv')
    
    # Find cases where BART predicted OCC-specific emotions
    interesting_cases = []
    
    for idx, row in df.iterrows():
        true_labels = json.loads(row['mapped_occ_labels'])
        
        # Check if BART predicted any OCC-specific labels
        for occ_label in OCC_ONLY_LABELS:
            if row[f"{occ_label}_pred"] == 1:
                interesting_cases.append({
                    'text': row['text'],
                    'ground_truth': true_labels,
                    'bart_predicted': occ_label,
                    'goemotions_source': json.loads(row['goemotions_labels'])
                })
    
    print(f"\nFound {len(interesting_cases)} cases where BART predicted OCC-specific emotions:")
    print("-" * 80)
    
    for i, case in enumerate(interesting_cases[:20], 1):  # Show first 20
        print(f"\n{i}. Text: {case['text'][:100]}...")
        print(f"   Ground Truth: {case['ground_truth']}")
        print(f"   BART Added: {case['bart_predicted']}")
        print(f"   GoEmotions Source: {case['goemotions_source']}")

def analyze_gpt4():
    print("\n" + "="*80)
    print("GPT-4.1 ANALYSIS: Cases where model predicts OCC-specific emotions")
    print("="*80)
    
    df = pd.read_csv('gpt4_results.csv')
    
    # Find cases where GPT-4 predicted OCC-specific emotions
    interesting_cases = []
    
    for idx, row in df.iterrows():
        true_labels = json.loads(row['mapped_occ_labels'])
        pred_labels = json.loads(row['gpt4_predicted_emotions'])
        
        # Check if GPT-4 predicted any OCC-specific labels
        occ_specific_preds = [l for l in pred_labels if l in OCC_ONLY_LABELS]
        
        if occ_specific_preds:
            interesting_cases.append({
                'text': row['text'],
                'ground_truth': true_labels,
                'gpt4_predicted': pred_labels,
                'occ_specific': occ_specific_preds,
                'goemotions_source': json.loads(row['goemotions_labels'])
            })
    
    print(f"\nFound {len(interesting_cases)} cases where GPT-4 predicted OCC-specific emotions:")
    print("-" * 80)
    
    for i, case in enumerate(interesting_cases[:30], 1):  # Show first 30
        print(f"\n{i}. Text: {case['text'][:100]}...")
        print(f"   Ground Truth: {case['ground_truth']}")
        print(f"   GPT-4 Predicted: {case['gpt4_predicted']}")
        print(f"   OCC-Specific: {case['occ_specific']}")
        print(f"   GoEmotions Source: {case['goemotions_source']}")

def analyze_disagreements():
    """Find cases where models agree with each other but disagree with ground truth."""
    print("\n" + "="*80)
    print("MODEL AGREEMENT: Cases where both models agree but differ from ground truth")
    print("="*80)
    
    bart_df = pd.read_csv('bart_results.csv')
    gpt4_df = pd.read_csv('gpt4_results.csv')
    
    agreements = []
    
    for idx in range(len(bart_df)):
        bart_row = bart_df.iloc[idx]
        gpt4_row = gpt4_df.iloc[idx]
        
        true_labels = set(json.loads(bart_row['mapped_occ_labels']))
        bart_pred = set([l for l in ["Resentment", "Gloating", "Pity", "Satisfaction", 
                                      "Gratification", "Happy-for", "Fears-Confirmed"]
                         if bart_row.get(f"{l}_pred", 0) == 1])
        gpt4_pred = set(json.loads(gpt4_row['gpt4_predicted_emotions']))
        
        # Both models predicted something that's not in ground truth and they agree on it
        common_additions = (bart_pred | gpt4_pred) - true_labels
        
        if common_additions and len(common_additions.intersection(bart_pred).intersection(gpt4_pred)) > 0:
            agreements.append({
                'text': bart_row['text'],
                'ground_truth': list(true_labels),
                'both_added': list(common_additions.intersection(bart_pred).intersection(gpt4_pred)),
                'goemotions_source': json.loads(bart_row['goemotions_labels'])
            })
    
    print(f"\nFound {len(agreements)} cases where both models agree on adding OCC-specific emotions:")
    print("-" * 80)
    
    for i, case in enumerate(agreements[:20], 1):
        print(f"\n{i}. Text: {case['text'][:100]}...")
        print(f"   Ground Truth: {case['ground_truth']}")
        print(f"   Both Models Added: {case['both_added']}")
        print(f"   GoEmotions Source: {case['goemotions_source']}")

if __name__ == "__main__":
    analyze_bart()
    analyze_gpt4()
    analyze_disagreements()
    
    print("\n" + "="*80)
    print("RECOMMENDATION:")
    print("="*80)
    print("""
The models are predicting OCC-specific emotions (Resentment, Gloating, Pity, etc.) 
that don't exist in the GoEmotions-mapped ground truth. This creates false negatives.

Consider:
1. Manually reviewing cases where models predict OCC-specific emotions
2. Re-labeling those that are actually correct
3. Creating a "corrected" evaluation set with proper OCC labels
    """)

