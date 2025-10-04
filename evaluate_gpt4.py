#!/usr/bin/env python3
"""
Evaluate GPT-4.1 (Azure OpenAI) zero-shot classification on OCC emotions.
Usage: python evaluate_gpt4.py --input occ_sample.csv --output gpt4_results.csv
"""

import argparse
import pandas as pd
import os
import json
from openai import AzureOpenAI
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

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
    parser.add_argument("--output", default="gpt4_results.csv", help="Output CSV with predictions")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for GPT-4")
    return parser.parse_args()

def create_prompt(text):
    """Create a prompt for GPT-4.1 to classify emotions."""
    labels_str = ", ".join(OCC_LABELS)
    prompt = f"""Classify the following text into one or more of these emotions: {labels_str}

Text: "{text}"

Return ONLY a JSON array of the applicable emotions, for example: ["Joy", "Love"] or ["Neutral"]
If no emotions apply, return ["Neutral"].

JSON array:"""
    return prompt

def classify_with_gpt4(client, text, deployment):
    """Classify text using GPT-4.1."""
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": "You are an emotion classification assistant. Always respond with a valid JSON array of emotion labels."
                },
                {
                    "role": "user",
                    "content": create_prompt(text)
                }
            ],
            temperature=0.0,
            max_completion_tokens=100
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        # Try to extract JSON array from response
        if content.startswith('[') and content.endswith(']'):
            predicted_emotions = json.loads(content)
        else:
            # Try to find JSON array in the text
            start = content.find('[')
            end = content.rfind(']') + 1
            if start != -1 and end > start:
                predicted_emotions = json.loads(content[start:end])
            else:
                print(f"Warning: Could not parse response: {content}")
                predicted_emotions = ["Neutral"]
        
        return predicted_emotions
    
    except Exception as e:
        print(f"Error: {e}")
        return ["Neutral"]

def main():
    args = parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    # Initialize Azure OpenAI client
    print("Initializing Azure OpenAI client...")
    client = AzureOpenAI(
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    
    # Run inference
    print(f"Running GPT-4.1 inference on {len(df)} samples...")
    predictions = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text']
        
        # Get predictions
        predicted_emotions = classify_with_gpt4(client, text, deployment)
        
        # Convert to binary format
        pred_dict = {label: 0 for label in OCC_LABELS}
        for emotion in predicted_emotions:
            if emotion in OCC_LABELS:
                pred_dict[emotion] = 1
        
        predictions.append(pred_dict)
        
        # Rate limiting (optional, adjust as needed)
        time.sleep(0.1)
    
    # Create predictions dataframe
    pred_df = pd.DataFrame(predictions)
    
    # Add predictions to original dataframe
    result_df = df.copy()
    for label in OCC_LABELS:
        result_df[f"{label}_pred"] = pred_df[label]
    
    # Add raw predictions for inspection
    result_df["gpt4_predicted_emotions"] = [json.dumps([l for l in OCC_LABELS if pred_df.loc[i, l] == 1]) 
                                             for i in range(len(pred_df))]
    
    # Save results
    result_df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
    
    # Calculate metrics
    print("\n" + "="*60)
    print("EVALUATION METRICS - GPT-4.1")
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
    print(f"Model: GPT-4.1 (Azure OpenAI)")
    print(f"Temperature: {args.temperature}")

if __name__ == "__main__":
    main()

