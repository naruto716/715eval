#!/usr/bin/env python3
"""
Create proper OCC ground truth labels by combining:
1. Exact word-to-word matches from GoEmotions (love→Love, joy→Joy, etc.)
2. GPT-4 reading the text fresh and labeling with all 23 OCC emotions
3. Union of both

Then you manually review and fix any errors.
"""

import pandas as pd
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
from tqdm import tqdm
import time

load_dotenv()

OCC_LABELS = [
    "Neutral", "Happy-for", "Resentment", "Gloating", "Pity", 
    "Satisfaction", "Fears-Confirmed", "Relief", "Disappointment", 
    "Gratification", "Remorse", "Gratitude", "Anger", "Love", 
    "Hate", "Hope", "Fear", "Joy", "Distress", "Pride", 
    "Shame", "Admiration", "Reproach"
]

# ONLY exact word-to-word matches (these are reliable ground truth from GoEmotions)
EXACT_MATCHES = {
    "neutral": "Neutral",
    "admiration": "Admiration",
    "anger": "Anger",
    "fear": "Fear",
    "gratitude": "Gratitude",
    "joy": "Joy",
    "love": "Love",
    "pride": "Pride",
    "relief": "Relief",
    "remorse": "Remorse",
}

def label_with_gpt4(client, text, deployment):
    """Ask GPT-4 to label text with OCC emotions."""
    
    prompt = f"""Classify this text into one or more OCC emotions. Return ONLY a JSON array.

OCC Emotions:
- Neutral, Happy-for, Resentment, Gloating, Pity, Satisfaction, Fears-Confirmed, Relief
- Disappointment, Gratification, Remorse, Gratitude, Anger, Love, Hate, Hope, Fear
- Joy, Distress, Pride, Shame, Admiration, Reproach

Text: "{text}"

JSON array:"""
    
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are an OCC emotion annotator. Return JSON arrays only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_completion_tokens=100
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON
        if content.startswith('['):
            emotions = json.loads(content)
        else:
            start = content.find('[')
            end = content.rfind(']') + 1
            emotions = json.loads(content[start:end]) if start != -1 else ["Neutral"]
        
        # Keep only valid OCC labels
        emotions = [e for e in emotions if e in OCC_LABELS]
        return emotions if emotions else ["Neutral"]
    
    except Exception as e:
        print(f"Error: {e}")
        return ["Neutral"]

def main():
    # Load data
    df = pd.read_csv('occ_sample.csv')
    
    # Initialize GPT-4
    print("Initializing GPT-4.1...")
    client = AzureOpenAI(
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    
    print(f"\nLabeling {len(df)} samples...")
    print("Logic: Exact GoEmotions matches (love→Love, etc.) UNION GPT-4 labels\n")
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text']
        goemotions = json.loads(row['goemotions_labels'])
        
        # Get exact word matches from GoEmotions
        exact_from_ge = set()
        for ge_label in goemotions:
            if ge_label in EXACT_MATCHES:
                exact_from_ge.add(EXACT_MATCHES[ge_label])
        
        # Get GPT-4 labels
        gpt4_labels = set(label_with_gpt4(client, text, deployment))
        
        # UNION both
        final_labels = exact_from_ge | gpt4_labels
        
        if not final_labels:
            final_labels = {"Neutral"}
        
        results.append({
            'exact_from_goemotions': sorted(exact_from_ge),
            'gpt4_labels': sorted(gpt4_labels),
            'final_labels': sorted(final_labels)
        })
        
        time.sleep(0.1)
    
    # Build output - CLEAN format
    output_df = pd.DataFrame({
        'id': df['id'],
        'text': df['text'],
        'occ_labels': [json.dumps(r['final_labels']) for r in results],
        'from_goemotions': [json.dumps(r['exact_from_goemotions']) for r in results],
        'from_gpt4': [json.dumps(r['gpt4_labels']) for r in results],
    })
    
    # Save
    output_df.to_csv('occ_ground_truth.csv', index=False)
    
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"\nSaved to: occ_ground_truth.csv")
    print(f"\nColumns:")
    print(f"  - id: sample ID")
    print(f"  - text: the sentence")
    print(f"  - occ_labels: final OCC labels (union of below two)")
    print(f"  - from_goemotions: exact word matches only (love→Love, joy→Joy, etc.)")
    print(f"  - from_gpt4: labels from GPT-4 reading the text")
    
    from collections import Counter
    counts = Counter(l for r in results for l in r['final_labels'])
    print(f"\nLabel distribution:")
    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {label:>18}: {count}")
    
    exact_count = sum(1 for r in results if r['exact_from_goemotions'])
    print(f"\nRows with exact GoEmotions matches: {exact_count}/{len(df)}")
    print(f"\n→ Now manually review occ_ground_truth.csv and fix any errors!")

if __name__ == "__main__":
    main()

