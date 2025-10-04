#!/usr/bin/env python3
"""
Re-label the dataset with proper OCC emotions.
- Keep GoEmotions labels where they map 1:1 to OCC
- Add OCC-specific labels (Resentment, Pity, Happy-for, etc.) based on text analysis
- Output for manual review
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

# STRICT word-to-word mappings ONLY (same word = reliable ground truth)
# Do NOT include mappings like sadness→Distress (different words = unreliable)
EXACT_WORD_MAPPINGS = {
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
    # Note: We exclude these because they're different words:
    # sadness→Distress, grief→Distress, disappointment→Disappointment
    # disapproval→Reproach, optimism→Hope, embarrassment→Shame, disgust→Hate
}

def get_occ_labels_from_gpt4(client, text, deployment):
    """Use GPT-4 to intelligently label with full OCC taxonomy."""
    
    occ_definitions = """
Neutral: No specific emotion
Happy-for: Joy about someone else's good fortune (e.g., "Congrats on your promotion!")
Resentment: Displeasure at someone else's fortune (e.g., "He got promoted after doing nothing")
Gloating: Joy at someone else's misfortune (e.g., "Serves him right for bragging")
Pity: Sadness for someone else's misfortune (e.g., "I feel bad for him losing his job")
Satisfaction: Pleased with outcome of own action (e.g., "Finally finished that project")
Fears-Confirmed: Feared outcome occurred (e.g., "I knew this would happen")
Relief: Disconfirmation of feared outcome (e.g., "Thank god it wasn't serious")
Disappointment: Displeased with outcome
Gratification: Pleased with own action/achievement (e.g., "I'm proud I built this")
Remorse: Regret for one's own action (e.g., "I shouldn't have said that")
Gratitude: Thankful for someone's action
Anger: Strong displeasure/hostility
Love: Affection, liking
Hate: Strong aversion
Hope: Desire for positive outcome
Fear: Concern about potential negative outcome
Joy: Happiness
Distress: Sadness, upset
Pride: Pleased with own attribute (e.g., "I'm proud to be American")
Shame: Ashamed of own action/attribute
Admiration: Approval of someone else's action/attribute
Reproach: Disapproval of someone else's action
"""
    
    prompt = f"""You are an expert emotion annotator using the OCC appraisal model. Classify this text into one or more of these emotions:

{occ_definitions}

Text: "{text}"

Return ONLY a JSON array of applicable emotions. Consider:
- Is this about SELF or OTHER?
- Is this about FORTUNE (outcome) or ACTION/ATTRIBUTE?
- Is the valence POSITIVE or NEGATIVE?

Examples:
"Congrats on your new job!" → ["Happy-for"]
"He got promoted after slacking off" → ["Resentment"]
"I knew the test would be hard" → ["Fears-Confirmed"]
"Thank god it wasn't broken" → ["Relief"]
"I feel sorry for him" → ["Pity"]
"I'm glad I finished this" → ["Satisfaction", "Gratification"]

JSON array:"""
    
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert OCC emotion annotator. Always return valid JSON arrays of emotion labels."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_completion_tokens=150
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON
        if content.startswith('[') and content.endswith(']'):
            predicted_emotions = json.loads(content)
        else:
            start = content.find('[')
            end = content.rfind(']') + 1
            if start != -1 and end > start:
                predicted_emotions = json.loads(content[start:end])
            else:
                predicted_emotions = ["Neutral"]
        
        # Validate labels
        predicted_emotions = [e for e in predicted_emotions if e in OCC_LABELS]
        if not predicted_emotions:
            predicted_emotions = ["Neutral"]
            
        return predicted_emotions
    
    except Exception as e:
        print(f"Error: {e}")
        return ["Neutral"]

def main():
    # Load original sample
    df = pd.read_csv('occ_sample.csv')
    
    # Initialize Azure OpenAI
    print("Initializing GPT-4.1 for intelligent labeling...")
    client = AzureOpenAI(
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    
    # Create new labels
    print(f"Re-labeling {len(df)} samples with proper OCC taxonomy...")
    
    new_labels_list = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text']
        goemotions = json.loads(row['goemotions_labels'])
        
        # Step 1: Keep ONLY exact word-to-word mappings from GoEmotions (reliable ground truth)
        exact_labels = set()
        for ge_label in goemotions:
            if ge_label in EXACT_WORD_MAPPINGS:
                exact_labels.add(EXACT_WORD_MAPPINGS[ge_label])
        
        # Step 2: Ask GPT-4 to label the text with ALL OCC emotions
        gpt4_labels = set(get_occ_labels_from_gpt4(client, text, deployment))
        
        # Step 3: Take UNION of both (exact GoEmotions + full GPT-4 judgment)
        final_labels = exact_labels | gpt4_labels
        
        # If empty, use Neutral
        if not final_labels:
            final_labels = {"Neutral"}
        
        new_labels_list.append(sorted(final_labels))
        
        time.sleep(0.1)  # Rate limiting
    
    # Rebuild with source tracking
    output_rows = []
    
    for idx, row in df.iterrows():
        goemotions = json.loads(row['goemotions_labels'])
        
        # Track exact word matches
        exact_labels = set()
        for ge_label in goemotions:
            if ge_label in EXACT_WORD_MAPPINGS:
                exact_labels.add(EXACT_WORD_MAPPINGS[ge_label])
        
        output_row = {
            'id': row['id'],
            'text': row['text'],
        }
        
        # Add 0/1 columns
        for label in OCC_LABELS:
            output_row[label] = 1 if label in new_labels_list[idx] else 0
        
        # Add metadata
        output_row['relabeled_occ_emotions'] = json.dumps(new_labels_list[idx])
        output_row['from_goemotions_exact'] = json.dumps(sorted(exact_labels))
        output_row['goemotions_labels'] = row['goemotions_labels']
        output_row['original_mapped_labels'] = row['mapped_occ_labels']
        
        output_rows.append(output_row)
    
    output_df = pd.DataFrame(output_rows)
    
    # Add GPT-4 and BART predictions for comparison
    gpt4_df = pd.read_csv('gpt4_results.csv')
    output_df['gpt4_predicted'] = gpt4_df['gpt4_predicted_emotions']
    
    # Save
    output_df.to_csv('occ_relabeled_for_review.csv', index=False)
    print(f"\nSaved to: occ_relabeled_for_review.csv")
    
    # Summary
    print("\n" + "="*60)
    print("RELABELING SUMMARY")
    print("="*60)
    
    from collections import Counter
    label_counts = Counter(l for labels in new_labels_list for l in labels)
    
    print("\nLabeling approach:")
    print("  1. Keep EXACT word matches from GoEmotions (love→Love, joy→Joy, etc.)")
    print("  2. GPT-4 labels everything else from text")
    print("  3. Union both sources")
    
    print("\nNew label distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:>18}: {count}")
    
    # Count exact matches
    exact_match_count = sum(1 for i in range(len(df)) 
                           if json.loads(output_df.iloc[i]['from_goemotions_exact']))
    
    print(f"\nRows with exact GoEmotions matches: {exact_match_count}/{len(df)}")
    print(f"\nColumns in output:")
    print(f"  - 23 OCC emotion columns (0/1)")
    print(f"  - relabeled_occ_emotions: final labels (easy to read)")
    print(f"  - from_goemotions_exact: which labels came from exact word matching")
    print(f"  - gpt4_predicted: what GPT-4 predicted during evaluation (for comparison)")
    print(f"  - goemotions_labels: original GoEmotions labels")
    print("\nPlease manually review and correct any errors!")

if __name__ == "__main__":
    main()

