import nltk
from nltk.tokenize import sent_tokenize

import numpy as np

nltk.download('punkt')
nltk.download('punkt_tab')





import numpy as np
from nltk.tokenize import sent_tokenize

def analyze_sentence_lengths(text, baseline_texts=[]):
    """
    Analyze sentence lengths and compare to baseline samples.
    
    Args:
        text: Text to analyze (assess)
        baseline_texts: List of baseline texts for comparison
        
    Returns:
        Dictionary containing analysis results
    """
    # Tokenize sentences and calculate lengths
    sentences = sent_tokenize(text)
    lengths = [len(sent.split()) for sent in sentences]
    
    # Calculate statistics for the assessed text
    stats = {
        'sentences': sentences,
        'lengths': lengths,
        'mean': np.mean(lengths) if lengths else 0,
        'std_dev': np.std(lengths) if len(lengths) > 1 else 0,
        'comparison': {}
    }
    
    # Compare with each baseline
    for i, baseline in enumerate(baseline_texts, 1):
        baseline_sents = sent_tokenize(baseline)
        baseline_lengths = [len(sent.split()) for sent in baseline_sents]
        
        stats['comparison'][f'baseline{i}'] = {
            'mean': np.mean(baseline_lengths) if baseline_lengths else 0,
            'std_dev': np.std(baseline_lengths) if len(baseline_lengths) > 1 else 0,
            'mean_diff': (stats['mean'] - np.mean(baseline_lengths)) if baseline_lengths else 0,
            'std_dev_ratio': (stats['std_dev'] / np.std(baseline_lengths)) if len(baseline_lengths) > 1 else 1
        }
    
    # Flag potential AI patterns
    stats['flags'] = []
    if stats['std_dev'] < 2.0:  # Very uniform sentence lengths
        stats['flags'].append('FIA-SIG-S01: Unusually low sentence length variation')
    if stats['mean'] > 25:       # Very long sentences
        stats['flags'].append('FIA-SIG-S02: Excessive sentence length')
    
    return stats

# Example usage
assess = "Hello world! This is NLTK. It splits sentences."
baseline1 = "Hello world. This is NLTK. It splits sentences."
baseline2 = "Hello worlds. This is NLTK. It splits sentences."

results = analyze_sentence_lengths(assess, [baseline1, baseline2])

# Print formatted results
print(results)
print("Sentence Length Analysis")
print("="*50)
print(f"Assessed text: {assess}")
print(f"\nSentence lengths (words): {results['lengths']}")
print(f"Mean length: {results['mean']:.2f} words")
print(f"Standard deviation: {results['std_dev']:.2f}")

print("\nComparison with Baselines:")
for baseline, data in results['comparison'].items():
    print(f"\n{baseline}:")
    print(f"- Mean: {data['mean']:.2f} (Difference: {data['mean_diff']:+.2f})")
    print(f"- Std Dev: {data['std_dev']:.2f} (Ratio: {data['std_dev_ratio']:.2f}x)")

if results['flags']:
    print("\nFlags:")
    for flag in results['flags']:
        print(f"- ⚠️ {flag}")
else:
    print("\nNo significant anomalies detected")

print("="*50)