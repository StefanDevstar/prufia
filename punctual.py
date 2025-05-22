import string
from collections import defaultdict
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize

def analyze_punctuation_patterns(assess_text, baseline1=None, baseline2=None):
    """
    Analyze punctuation distribution and spacing with baseline comparisons.
    
    Args:
        assess_text: Standard assessment text to analyze
        baseline1: First test baseline text
        baseline2: Second test baseline text
        
    Returns:
        Dictionary containing:
        - counts: Punctuation frequencies
        - spacing: Average words between punctuation
        - comparisons: Baseline comparisons
        - flags: Detected patterns
    """
    def calculate_punctuation_metrics(text):
        """Calculate punctuation metrics for a text"""
        # Count punctuation marks
        punct_counts = {p: text.count(p) for p in string.punctuation if text.count(p) > 0}
        
        # Calculate spacing between punctuation
        words = word_tokenize(text)
        punct_positions = []
        word_gaps = []
        prev_punct_pos = -1
        
        for i, word in enumerate(words):
            if any(c in string.punctuation for c in word):
                punct_positions.append(i)
                if prev_punct_pos != -1:
                    word_gaps.append(i - prev_punct_pos - 1)
                prev_punct_pos = i
        
        avg_gap = np.mean(word_gaps) if word_gaps else 0
        gap_std = np.std(word_gaps) if len(word_gaps) > 1 else 0
        
        # Sentence length punctuation analysis
        sentences = sent_tokenize(text)
        sentence_ends = [sent[-1] if sent else '' for sent in sentences]
        end_punct_dist = defaultdict(int)
        for p in sentence_ends:
            if p in string.punctuation:
                end_punct_dist[p] += 1
        
        return {
            'counts': punct_counts,
            'avg_gap': avg_gap,
            'gap_std': gap_std,
            'end_punct_dist': dict(end_punct_dist),
            'total_punct': sum(punct_counts.values()),
            'word_count': len([w for w in words if w not in string.punctuation])
        }

    # Analyze assessment text
    assess = calculate_punctuation_metrics(assess_text)
    
    # Analyze baselines
    base1 = calculate_punctuation_metrics(baseline1) if baseline1 else None
    base2 = calculate_punctuation_metrics(baseline2) if baseline2 else None
    
    # Prepare results
    results = {
        'assessment': {
            'punctuation_counts': assess['counts'],
            'avg_words_between_punct': assess['avg_gap'],
            'punct_spacing_consistency': assess['gap_std'],
            'sentence_end_distribution': assess['end_punct_dist'],
            'punct_per_word': assess['total_punct'] / assess['word_count'] if assess['word_count'] > 0 else 0
        },
        'comparisons': {},
        'flags': []
    }
    
    # Create comparisons
    baselines = [('baseline1', base1), ('baseline2', base2)]
    for name, base in baselines:
        if base:
            results['comparisons'][name] = {
                'avg_gap': base['avg_gap'],
                'gap_std': base['gap_std'],
                'punct_per_word': base['total_punct'] / base['word_count'] if base['word_count'] > 0 else 0,
                'gap_diff': assess['avg_gap'] - base['avg_gap'],
                'consistency_ratio': assess['gap_std'] / base['gap_std'] if base['gap_std'] > 0 else 1
            }
    
    # AI Detection Rules
    # 1. Punctuation spacing consistency (low std dev suggests AI)
    if assess['gap_std'] < 1.5:
        results['flags'].append('FIA-SIG-P01: Unusually consistent punctuation spacing (std dev < 1.5)')
    
    # 2. Overuse of certain punctuation
    if assess['counts'].get('.', 0) / max(1, assess['total_punct']) > 0.5:
        results['flags'].append('FIA-SIG-P02: Overuse of periods (>50% of all punctuation)')
    
    # 3. Compare to baselines
    if base1 and base2:
        avg_base_std = (base1['gap_std'] + base2['gap_std']) / 2
        if assess['gap_std'] < avg_base_std * 0.6:
            results['flags'].append('FIA-SIG-P03: Punctuation spacing significantly more consistent than baselines')
            
        avg_base_punct_rate = ((base1['total_punct'] / base1['word_count']) + 
                             (base2['total_punct'] / base2['word_count'])) / 2
        assess_punct_rate = assess['total_punct'] / assess['word_count']
        if abs(assess_punct_rate - avg_base_punct_rate) > 0.2:
            results['flags'].append('FIA-SIG-P04: Punctuation rate differs significantly from baselines')
    
    # 4. Sentence end diversity
    if len(assess['end_punct_dist']) < 2 and len(sent_tokenize(assess_text)) > 3:
        results['flags'].append('FIA-SIG-P05: Limited sentence ending variety')
    
    return results

def print_punctuation_report(results):
    """Print a professional punctuation analysis report"""
    print("\nPUNCTUATION PATTERN ANALYSIS REPORT")
    print("=" * 80)
    print("\nASSESSMENT TEXT RESULTS:")
    print(f"- Punctuation counts: {results['assessment']['punctuation_counts']}")
    print(f"- Avg words between punctuation: {results['assessment']['avg_words_between_punct']:.1f}")
    print(f"- Punctuation spacing consistency (std dev): {results['assessment']['punct_spacing_consistency']:.2f}")
    print(f"- Punctuation per word: {results['assessment']['punct_per_word']:.3f}")
    print(f"- Sentence end distribution: {results['assessment']['sentence_end_distribution']}")
    
    if results['comparisons']:
        print("\nBASELINE COMPARISONS:")
        for name, comp in results['comparisons'].items():
            print(f"\n{name.upper()}:")
            print(f"  Avg words between punct: {comp['avg_gap']:.1f} (Difference: {comp['gap_diff']:+.1f})")
            print(f"  Spacing consistency: {comp['gap_std']:.2f} (Ratio: {comp['consistency_ratio']:.2f}x)")
            print(f"  Punctuation rate: {comp['punct_per_word']:.3f} per word")
    
    if results['flags']:
        print("\nDETECTED PATTERNS:")
        for flag in results['flags']:
            print(f"  ⚠️ {flag}")
    else:
        print("\nNo significant AI patterns detected")
    
    print("\n" + "=" * 80)
    print("Note: Highly consistent punctuation patterns may indicate AI-generated content")
    print("=" * 80)

# Example usage
assess = "The quick brown fox, jumps over the lazy dog. The dog wasn't amused... Was he?"
test1 = "This is a sample text, with normal punctuation. It varies! Some sentences are short; others longer."
test2 = "Another example... with different pacing! Questions? Commas, and more."

analysis = analyze_punctuation_patterns(assess, test1, test2)
print(analysis)
print_punctuation_report(analysis)