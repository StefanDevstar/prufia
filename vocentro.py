import math
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

def analyze_lexical_diversity(assess_text, baseline1=None, baseline2=None):
    """
    Analyze lexical diversity with comparison to baseline texts.
    
    Args:
        assess_text: Standard assessment text to analyze
        baseline1: First test baseline text
        baseline2: Second test baseline text
        
    Returns:
        Dictionary containing:
        - metrics: Entropy and TTR for assessment text
        - comparisons: Comparison with baselines
        - flags: Detected patterns
    """
    def calculate_diversity(text):
        """Calculate lexical diversity metrics for a text"""
        words = [word.lower() for word in word_tokenize(text) if word.isalpha()]
        if not words:
            return None
            
        word_count = len(words)
        unique_words = len(set(words))
        freq_dist = Counter(words)
        
        # Shannon entropy
        entropy = -sum(
            (count/word_count) * math.log2(count/word_count)
            for count in freq_dist.values()
        )
        
        # Type-Token Ratio (TTR)
        ttr = unique_words / word_count
        
        return {
            'word_count': word_count,
            'unique_words': unique_words,
            'entropy': entropy,
            'ttr': ttr,
            'top_words': freq_dist.most_common(3)
        }

    # Analyze assessment text
    assess = calculate_diversity(assess_text)
    if not assess:
        return {
            'error': 'Assessment text contains no valid words',
            'flags': ['FIA-ERROR: No valid words in assessment text']
        }
    
    # Analyze baselines
    base1 = calculate_diversity(baseline1) if baseline1 else None
    base2 = calculate_diversity(baseline2) if baseline2 else None
    
    # Prepare results
    results = {
        'assessment': {
            'shannon_entropy': assess['entropy'],
            'type_token_ratio': assess['ttr'],
            'vocabulary_size': assess['unique_words'],
            'word_count': assess['word_count']
        },
        'comparisons': {},
        'flags': []
    }
    
    # Create comparisons
    baselines = [('baseline1', base1), ('baseline2', base2)]
    for name, base in baselines:
        if base:
            results['comparisons'][name] = {
                'entropy': base['entropy'],
                'ttr': base['ttr'],
                'entropy_diff': assess['entropy'] - base['entropy'],
                'ttr_diff': assess['ttr'] - base['ttr'],
                'entropy_ratio': assess['entropy'] / base['entropy'] if base['entropy'] > 0 else 1,
                'ttr_ratio': assess['ttr'] / base['ttr'] if base['ttr'] > 0 else 1
            }
    
    # AI Detection Rules
    # 1. Absolute thresholds
    if assess['entropy'] < 2.0:
        results['flags'].append('FIA-SIG-V01: Low Shannon entropy (<2.0 bits) - may indicate repetitive vocabulary')
    if assess['ttr'] < 0.5:
        results['flags'].append('FIA-SIG-V02: Low Type-Token Ratio (<0.5) - suggests limited lexical diversity')
    
    # 2. Relative to baselines
    if base1 and base2:
        avg_base_entropy = (base1['entropy'] + base2['entropy']) / 2
        if assess['entropy'] < avg_base_entropy * 0.7:
            results['flags'].append('FIA-SIG-V03: Entropy significantly lower (30%+) than baseline average')
            
        avg_base_ttr = (base1['ttr'] + base2['ttr']) / 2
        if assess['ttr'] < avg_base_ttr * 0.7:
            results['flags'].append('FIA-SIG-V04: TTR significantly lower (30%+) than baseline average')
    
    # 3. Word repetition
    if assess['top_words'][0][1] > assess['word_count'] * 0.3:
        results['flags'].append(
            f'FIA-SIG-V05: High word repetition ("{assess["top_words"][0][0]}" appears {assess["top_words"][0][1]}/{assess["word_count"]} times)'
        )
    
    return results

def print_diversity_report(results):
    """Print a professional diversity analysis report"""
    print("\nLEXICAL DIVERSITY ANALYSIS REPORT")
    print("=" * 70)
    print("\nASSESSMENT TEXT RESULTS:")
    print(f"- Vocabulary Size: {results['assessment']['vocabulary_size']} unique words")
    print(f"- Total Words: {results['assessment']['word_count']}")
    print(f"- Shannon Entropy: {results['assessment']['shannon_entropy']:.3f} bits")
    print(f"- Type-Token Ratio: {results['assessment']['type_token_ratio']:.3f}")
    
    if results['comparisons']:
        print("\nBASELINE COMPARISONS:")
        for name, comp in results['comparisons'].items():
            print(f"\n{name.upper()}:")
            print(f"  Shannon Entropy: {comp['entropy']:.3f} bits")
            print(f"    → Difference: {comp['entropy_diff']:+.3f} ({comp['entropy_ratio']:.2f}x)")
            print(f"  Type-Token Ratio: {comp['ttr']:.3f}")
            print(f"    → Difference: {comp['ttr_diff']:+.3f} ({comp['ttr_ratio']:.2f}x)")
    
    if results['flags']:
        print("\nDETECTED PATTERNS:")
        for flag in results['flags']:
            print(f"  ⚠️ {flag}")
    else:
        print("\nNo significant AI patterns detected")
    
    print("\n" + "=" * 70)
    print("Note: Lower entropy/TTR may indicate AI-generated content")
    print("=" * 70)

# Example usage
assess = "The quick brown fox jumps over the lazy dog. The dog was not amused."
test1 = "This is a sample baseline text with moderate vocabulary diversity."
test2 = "Another example text containing various words of different lengths."

analysis = analyze_lexical_diversity(assess, test1, test2)
print(analysis)
print_diversity_report(analysis)