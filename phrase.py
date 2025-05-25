from collections import defaultdict, Counter
import re
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

def get_repeated_ngrams(text, min_n=3, max_n=5, min_count=2):
    """
    Find repeated n-grams (multi-word phrases) in text
    Args:
        text: input text to analyze
        min_n: minimum n-gram length (default 3)
        max_n: maximum n-gram length (default 5)
        min_count: minimum occurrence count to consider (default 2)
    Returns:
        Dictionary of {n_gram_length: {phrase: count}}
    """
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum()]  
    repeated_phrases = defaultdict(dict)
    for n in range(min_n, max_n + 1):
        phrase_counts = Counter(ngrams(words, n))
        for phrase, count in phrase_counts.items():
            if count >= min_count:
                phrase_text = ' '.join(phrase)
                repeated_phrases[n][phrase_text] = count
                
    return dict(repeated_phrases)

def compare_repeated_phrases(assess_text, baseline1_text, baseline2_text):
    """
    Compare repeated phrase patterns between assessment and baseline texts
    Returns analysis with metrics and recommendations
    """
    assess_phrases = get_repeated_ngrams(assess_text)
    baseline1_phrases = get_repeated_ngrams(baseline1_text)
    baseline2_phrases = get_repeated_ngrams(baseline2_text)
    
    def count_total_repeats(phrase_dict):
        return sum(len(phrases) for phrases in phrase_dict.values())
    
    assess_count = count_total_repeats(assess_phrases)
    baseline1_count = count_total_repeats(baseline1_phrases)
    baseline2_count = count_total_repeats(baseline2_phrases)
    baseline_avg = (baseline1_count + baseline2_count) / 2
    
    results = {
        'repeated_phrases': {
            'assessment': assess_phrases,
            'baseline1': baseline1_phrases,
            'baseline2': baseline2_phrases
        },
        'metrics': {
            'total_repeated_phrases': {
                'assessment': assess_count,
                'baseline1': baseline1_count,
                'baseline2': baseline2_count,
                'baseline_average': baseline_avg
            },
            'deviation_from_baseline': assess_count - baseline_avg,
            'deviation_percentage': ((assess_count - baseline_avg) / baseline_avg * 100) if baseline_avg != 0 else float('inf')
        }
    }
    
    recommendations = []
    if assess_count > baseline_avg * 1.5:
        recommendations.append(
            "Significant repetition detected (50%+ above baseline). "
            "This may indicate AI-generated or templated content. "
            "Suggested action: Revise to vary phrasing and reduce repetition."
        )
    elif assess_count > baseline_avg * 1.2:
        recommendations.append(
            "Moderate repetition detected (20%+ above baseline). "
            "Consider reviewing for overused phrases."
        )
    elif assess_count < baseline_avg * 0.8:
        recommendations.append(
            "Lower repetition than baselines. Good variation in phrasing."
        )
    else:
        recommendations.append(
            "Repetition level within normal range compared to baselines."
        )
    
    results['recommendations'] = recommendations
    
    assess_unique_phrases = set()
    for n in assess_phrases:
        for phrase in assess_phrases[n]:
            in_baseline1 = any(phrase in baseline1_phrases.get(m, {}) for m in baseline1_phrases)
            in_baseline2 = any(phrase in baseline2_phrases.get(m, {}) for m in baseline2_phrases)
            if not in_baseline1 and not in_baseline2:
                assess_unique_phrases.add(phrase)
    
    if assess_unique_phrases:
        results['unique_to_assessment'] = sorted(assess_unique_phrases)
    
    return results

# Example usage
if __name__ == "__main__":
    assess_text = (
        "The results indicate that the model performs well. "
        "The results indicate that accuracy is high. "
        "The results indicate that further testing is needed. "
        "In conclusion, the results indicate success."
    )
    
    baseline1_text = (
        "Our analysis shows good performance. "
        "The accuracy metrics are positive. "
        "Additional testing may be beneficial."
    )
    
    baseline2_text = (
        "We found the system works effectively. "
        "Effectiveness was measured across multiple dimensions. "
        "The system works with good reliability."
    )
    
    analysis = compare_repeated_phrases(assess_text, baseline1_text, baseline2_text)
    print(analysis)
    print("=== Repeated Phrases ===")
    print("Assessment:", analysis['repeated_phrases']['assessment'])
    print("Baseline 1:", analysis['repeated_phrases']['baseline1'])
    print("Baseline 2:", analysis['repeated_phrases']['baseline2'])
    
    print("\n=== Metrics ===")
    print("Total repeated phrases:", analysis['metrics']['total_repeated_phrases'])
    print("Deviation from baseline:", analysis['metrics']['deviation_from_baseline'])
    
    print("\n=== Recommendations ===")
    for rec in analysis['recommendations']:
        print("-", rec)
    
    if 'unique_to_assessment' in analysis:
        print("\n=== Unique Phrases in Assessment ===")
        print(analysis['unique_to_assessment'])