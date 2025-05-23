from difflib import SequenceMatcher
from collections import Counter, defaultdict
import math
import spacy
import string
import nltk
import re
from nltk.util import ngrams
from textstat import textstat
from nltk.tokenize import sent_tokenize, word_tokenize
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import textstat
from typing import Dict, Any


nltk.download('punkt')
nltk.download('punkt_tab')

@lru_cache(maxsize=2)
def load_spacy_model(model_name):
    """Load spaCy model with caching to avoid reloading"""
    return spacy.load(model_name)

class TextAnalyzer:
    def __init__(self):
        # Initialize models (will load on first use due to caching)
        self.nlp_sm = None  # Small model
        self.nlp_lg = None  # Large model
    
    def get_model(self, model_type='sm'):
        """Get the appropriate model instance"""
        if model_type == 'sm':
            if self.nlp_sm is None:
                self.nlp_sm = load_spacy_model('en_core_web_sm')
            return self.nlp_sm
        elif model_type == 'lg':
            if self.nlp_lg is None:
                self.nlp_lg = load_spacy_model('en_core_web_lg')
            return self.nlp_lg
        raise ValueError("Invalid model type. Use 'sm' or 'lg'")

# Create a global analyzer instance
analyzer = TextAnalyzer()


# text = "Hello world! This is NLTK. It splits sentences."
# output = ['Hello world!', 'This is NLTK.', 'It splits sentences.']
def get_sentences(text):
    return sent_tokenize(text)
def Sentence_Length_Variation(assess, baseline1, baseline2):
    """
    Analyze sentence length variation and estimate if text appears human-written or AI-generated
    by comparing against two baseline samples.
    
    Args:
        assess: Text to analyze
        baseline1: First baseline text for comparison
        baseline2: Second baseline text for comparison
        
    Returns:
        Dictionary containing detailed analysis and estimation
    """
    # Analyze sentence lengths
    results = analyze_sentence_lengths(assess, [baseline1, baseline2])
    
    # Calculate additional comparison metrics
    baseline1_mean = results['comparison']['baseline1']['mean']
    baseline2_mean = results['comparison']['baseline2']['mean']
    baseline_mean = (baseline1_mean + baseline2_mean) / 2
    
    baseline1_std = results['comparison']['baseline1']['std_dev']
    baseline2_std = results['comparison']['baseline2']['std_dev']
    baseline_std = (baseline1_std + baseline2_std) / 2
    
    # Calculate deviation scores (how far assess text is from baselines)
    mean_deviation = abs(results['mean'] - baseline_mean) / baseline_mean if baseline_mean != 0 else 0
    std_deviation = abs(results['std_dev'] - baseline_std) / baseline_std if baseline_std != 0 else 0
    
    # Estimate likelihood of AI generation
    ai_score = 0
    
    # Penalize for low variation (AI tends to have more uniform lengths)
    if results['std_dev'] < baseline_std * 0.6:
        ai_score += 30 * (1 - (results['std_dev'] / (baseline_std * 0.6)))
    
    # Penalize for mean length being too different from baselines
    if mean_deviation > 0.5:  # More than 50% different from baseline mean
        ai_score += 20 * min(mean_deviation, 2)  # Cap at 40 points
        
    # Add scoring based on flags
    for flag in results['flags']:
        if 'FIA-SIG-S01' in flag:  # Low variation flag
            ai_score += 25
        if 'FIA-SIG-S02' in flag:  # Long sentences flag
            ai_score += 15
    
    # Cap score at 100
    ai_score = min(100, ai_score)
    
    # Determine assessment
    if ai_score >= 70:
        assessment = "Likely AI-generated"
    elif ai_score >= 40:
        assessment = "Possibly AI-generated"
    else:
        assessment = "Likely human-written"
    
    # Add new metrics to results
    results['assessment'] = {
        'mean_deviation': mean_deviation,
        'std_deviation': std_deviation,
        'ai_score': int(ai_score),
        'assessment': assessment,
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std
    }
    
    return results
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
# def Sentence_Length_Variation(assess, baseline1, baseline2):
#     results = analyze_sentence_lengths(assess, [baseline1, baseline2])
#     return results

# text1 = "hello hello hello"  # Low entropy (repetitive)
# text2 = "the quick brown fox"  # Higher entropy (all unique)
# print(word_entropy(text1))  # Output: 0.0
# print(word_entropy(text2))  # Output: 2.0
def word_entropy(text):
    words = text.split()
    freq = Counter(words)
    total = sum(freq.values())
    return -sum((count/total) * math.log2(count/total) for count in freq.values())
# def analyze_lexical_diversity(assess_text, baseline1=None, baseline2=None):
#     """
#     Analyze lexical diversity with comparison to baseline texts.
    
#     Args:
#         assess_text: Standard assessment text to analyze
#         baseline1: First test baseline text
#         baseline2: Second test baseline text
        
#     Returns:
#         Dictionary containing:
#         - metrics: Entropy and TTR for assessment text
#         - comparisons: Comparison with baselines
#         - flags: Detected patterns
#     """
#     def calculate_diversity(text):
#         """Calculate lexical diversity metrics for a text"""
#         words = [word.lower() for word in word_tokenize(text) if word.isalpha()]
#         if not words:
#             return None
            
#         word_count = len(words)
#         unique_words = len(set(words))
#         freq_dist = Counter(words)
        
#         # Shannon entropy
#         entropy = -sum(
#             (count/word_count) * math.log2(count/word_count)
#             for count in freq_dist.values()
#         )
        
#         # Type-Token Ratio (TTR)
#         ttr = unique_words / word_count
        
#         return {
#             'word_count': word_count,
#             'unique_words': unique_words,
#             'entropy': entropy,
#             'ttr': ttr,
#             'top_words': freq_dist.most_common(3)
#         }

#     # Analyze assessment text
#     assess = calculate_diversity(assess_text)
#     if not assess:
#         return {
#             'error': 'Assessment text contains no valid words',
#             'flags': ['FIA-ERROR: No valid words in assessment text']
#         }
    
#     # Analyze baselines
#     base1 = calculate_diversity(baseline1) if baseline1 else None
#     base2 = calculate_diversity(baseline2) if baseline2 else None
    
#     # Prepare results
#     results = {
#         'assessment': {
#             'shannon_entropy': assess['entropy'],
#             'type_token_ratio': assess['ttr'],
#             'vocabulary_size': assess['unique_words'],
#             'word_count': assess['word_count']
#         },
#         'comparisons': {},
#         'flags': []
#     }
    
#     # Create comparisons
#     baselines = [('baseline1', base1), ('baseline2', base2)]
#     for name, base in baselines:
#         if base:
#             results['comparisons'][name] = {
#                 'entropy': base['entropy'],
#                 'ttr': base['ttr'],
#                 'entropy_diff': assess['entropy'] - base['entropy'],
#                 'ttr_diff': assess['ttr'] - base['ttr'],
#                 'entropy_ratio': assess['entropy'] / base['entropy'] if base['entropy'] > 0 else 1,
#                 'ttr_ratio': assess['ttr'] / base['ttr'] if base['ttr'] > 0 else 1
#             }
    
#     # AI Detection Rules
#     # 1. Absolute thresholds
#     if assess['entropy'] < 2.0:
#         results['flags'].append('FIA-SIG-V01: Low Shannon entropy (<2.0 bits) - may indicate repetitive vocabulary')
#     if assess['ttr'] < 0.5:
#         results['flags'].append('FIA-SIG-V02: Low Type-Token Ratio (<0.5) - suggests limited lexical diversity')
    
#     # 2. Relative to baselines
#     if base1 and base2:
#         avg_base_entropy = (base1['entropy'] + base2['entropy']) / 2
#         if assess['entropy'] < avg_base_entropy * 0.7:
#             results['flags'].append('FIA-SIG-V03: Entropy significantly lower (30%+) than baseline average')
            
#         avg_base_ttr = (base1['ttr'] + base2['ttr']) / 2
#         if assess['ttr'] < avg_base_ttr * 0.7:
#             results['flags'].append('FIA-SIG-V04: TTR significantly lower (30%+) than baseline average')
    
#     # 3. Word repetition
#     if assess['top_words'][0][1] > assess['word_count'] * 0.3:
#         results['flags'].append(
#             f'FIA-SIG-V05: High word repetition ("{assess["top_words"][0][0]}" appears {assess["top_words"][0][1]}/{assess["word_count"]} times)'
#         )
    
#     return results
def analyze_lexical_diversity(baseline1, baseline2, assess_text=None):
    """
    Analyze vocabulary entropy and estimate human reference from baselines.
    If assess_text is provided, compares it against estimated human profile.
    
    Args:
        baseline1: First baseline text (required)
        baseline2: Second baseline text (required)
        assess_text: Optional text to analyze against estimated human profile
        
    Returns:
        Dictionary containing analysis results
    """
    def calculate_entropy(text):
        """Calculate vocabulary metrics for a text."""
        if not text or not isinstance(text, str):
            return None
            
        try:
            words = [word.lower() for word in word_tokenize(text) if word.isalpha()]
            if not words:
                return None
                
            word_count = len(words)
            unique_words = len(set(words))
            freq_dist = Counter(words)
            
            # Shannon entropy with zero division protection
            entropy = 0.0
            for count in freq_dist.values():
                probability = count / word_count
                if probability > 0:  # Avoid log(0)
                    entropy -= probability * math.log2(probability)
            
            # Type-Token Ratio (TTR) with zero division protection
            ttr = unique_words / word_count if word_count > 0 else 0
            
            return {
                'word_count': word_count,
                'unique_words': unique_words,
                'entropy': entropy,
                'ttr': ttr,
                'top_words': freq_dist.most_common(3),
                'text': text if len(text) < 100 else text[:100] + "..."
            }
        except Exception as e:
            print(f"Error calculating entropy: {e}")
            return None

    # --- ESTIMATE HUMAN REFERENCE FROM BASELINES ---
    baseline_metrics = []
    for text in [baseline1, baseline2]:
        metrics = calculate_entropy(text)
        if metrics:
            baseline_metrics.append(metrics)
    
    if len(baseline_metrics) < 2:
        return {'error': 'Need at least 2 valid baselines for estimation'}
    
    # Select the most human-like baseline (highest entropy + TTR)
    baseline_metrics.sort(key=lambda x: (x['entropy'], x['ttr']), reverse=True)
    estimated_reference = baseline_metrics[0]
    
    # --- ANALYZE ASSESS TEXT (IF PROVIDED) ---
    assessment = None
    if assess_text:
        assess_data = calculate_entropy(assess_text)
        if assess_data:
            # Safe comparison with zero division protection
            ref_entropy = estimated_reference['entropy'] or 0.0001  # Avoid division by zero
            ref_ttr = estimated_reference['ttr'] or 0.0001
            
            entropy_ratio = (assess_data['entropy'] or 0) / ref_entropy
            ttr_ratio = (assess_data['ttr'] or 0) / ref_ttr
            
            # Calculate AI probability score (0-100) with bounds checking
            try:
                ai_score = max(0, min(100, 100 - (entropy_ratio * 50 + ttr_ratio * 50) * 100))
            except:
                ai_score = 50  # Default if calculation fails
            
            assessment = {
                'metrics': assess_data,
                'comparison': {
                    'entropy_vs_estimated': f"{entropy_ratio:.1%}" if ref_entropy else "N/A",
                    'ttr_vs_estimated': f"{ttr_ratio:.1%}" if ref_ttr else "N/A"
                },
                'ai_score': ai_score,
                'assessment': "Likely AI-generated" if ai_score >= 70 else 
                            "Possibly AI-generated" if ai_score >= 40 else 
                            "Likely human-written" if ai_score >= 0 else
                            "Invalid score"
            }
    
    # --- FINAL RESULTS ---
    result = {
        'estimated_human_reference': {
            'metrics': {
                'shannon_entropy': estimated_reference.get('entropy', 0),
                'type_token_ratio': estimated_reference.get('ttr', 0),
                'vocabulary_size': estimated_reference.get('unique_words', 0)
            },
            'source': f"Estimated from baseline{'1' if baseline_metrics[0]['text'] == baseline1 else '2'}",
            'text_sample': estimated_reference.get('text', '')
        },
        'assess_text_analysis': assessment if assess_text else "No assess_text provided",
        'baseline_comparison': {
            'baseline1': calculate_entropy(baseline1),
            'baseline2': calculate_entropy(baseline2)
        },
        'scoring_notes': {
            'entropy_interpretation': "Higher = more human-like",
            'ttr_interpretation': "Closer to 1 = more human-like",
            'ai_thresholds': ">70 = likely AI, 40-70 = possible AI, <40 = likely human"
        }
    }
    
    return result



# text = "The book was read by the student. The student enjoyed it."
# print(passive_voice_ratio(text))  # Output: 0.5 (1 passive out of 2 sentences)
# def passive_voice_ratio(text):
#     nlp = analyzer.get_model('sm')
#     doc = nlp(text)
#     passive_sentences = [sent for sent in doc.sents if any(tok.dep_ == "nsubjpass" for tok in sent)]
#     return len(passive_sentences) / len(list(doc.sents))

# def passive_voice_ratio(text):
#     """Calculate the ratio of passive voice sentences to total sentences"""
#     nlp = analyzer.get_model('sm')
#     doc = nlp(text)
#     sentences = list(doc.sents)
    
#     if not sentences:
#         return 0.0
    
#     passive_count = sum(1 for sent in sentences if any(tok.dep_ == "nsubjpass" for tok in sent))
#     return passive_count / len(sentences)

# def passive_voice_analysis(assess_text, baseline1_text, baseline2_text):
#     """Returns consistent structure with punctuation analysis"""
#     # Calculate ratios
#     assess_ratio = passive_voice_ratio(assess_text)
#     baseline1_ratio = passive_voice_ratio(baseline1_text) if baseline1_text else None
#     baseline2_ratio = passive_voice_ratio(baseline2_text) if baseline2_text else None
    
#     # Main results structure
#     results = {
#         'assessment': {
#             'passive_ratio': assess_ratio,
#             'total_sentences': len(list(analyzer.get_model('sm')(assess_text).sents))
#         },
#         'comparisons': {},
#         'flags': []
#     }
    
#     # Add baseline comparisons if they exist
#     baselines = [('baseline1', baseline1_ratio), ('baseline2', baseline2_ratio)]
#     for name, ratio in baselines:
#         if ratio is not None:
#             results['comparisons'][name] = {
#                 'passive_ratio': ratio,
#                 'ratio_difference': assess_ratio - ratio
#             }
    
#     # Detection rules
#     if baseline1_ratio and baseline2_ratio:
#         baseline_avg = (baseline1_ratio + baseline2_ratio) / 2
#         if assess_ratio > baseline_avg * 1.5:
#             results['flags'].append('FIA-PV01: High passive voice usage')
#         elif assess_ratio > baseline_avg * 1.2:
#             results['flags'].append('FIA-PV02: Moderate passive voice usage')
    
#     return results
def passive_voice_analysis(baseline1_text, baseline2_text, assess_text=None):
    """
    Analyze passive voice usage by estimating human reference from baselines.
    If assess_text is provided, compares it against estimated human profile.
    
    Args:
        baseline1_text: First baseline text (required)
        baseline2_text: Second baseline text (required)
        assess_text: Optional text to analyze against estimated human profile
        
    Returns:
        Dictionary containing:
        - estimated_human_reference: Passive voice profile from baselines
        - assess_text_analysis: Comparison results (if assess_text provided)
        - baseline_comparison: Raw baseline metrics
        - flags: Detected patterns
    """
    def calculate_passive_ratio(text):
        """Calculate passive voice ratio for a text"""
        if not text:
            return None
            
        nlp = analyzer.get_model('sm')
        doc = nlp(text)
        sentences = list(doc.sents)
        
        if not sentences:
            return None
            
        passive_count = sum(1 for sent in sentences if any(tok.dep_ == "nsubjpass" for tok in sent))
        return {
            'ratio': passive_count / len(sentences),
            'total_sentences': len(sentences),
            'passive_count': passive_count,
            'text_sample': text if len(text) < 100 else text[:100] + "..."
        }

    # --- ESTIMATE HUMAN REFERENCE FROM BASELINES ---
    baseline_metrics = []
    for text in [baseline1_text, baseline2_text]:
        metrics = calculate_passive_ratio(text)
        if metrics:
            baseline_metrics.append(metrics)
    
    if len(baseline_metrics) < 2:
        return {'error': 'Need at least 2 valid baselines for estimation'}
    
    # Create composite human reference
    estimated_reference = {
        'passive_ratio': np.mean([b['ratio'] for b in baseline_metrics]),
        'total_sentences': int(np.mean([b['total_sentences'] for b in baseline_metrics])),
        'text_samples': [b['text_sample'] for b in baseline_metrics]
    }
    
    # --- ANALYZE ASSESS TEXT (IF PROVIDED) ---
    assessment = None
    flags = []
    if assess_text:
        assess_data = calculate_passive_ratio(assess_text)
        if assess_data:
            # Calculate deviation
            ratio_diff = assess_data['ratio'] - estimated_reference['passive_ratio']
            ratio_deviation = ratio_diff / estimated_reference['passive_ratio'] if estimated_reference['passive_ratio'] > 0 else 0
            
            # AI Probability Score (0-100)
            ai_score = 0
            
            # 1. High passive voice usage
            if ratio_diff > 0.15:  # Absolute difference threshold
                ai_score += min(40, 40 * (ratio_diff / 0.3))  # Cap at +40 for 30%+ difference
                flags.append(f'FIA-PV01: High passive voice (+{ratio_diff:.2f} vs reference)')
            
            # 2. Unusually low passive voice
            elif ratio_diff < -0.1:  # Significantly lower than human baseline
                ai_score += min(20, 20 * abs(ratio_diff / 0.2))
                flags.append(f'FIA-PV02: Unusually low passive voice ({ratio_diff:.2f} vs reference)')
            
            # 3. Extreme values
            if assess_data['ratio'] > 0.5:  # Over 50% passive
                ai_score += 30
                flags.append('FIA-PV03: Excessive passive voice (>50%)')
            elif assess_data['ratio'] < 0.05:  # Under 5% passive
                ai_score += 15
                flags.append('FIA-PV04: Abnormally low passive voice (<5%)')
            
            ai_score = min(100, ai_score)
            
            assessment = {
                'passive_ratio': assess_data['ratio'],
                'total_sentences': assess_data['total_sentences'],
                'comparison': {
                    'ratio_vs_reference': f"{assess_data['ratio']/estimated_reference['passive_ratio']:.1%}",
                    'absolute_difference': f"{ratio_diff:.3f}"
                },
                'ai_score': int(ai_score),
                'assessment': "Likely AI-generated" if ai_score >= 70 else
                            "Possibly AI-generated" if ai_score >= 40 else
                            "Likely human-written",
                'flags': flags
            }
    
    # --- FINAL RESULTS ---
    return {
        'estimated_human_reference': {
            'passive_ratio': estimated_reference['passive_ratio'],
            'total_sentences': estimated_reference['total_sentences'],
            'source': 'Composite of baseline1 and baseline2',
            'text_samples': estimated_reference['text_samples']
        },
        'assess_text_analysis': assessment if assess_text else "No assess_text provided",
        'baseline_comparison': {
            'baseline1': calculate_passive_ratio(baseline1_text),
            'baseline2': calculate_passive_ratio(baseline2_text)
        },
        'scoring_notes': {
            'ai_thresholds': ">70=AI, 40-70=Possible AI, <40=Human",
            'human_reference_range': f"Expected passive ratio: {estimated_reference['passive_ratio']:.2f}-{estimated_reference['passive_ratio']*1.3:.2f}"
        }
    }
# text = "Hello, world! How's it going?"
# punctuation_counts(text)
# Output:{',': 1, '!': 1, "'": 1, '?': 1}
# def punctuation_counts(text):
#     return {p: text.count(p) for p in string.punctuation if text.count(p) > 0}
# def analyze_punctuation_patterns(assess_text, baseline1=None, baseline2=None):
#     """
#     Analyze punctuation distribution and spacing with baseline comparisons.
    
#     Args:
#         assess_text: Standard assessment text to analyze
#         baseline1: First test baseline text
#         baseline2: Second test baseline text
        
#     Returns:
#         Dictionary containing:
#         - counts: Punctuation frequencies
#         - spacing: Average words between punctuation
#         - comparisons: Baseline comparisons
#         - flags: Detected patterns
#     """
#     def calculate_punctuation_metrics(text):
#         """Calculate punctuation metrics for a text"""
#         # Count punctuation marks
#         punct_counts = {p: text.count(p) for p in string.punctuation if text.count(p) > 0}
        
#         # Calculate spacing between punctuation
#         words = word_tokenize(text)
#         punct_positions = []
#         word_gaps = []
#         prev_punct_pos = -1
        
#         for i, word in enumerate(words):
#             if any(c in string.punctuation for c in word):
#                 punct_positions.append(i)
#                 if prev_punct_pos != -1:
#                     word_gaps.append(i - prev_punct_pos - 1)
#                 prev_punct_pos = i
        
#         avg_gap = np.mean(word_gaps) if word_gaps else 0
#         gap_std = np.std(word_gaps) if len(word_gaps) > 1 else 0
        
#         # Sentence length punctuation analysis
#         sentences = sent_tokenize(text)
#         sentence_ends = [sent[-1] if sent else '' for sent in sentences]
#         end_punct_dist = defaultdict(int)
#         for p in sentence_ends:
#             if p in string.punctuation:
#                 end_punct_dist[p] += 1
        
#         return {
#             'counts': punct_counts,
#             'avg_gap': avg_gap,
#             'gap_std': gap_std,
#             'end_punct_dist': dict(end_punct_dist),
#             'total_punct': sum(punct_counts.values()),
#             'word_count': len([w for w in words if w not in string.punctuation])
#         }

#     # Analyze assessment text
#     assess = calculate_punctuation_metrics(assess_text)
    
#     # Analyze baselines
#     base1 = calculate_punctuation_metrics(baseline1) if baseline1 else None
#     base2 = calculate_punctuation_metrics(baseline2) if baseline2 else None
    
#     # Prepare results
#     results = {
#         'assessment': {
#             'punctuation_counts': assess['counts'],
#             'avg_words_between_punct': assess['avg_gap'],
#             'punct_spacing_consistency': assess['gap_std'],
#             'sentence_end_distribution': assess['end_punct_dist'],
#             'punct_per_word': assess['total_punct'] / assess['word_count'] if assess['word_count'] > 0 else 0
#         },
#         'comparisons': {},
#         'flags': []
#     }
    
#     # Create comparisons
#     baselines = [('baseline1', base1), ('baseline2', base2)]
#     for name, base in baselines:
#         if base:
#             results['comparisons'][name] = {
#                 'avg_gap': base['avg_gap'],
#                 'gap_std': base['gap_std'],
#                 'punct_per_word': base['total_punct'] / base['word_count'] if base['word_count'] > 0 else 0,
#                 'gap_diff': assess['avg_gap'] - base['avg_gap'],
#                 'consistency_ratio': assess['gap_std'] / base['gap_std'] if base['gap_std'] > 0 else 1
#             }
    
#     # AI Detection Rules
#     # 1. Punctuation spacing consistency (low std dev suggests AI)
#     if assess['gap_std'] < 1.5:
#         results['flags'].append('FIA-SIG-P01: Unusually consistent punctuation spacing (std dev < 1.5)')
    
#     # 2. Overuse of certain punctuation
#     if assess['counts'].get('.', 0) / max(1, assess['total_punct']) > 0.5:
#         results['flags'].append('FIA-SIG-P02: Overuse of periods (>50% of all punctuation)')
    
#     # 3. Compare to baselines
#     if base1 and base2:
#         avg_base_std = (base1['gap_std'] + base2['gap_std']) / 2
#         if assess['gap_std'] < avg_base_std * 0.6:
#             results['flags'].append('FIA-SIG-P03: Punctuation spacing significantly more consistent than baselines')
            
#         avg_base_punct_rate = ((base1['total_punct'] / base1['word_count']) + 
#                              (base2['total_punct'] / base2['word_count'])) / 2
#         assess_punct_rate = assess['total_punct'] / assess['word_count']
#         if abs(assess_punct_rate - avg_base_punct_rate) > 0.2:
#             results['flags'].append('FIA-SIG-P04: Punctuation rate differs significantly from baselines')
    
#     # 4. Sentence end diversity
#     if len(assess['end_punct_dist']) < 2 and len(sent_tokenize(assess_text)) > 3:
#         results['flags'].append('FIA-SIG-P05: Limited sentence ending variety')
    
#     return results



def analyze_punctuation_patterns(baseline1, baseline2, assess_text=None):
    """
    Analyze punctuation patterns by estimating human reference from baselines.
    If assess_text is provided, compares it against estimated human profile.
    
    Args:
        baseline1: First baseline text (required)
        baseline2: Second baseline text (required)
        assess_text: Optional text to analyze against estimated human profile
        
    Returns:
        Dictionary containing:
        - estimated_human_reference: Punctuation profile from baselines
        - assess_text_analysis: Comparison results (if assess_text provided)
        - baseline_comparison: Raw baseline metrics
        - flags: Detected patterns
    """
    def calculate_punctuation_metrics(text):
        """Calculate all punctuation metrics for a text"""
        if not text:
            return None
            
        # Count all punctuation marks
        punct_counts = Counter(char for char in text if char in string.punctuation)
        
        # Calculate spacing between punctuation
        words = word_tokenize(text)
        word_gaps = []
        prev_punct_pos = -1
        
        for i, word in enumerate(words):
            if any(c in string.punctuation for c in word):
                if prev_punct_pos != -1:
                    word_gaps.append(i - prev_punct_pos - 1)
                prev_punct_pos = i
        
        # Sentence ending analysis
        sentences = [sent.strip() for sent in sent_tokenize(text) if sent.strip()]
        end_punct = [sent[-1] for sent in sentences if sent and sent[-1] in string.punctuation]
        end_punct_dist = Counter(end_punct)
        
        return {
            'counts': dict(punct_counts),
            'avg_gap': np.mean(word_gaps) if word_gaps else 0,
            'gap_std': np.std(word_gaps) if len(word_gaps) > 1 else 0,
            'end_punct_dist': end_punct_dist,
            'total_punct': sum(punct_counts.values()),
            'word_count': len([w for w in words if w not in string.punctuation]),
            'text_sample': text if len(text) < 100 else text[:100] + "..."
        }

    # --- ESTIMATE HUMAN REFERENCE FROM BASELINES ---
    baseline_metrics = []
    for text in [baseline1, baseline2]:
        metrics = calculate_punctuation_metrics(text)
        if metrics:
            baseline_metrics.append(metrics)
    
    if len(baseline_metrics) < 2:
        return {'error': 'Need at least 2 valid baselines for estimation'}
    
    # Create composite human reference (averaging key metrics)
    estimated_reference = {
        'counts': defaultdict(int),
        'avg_gap': np.mean([b['avg_gap'] for b in baseline_metrics]),
        'gap_std': np.mean([b['gap_std'] for b in baseline_metrics]),
        'end_punct_dist': defaultdict(int),
        'total_punct': int(np.mean([b['total_punct'] for b in baseline_metrics])),
        'word_count': int(np.mean([b['word_count'] for b in baseline_metrics]))
    }
    
    # Merge punctuation counts
    for b in baseline_metrics:
        for p, count in b['counts'].items():
            estimated_reference['counts'][p] += count
        for p, count in b['end_punct_dist'].items():
            estimated_reference['end_punct_dist'][p] += count
    
    # --- ANALYZE ASSESS TEXT (IF PROVIDED) ---
    assessment = None
    flags = []
    if assess_text:
        assess_data = calculate_punctuation_metrics(assess_text)
        if assess_data:
            # Calculate deviation scores
            gap_deviation = abs(assess_data['avg_gap'] - estimated_reference['avg_gap']) / estimated_reference['avg_gap'] if estimated_reference['avg_gap'] > 0 else 0
            std_deviation = abs(assess_data['gap_std'] - estimated_reference['gap_std']) / estimated_reference['gap_std'] if estimated_reference['gap_std'] > 0 else 0
            
            # AI Probability Score (0-100)
            ai_score = 0
            
            # 1. Punctuation consistency (low std dev suggests AI)
            if assess_data['gap_std'] < estimated_reference['gap_std'] * 0.6:
                ai_score += 30 * (1 - (assess_data['gap_std'] / (estimated_reference['gap_std'] * 0.6)))
                flags.append('FIA-SIG-P01: Unusually consistent punctuation spacing')
            
            # 2. Punctuation rate deviation
            ref_rate = estimated_reference['total_punct'] / estimated_reference['word_count']
            assess_rate = assess_data['total_punct'] / assess_data['word_count']
            if abs(assess_rate - ref_rate) > 0.2:
                ai_score += 20 * min(abs(assess_rate - ref_rate) / 0.2, 2)
                flags.append('FIA-SIG-P02: Abnormal punctuation frequency')
            
            # 3. Ending punctuation diversity
            if len(assess_data['end_punct_dist']) < 2 and len(sent_tokenize(assess_text)) > 3:
                ai_score += 15
                flags.append('FIA-SIG-P03: Limited sentence ending variety')
            
            # 4. Unusual punctuation distribution
            common_punct = {k for k,v in estimated_reference['counts'].items() if v > 0}
            assess_punct = set(assess_data['counts'].keys())
            if len(assess_punct - common_punct) > 2:
                ai_score += 10 * min(len(assess_punct - common_punct), 3)
                flags.append('FIA-SIG-P04: Uncommon punctuation marks used')
            
            ai_score = min(100, ai_score)
            
            assessment = {
                'metrics': assess_data,
                'comparison': {
                    'gap_vs_reference': f"{assess_data['avg_gap']/estimated_reference['avg_gap']:.1%}",
                    'consistency_ratio': f"{assess_data['gap_std']/estimated_reference['gap_std']:.1%}",
                    'punct_rate_diff': f"{(assess_rate - ref_rate):+.2f}"
                },
                'ai_score': int(ai_score),
                'assessment': "Likely AI-generated" if ai_score >= 70 else
                            "Possibly AI-generated" if ai_score >= 40 else
                            "Likely human-written",
                'flags': flags
            }
    
    # --- FINAL RESULTS ---
    return {
        'estimated_human_reference': {
            'punctuation_profile': {
                'avg_words_between_punct': estimated_reference['avg_gap'],
                'punct_spacing_consistency': estimated_reference['gap_std'],
                'common_punctuation': dict(estimated_reference['counts']),
                'sentence_endings': dict(estimated_reference['end_punct_dist']),
                'punct_per_word': estimated_reference['total_punct'] / estimated_reference['word_count']
            },
            'source': 'Composite of baseline1 and baseline2',
            'text_samples': [b['text_sample'] for b in baseline_metrics]
        },
        'assess_text_analysis': assessment if assess_text else "No assess_text provided",
        'baseline_comparison': {
            'baseline1': calculate_punctuation_metrics(baseline1),
            'baseline2': calculate_punctuation_metrics(baseline2)
        },
        'scoring_notes': {
            'ai_thresholds': ">70=AI, 40-70=Possible AI, <40=Human",
            'key_metrics': "Focus on spacing consistency and punctuation variety"
        }
    }
# text = "In conclusion, this means that AI is helpful. Therefore, we should use it."
# print(gpt_style_phrases(text)) 
# Output: ['in conclusion', 'this means that', 'therefore']
def gpt_style_phrases(text):
    patterns = [
        "in conclusion", "as a result", "this means that",
        "it is important to note", "therefore", "in summary",
        "on the other hand", "to sum up"
    ]
    return [phrase for phrase in patterns if phrase in text.lower()]


# def get_repeated_ngrams(text, min_n=3, max_n=5, min_count=2):
#     """
#     Find repeated n-grams (multi-word phrases) in text
#     Args:
#         text: input text to analyze
#         min_n: minimum n-gram length (default 3)
#         max_n: maximum n-gram length (default 5)
#         min_count: minimum occurrence count to consider (default 2)
#     Returns:
#         Dictionary of {n_gram_length: {phrase: count}}
#     """
#     # Clean and tokenize text
#     words = word_tokenize(text.lower())
#     words = [word for word in words if word.isalnum()]  # Remove punctuation
    
#     repeated_phrases = defaultdict(dict)
    
#     for n in range(min_n, max_n + 1):
#         # Generate all n-grams
#         phrase_counts = Counter(ngrams(words, n))
        
#         # Filter for repeated phrases
#         for phrase, count in phrase_counts.items():
#             if count >= min_count:
#                 phrase_text = ' '.join(phrase)
#                 repeated_phrases[n][phrase_text] = count
                
#     return dict(repeated_phrases)
# def compare_repeated_phrases(assess_text, baseline1_text, baseline2_text):
#     """
#     Compare repeated phrase patterns between assessment and baseline texts
#     Returns analysis with metrics and recommendations in JSON-serializable format
#     """
#     def convert_value(v):
#         """Convert numpy and special values to JSON-safe types"""
#         if hasattr(v, 'item'):  # Handle numpy types
#             v = v.item()
#         if isinstance(v, (int, float)):
#             if math.isinf(v):
#                 return "inf" if v > 0 else "-inf"
#             if math.isnan(v):
#                 return "nan"
#         return v

#     # Analyze all texts with default empty dict if None
#     assess_phrases = get_repeated_ngrams(assess_text) if assess_text else {}
#     baseline1_phrases = get_repeated_ngrams(baseline1_text) if baseline1_text else {}
#     baseline2_phrases = get_repeated_ngrams(baseline2_text) if baseline2_text else {}

#     # Safely calculate metrics
#     def count_total_repeats(phrase_dict):
#         if not phrase_dict or not isinstance(phrase_dict, dict):
#             return 0
#         return sum(len(phrases) for phrases in phrase_dict.values() if phrases)

#     assess_count = count_total_repeats(assess_phrases)
#     baseline1_count = count_total_repeats(baseline1_phrases)
#     baseline2_count = count_total_repeats(baseline2_phrases)
    
#     # Calculate metrics with native Python types
#     baseline_avg = float(baseline1_count + baseline2_count) / 2 if (baseline1_count + baseline2_count) > 0 else 0.0
#     deviation = float(assess_count) - baseline_avg
#     deviation_percent = (deviation / baseline_avg * 100) if baseline_avg != 0 else float('inf')

#     # Generate results with JSON-safe types
#     results = {
#         'assessment': assess_phrases or {},
#         'comparisons': {
#             'baseline1': baseline1_phrases or {},
#             'baseline2': baseline2_phrases or {}
#         },
#         'metrics': {
#             'total_repeats': {
#                 'assessment': int(assess_count),
#                 'baseline1': int(baseline1_count),
#                 'baseline2': int(baseline2_count),
#                 'baseline_average': convert_value(baseline_avg)
#             },
#             'deviation': convert_value(deviation),
#             'deviation_percent': convert_value(deviation_percent)
#         },
#         'flags': [],
#         'recommendations': [],
#         'unique_phrases': []
#     }

#     # Generate recommendations
#     if baseline_avg > 0:  # Only compare if we have baselines
#         if assess_count > baseline_avg * 1.5:
#             results['flags'].append('FIA-RP01: High repetition')
#             results['recommendations'].append("Significant repetition detected (50%+ above baseline)")
#         elif assess_count > baseline_avg * 1.2:
#             results['flags'].append('FIA-RP02: Moderate repetition')
#             results['recommendations'].append("Moderate repetition detected (20%+ above baseline)")
#         elif assess_count < baseline_avg * 0.8:
#             results['recommendations'].append("Good phrase variation (below baseline)")
#         else:
#             results['recommendations'].append("Normal repetition level")
#     else:
#         results['flags'].append('FIA-RP03: No baseline comparison')
#         results['recommendations'].append("No baseline data available for comparison")

#     # Identify unique phrases
#     unique_phrases = set()
#     if assess_phrases:
#         for n in assess_phrases:
#             for phrase in assess_phrases.get(n, []):
#                 in_baseline1 = any(phrase in baseline1_phrases.get(m, {}) for m in baseline1_phrases)
#                 in_baseline2 = any(phrase in baseline2_phrases.get(m, {}) for m in baseline2_phrases)
#                 if not in_baseline1 and not in_baseline2:
#                     unique_phrases.add(phrase)

#     if unique_phrases:
#         results['unique_phrases'] = sorted(unique_phrases)
#         results['flags'].append('FIA-RP04: Unique repeated phrases')

#     # Final JSON serialization check
#     def make_json_serializable(obj):
#         if isinstance(obj, dict):
#             return {k: make_json_serializable(v) for k, v in obj.items()}
#         elif isinstance(obj, (list, tuple, set)):
#             return [make_json_serializable(x) for x in obj]
#         return convert_value(obj)

#     return make_json_serializable(results)

def compare_repeated_phrases(baseline1_text, baseline2_text, assess_text=None):
    """
    Compare repeated phrase patterns by estimating human reference from baselines.
    If assess_text is provided, compares it against estimated human profile.
    
    Args:
        baseline1_text: First baseline text (required)
        baseline2_text: Second baseline text (required)
        assess_text: Optional text to analyze against estimated human profile
        
    Returns:
        Dictionary containing:
        - estimated_human_reference: Repetition profile from baselines
        - assess_text_analysis: Comparison results (if assess_text provided)
        - baseline_comparison: Raw baseline metrics
        - flags: Detected patterns
    """
    def get_repeated_ngrams(text):
        """Find repeated n-grams in text (3-5 word phrases)"""
        if not text:
            return {}
            
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum()]
        
        repeated = defaultdict(dict)
        for n in range(3, 6):  # 3-5 word phrases
            phrase_counts = Counter(ngrams(words, n))
            for phrase, count in phrase_counts.items():
                if count >= 2:  # Minimum 2 occurrences
                    repeated[n][' '.join(phrase)] = count
        return dict(repeated)

    def count_total_repeats(phrase_dict):
        """Count total repeated phrases in a phrase dictionary"""
        return sum(len(phrases) for phrases in phrase_dict.values()) if phrase_dict else 0

    # --- ESTIMATE HUMAN REFERENCE FROM BASELINES ---
    baseline1_phrases = get_repeated_ngrams(baseline1_text)
    baseline2_phrases = get_repeated_ngrams(baseline2_text)
    
    baseline1_count = count_total_repeats(baseline1_phrases)
    baseline2_count = count_total_repeats(baseline2_phrases)
    
    estimated_reference = {
        'avg_repeats': (baseline1_count + baseline2_count) / 2,
        'phrase_length_dist': {
            3: len(baseline1_phrases.get(3, {})) + len(baseline2_phrases.get(3, {})) / 2,
            4: len(baseline1_phrases.get(4, {})) + len(baseline2_phrases.get(4, {})) / 2,
            5: len(baseline1_phrases.get(5, {})) + len(baseline2_phrases.get(5, {})) / 2
        },
        'common_phrases': {
            n: list(set(baseline1_phrases.get(n, {}).keys()) | set(baseline2_phrases.get(n, {}).keys()))
            for n in range(3, 6)
        },
        'text_samples': [
            baseline1_text[:100] + "..." if len(baseline1_text) > 100 else baseline1_text,
            baseline2_text[:100] + "..." if len(baseline2_text) > 100 else baseline2_text
        ]
    }

    # --- ANALYZE ASSESS TEXT (IF PROVIDED) ---
    assessment = None
    flags = []
    if assess_text:
        assess_phrases = get_repeated_ngrams(assess_text)
        assess_count = count_total_repeats(assess_phrases)
        
        # Calculate deviation metrics
        deviation = assess_count - estimated_reference['avg_repeats']
        deviation_pct = (deviation / estimated_reference['avg_repeats'] * 100) if estimated_reference['avg_repeats'] > 0 else float('inf')
        
        # AI Probability Score (0-100)
        ai_score = 0
        
        # 1. High repetition
        if deviation_pct > 50:
            ai_score += min(40, 40 * (deviation_pct / 100))
            flags.append(f'FIA-RP01: High repetition (+{deviation_pct:.0f}% above baseline)')
        
        # 2. Unique repeated phrases
        unique_phrases = []
        for n in assess_phrases:
            for phrase in assess_phrases[n]:
                if phrase not in estimated_reference['common_phrases'][n]:
                    unique_phrases.append(phrase)
        
        if unique_phrases:
            ai_score += min(30, len(unique_phrases) * 5)  # +5 per unique phrase, max +30
            flags.append(f'FIA-RP02: {len(unique_phrases)} unique repeated phrases')
        
        # 3. Unusual phrase length distribution
        assess_dist = {n: len(assess_phrases.get(n, {})) for n in range(3, 6)}
        dist_deviation = sum(
            abs(assess_dist[n] - estimated_reference['phrase_length_dist'][n]) 
            for n in range(3, 6)
        )
        if dist_deviation > 3:
            ai_score += min(20, dist_deviation * 5)
            flags.append('FIA-RP03: Unusual phrase length distribution')
        
        ai_score = min(100, ai_score)
        
        assessment = {
            'total_repeats': assess_count,
            'phrase_distribution': assess_dist,
            'comparison': {
                'deviation_from_avg': f"{deviation:+.1f}",
                'percent_deviation': f"{deviation_pct:+.1f}%",
                'unique_phrases_count': len(unique_phrases)
            },
            'ai_score': int(ai_score),
            'assessment': "Likely AI-generated" if ai_score >= 70 else
                         "Possibly AI-generated" if ai_score >= 40 else
                         "Likely human-written",
            'flags': flags,
            'unique_phrases': unique_phrases[:10]  # Show first 10 for brevity
        }

    # --- FINAL RESULTS ---
    return {
        'estimated_human_reference': {
            'avg_repeated_phrases': estimated_reference['avg_repeats'],
            'phrase_length_distribution': estimated_reference['phrase_length_dist'],
            'source': 'Composite of baseline1 and baseline2',
            'text_samples': estimated_reference['text_samples']
        },
        'assess_text_analysis': assessment if assess_text else "No assess_text provided",
        'baseline_comparison': {
            'baseline1': {
                'total_repeats': baseline1_count,
                'phrase_distribution': {n: len(baseline1_phrases.get(n, {})) for n in range(3, 6)}
            },
            'baseline2': {
                'total_repeats': baseline2_count,
                'phrase_distribution': {n: len(baseline2_phrases.get(n, {})) for n in range(3, 6)}
            }
        },
        'scoring_notes': {
            'ai_thresholds': ">70=AI, 40-70=Possible AI, <40=Human",
            'key_metrics': "Focus on total repeats and unique phrase count"
        }
    }

# test_text = """
# The rapid advancement of artificial intelligence presents both opportunities and challenges for modern society. 
# Many experts believe AI will significantly transform the workforce in coming decades. 
# However, the ethical implications of these changes require careful consideration.

# [Several paragraphs of content...]

# In conclusion, AI development is progressing at an unprecedented pace. 
# We should carefully think about how these technologies affect people. 
# Are we prepared for the societal impacts of artificial intelligence?
# """
# Result
# {
#     "opening": {
#         "avg_sentence_length": 13.3,  # Approximate
#         "avg_word_length": 4.5,       # Approximate  
#         "pos_distribution": {"NOUN": 5, "VERB": 3, ...},  # POS counts
#         "readability": 50.2,           # Flesch reading ease score
#         "lexical_diversity": 0.75,      # Ratio of unique words
#         "tone": "formal"
#     },
#     "closing": {
#         "avg_sentence_length": 10.7,
#         "avg_word_length": 4.2,
#         "pos_distribution": {"NOUN": 4, "VERB": 4, ...},
#         "readability": 55.1,
#         "lexical_diversity": 0.72,
#         "tone": "interrogative"
#     },
#     "comparison": {
#         "semantic_similarity": 0.78,    # Between 0-1
#         "length_diff": 2.6,
#         "readability_diff": 4.9,
#         "tone_consistency": False,
#         "pos_correlation": 0.65         # Between 0-1
#     }
# }
# def _analyze_opening_closing(text, num_sentences=3):
#     nlp = analyzer.get_model('lg')
#     """
#     Analyze consistency between opening and closing sections of text.
    
#     Args:
#         text: Input text to analyze
#         num_sentences: Number of sentences to consider as opening/closing
        
#     Returns:
#         Dictionary containing comparison metrics
#     """
#     doc = nlp(text)
#     sentences = [sent.text for sent in doc.sents]
    
#     if len(sentences) < num_sentences*2:
#         return {"error": f"Text too short - needs at least {num_sentences*2} sentences"}
    
#     opening = sentences[:num_sentences]
#     closing = sentences[-num_sentences:]
    
#     # Process sections
#     def process_section(section):
#         combined = " ".join(section)
#         section_doc = nlp(combined)
        
#         return {
#             "avg_sentence_length": sum(len(s.split()) for s in section)/num_sentences,
#             "avg_word_length": sum(len(token.text) for token in section_doc)/len(list(section_doc)),
#             "pos_distribution": defaultdict(int, 
#                 [(token.pos_, token.pos) for token in section_doc]),
#             "readability": textstat.flesch_reading_ease(combined),
#             "lexical_diversity": len(set(token.text.lower() for token in section_doc))/len(list(section_doc)),
#             "tone": analyze_tone(combined)
#         }
    
#     opening_stats = process_section(opening)
#     closing_stats = process_section(closing)
    
#     # Calculate similarity scores
#     def cosine_sim(vec1, vec2):
#         return vec1.similarity(vec2) if vec1.has_vector and vec2.has_vector else 0
        
#     opening_doc = nlp(" ".join(opening))
#     closing_doc = nlp(" ".join(closing))
    
#     return {
#         "opening": opening_stats,
#         "closing": closing_stats,
#         "comparison": {
#             "semantic_similarity": cosine_sim(opening_doc, closing_doc),
#             "length_diff": abs(opening_stats["avg_sentence_length"] - closing_stats["avg_sentence_length"]),
#             "readability_diff": abs(opening_stats["readability"] - closing_stats["readability"]),
#             "tone_consistency": opening_stats["tone"] == closing_stats["tone"],
#             "pos_correlation": sum(
#                 min(opening_stats["pos_distribution"][pos], closing_stats["pos_distribution"][pos])
#                 for pos in set(opening_stats["pos_distribution"]) | set(closing_stats["pos_distribution"])
#             ) / (len(list(opening_doc)) + len(list(closing_doc)))
#         }
#     }

# def analyze_opening_closing(ass,baseline1,baseline2):
#     return{
#         "assess":_analyze_opening_closing(ass),
#         "baseline1":_analyze_opening_closing(baseline1),
#         "baseline2":_analyze_opening_closing(baseline2)
#     }
def analyze_opening_closing(baseline1, baseline2, ass=None):
    """
    Analyze opening/closing consistency with robust error handling.
    Estimates human reference from baselines when possible.
    """
    def _analyze_text(text):
        """Analyze a single text with comprehensive error handling"""
        if not text or not isinstance(text, str):
            return {"error": "Invalid text input"}
            
        try:
            nlp = analyzer.get_model('lg')
            doc = nlp(text)
            sentences = [sent.text for sent in doc.sents]
            
            if len(sentences) < 6:
                return {"error": f"Needs 6+ sentences (got {len(sentences)})"}
            
            opening = sentences[:3]
            closing = sentences[-3:]
            
            def process_section(section):
                if not section:
                    return None
                combined = " ".join(section)
                section_doc = nlp(combined)
                return {
                    "avg_sentence_length": sum(len(s.split()) for s in section)/len(section),
                    "avg_word_length": sum(len(token.text) for token in section_doc)/max(1, len(list(section_doc))),
                    "readability": textstat.flesch_reading_ease(combined),
                    "lexical_diversity": len(set(token.text.lower() for token in section_doc))/max(1, len(list(section_doc))),
                    "vector": section_doc.vector if section_doc.has_vector else None
                }
            
            opening_stats = process_section(opening) or {}
            closing_stats = process_section(closing) or {}
            
            similarity = 0
            if opening_stats.get('vector') is not None and closing_stats.get('vector') is not None:
                similarity = cosine_similarity(
                    [opening_stats['vector']],
                    [closing_stats['vector']]
                )[0][0]
            
            return {
                "opening": opening_stats,
                "closing": closing_stats,
                "consistency_score": similarity,
                "text_sample": text[:100] + "..." if len(text) > 100 else text
            }
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    # --- ESTIMATE HUMAN REFERENCE ---
    baseline1_analysis = _analyze_text(baseline1)
    baseline2_analysis = _analyze_text(baseline2)
    
    # Initialize empty reference structure
    estimated_reference = {
        "opening": {},
        "closing": {},
        "expected_consistency": 0,
        "source": "Composite of available baselines",
        "text_samples": [],
        "warnings": []
    }
    
    # Calculate composite metrics only for valid baselines
    valid_baselines = []
    for i, analysis in enumerate([baseline1_analysis, baseline2_analysis], 1):
        if not analysis.get('error') and analysis.get('opening') and analysis.get('closing'):
            valid_baselines.append(analysis)
        elif analysis.get('error'):
            estimated_reference['warnings'].append(f"baseline{i} error: {analysis['error']}")
    
    if len(valid_baselines) >= 1:
        # Calculate averages for each metric
        for section in ['opening', 'closing']:
            for metric in ['avg_sentence_length', 'avg_word_length', 'readability', 'lexical_diversity']:
                values = [b[section].get(metric, 0) for b in valid_baselines if section in b]
                if values:
                    estimated_reference[section][metric] = sum(values) / len(values)
        
        # Calculate average consistency score
        consistency_scores = [b['consistency_score'] for b in valid_baselines if 'consistency_score' in b]
        if consistency_scores:
            estimated_reference['expected_consistency'] = sum(consistency_scores) / len(consistency_scores)
        
        # Add text samples
        estimated_reference['text_samples'] = [b.get('text_sample', '') for b in valid_baselines]
    else:
        estimated_reference['error'] = "No valid baselines for estimation"
    
    # --- ANALYZE ASSESSMENT TEXT ---
    ass_analysis = None
    flags = []
    if ass:
        ass_data = _analyze_text(ass)
        if not ass_data.get('error') and estimated_reference.get('opening'):
            try:
                # Calculate deviations
                deviations = {
                    'opening': {},
                    'closing': {},
                    'consistency_diff': 0
                }
                
                for section in ['opening', 'closing']:
                    for metric in ['avg_sentence_length', 'avg_word_length', 'readability', 'lexical_diversity']:
                        if metric in ass_data[section] and metric in estimated_reference[section]:
                            deviations[section][metric] = abs(
                                ass_data[section][metric] - estimated_reference[section][metric]
                            )
                
                if 'consistency_score' in ass_data:
                    deviations['consistency_diff'] = abs(
                        ass_data['consistency_score'] - estimated_reference['expected_consistency']
                    )
                
                # AI scoring
                ai_score = 0
                
                # 1. Consistency difference
                if deviations['consistency_diff'] > 0.3:
                    ai_score += min(40, 40 * (deviations['consistency_diff'] / 0.5))
                    flags.append(f'FIA-OC01: High inconsistency ({deviations["consistency_diff"]:.2f})')
                
                # 2. Lexical diversity deviation
                if deviations['opening'].get('lexical_diversity', 0) > 0.15:
                    ai_score += 15
                    flags.append('FIA-OC02: Unusual opening vocabulary')
                
                if deviations['closing'].get('lexical_diversity', 0) > 0.15:
                    ai_score += 15
                    flags.append('FIA-OC03: Unusual closing vocabulary')
                
                # 3. Readability difference
                if deviations['opening'].get('readability', 0) > 15:
                    ai_score += 10
                    flags.append('FIA-OC04: Opening readability mismatch')
                
                ai_score = min(100, ai_score)
                
                ass_analysis = {
                    "metrics": ass_data,
                    "deviations": deviations,
                    "ai_score": int(ai_score),
                    "assessment": "Likely AI-generated" if ai_score >= 70 else
                                "Possibly AI-generated" if ai_score >= 40 else
                                "Likely human-written",
                    "flags": flags
                }
            except Exception as e:
                ass_analysis = {"error": f"Assessment analysis failed: {str(e)}"}
        else:
            ass_analysis = {
                "error": ass_data.get('error', 'Invalid assessment text'),
                "reference_error": estimated_reference.get('error')
            }
    
    # --- FINAL RESULTS ---
    result = {
        "estimated_human_reference": estimated_reference,
        "baseline_comparison": {
            "baseline1": baseline1_analysis,
            "baseline2": baseline2_analysis
        },
        "scoring_notes": {
            "ai_thresholds": ">70=AI, 40-70=Possible AI, <40=Human",
            "requirements": "Minimum 6 sentences for analysis"
        }
    }
    
    if ass:
        result["ass_analysis"] = ass_analysis
    
    return result

# def analyze_tone(text):
#     """Simple tone analyzer (expand with more sophisticated NLP)"""
#     nlp = analyzer.get_model('lg')
#     doc = nlp(text)
    
#     # Count formal/informal markers
#     formal = sum(1 for token in doc if token.text.lower() in ["furthermore", "moreover", "however"])
#     informal = sum(1 for token in doc if token.text.lower() in ["so", "well", "you know"])
    
#     questions = sum(1 for sent in doc.sents if sent.text.endswith("?"))
    
#     if questions > 1:
#         return "interrogative"
#     elif formal > informal:
#         return "formal"
#     elif informal > formal:
#         return "informal"
#     return "neutral"

def compare_opening_closing(assess_text: str, baseline1_text: str, baseline2_text: str, num_sentences: int = 3) -> Dict[str, Any]:
    """
    Compare opening and closing sections across assessment and baseline texts.
    
    Args:
        assess_text: Text to be assessed
        baseline1_text: First baseline comparison text
        baseline2_text: Second baseline comparison text
        num_sentences: Number of sentences to consider as opening/closing
        
    Returns:
        Dictionary containing comparison metrics across all texts
    """
    nlp = analyzer.get_model('lg')
    
    def analyze_text(text: str) -> Dict[str, Any]:
        """Helper function to analyze a single text"""
        if not text or not isinstance(text, str):
            return {"error": "Invalid text input"}
            
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        if len(sentences) < num_sentences * 2:
            return {"error": f"Text too short - needs at least {num_sentences*2} sentences"}
        
        opening = sentences[:num_sentences]
        closing = sentences[-num_sentences:]
        
        def process_section(section: list) -> Dict[str, Any]:
            """Process a text section (opening or closing)"""
            combined = " ".join(section)
            section_doc = nlp(combined)
            words = [token.text for token in section_doc if not token.is_punct]
            
            # POS distribution counting both coarse and fine tags
            pos_counts = defaultdict(int)
            for token in section_doc:
                pos_counts[token.pos_] += 1
                pos_counts[token.tag_] += 1
            
            return {
                "sentences": section,
                "avg_sentence_length": sum(len(s.split()) for s in section) / num_sentences,
                "avg_word_length": sum(len(token.text) for token in section_doc if not token.is_punct) / max(1, len(words)),
                "pos_distribution": dict(pos_counts),
                "readability": textstat.flesch_reading_ease(combined),
                "lexical_diversity": len(set(token.text.lower() for token in section_doc if not token.is_punct)) / max(1, len(words)),
                "tone": analyze_tone(combined),
                "word_count": len(words)
            }
        
        opening_stats = process_section(opening)
        closing_stats = process_section(closing)
        
        # Calculate comparisons between opening and closing
        opening_doc = nlp(" ".join(opening))
        closing_doc = nlp(" ".join(closing))
        
        return {
            "opening": opening_stats,
            "closing": closing_stats,
            "comparison": {
                "semantic_similarity": cosine_sim(opening_doc, closing_doc),
                "length_diff": abs(opening_stats["avg_sentence_length"] - closing_stats["avg_sentence_length"]),
                "readability_diff": abs(opening_stats["readability"] - closing_stats["readability"]),
                "tone_consistency": opening_stats["tone"] == closing_stats["tone"],
                "pos_correlation": calculate_pos_correlation(opening_stats["pos_distribution"], closing_stats["pos_distribution"]),
                "content_continuity": check_content_continuity(opening, closing)
            }
        }
    
    def cosine_sim(vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors"""
        return vec1.similarity(vec2) if vec1.has_vector and vec2.has_vector else 0.0
    
    def calculate_pos_correlation(pos1: dict, pos2: dict) -> float:
        """Calculate POS tag correlation between sections"""
        all_tags = set(pos1.keys()).union(set(pos2.keys()))
        if not all_tags:
            return 0.0
        return sum(min(pos1.get(tag, 0), pos2.get(tag, 0)) for tag in all_tags) / sum(max(pos1.get(tag, 0), pos2.get(tag, 0)) for tag in all_tags)
    
    def check_content_continuity(opening: list, closing: list) -> bool:
        """Check if key concepts are maintained between opening and closing"""
        opening_concepts = set(token.lemma_ for token in nlp(" ".join(opening)) if token.pos_ in ["NOUN", "PROPN"])
        closing_concepts = set(token.lemma_ for token in nlp(" ".join(closing)) if token.pos_ in ["NOUN", "PROPN"])
        return len(opening_concepts.intersection(closing_concepts)) / len(opening_concepts.union(closing_concepts)) > 0.5
    
    # Analyze all three texts
    assess_results = analyze_text(assess_text)
    baseline1_results = analyze_text(baseline1_text)
    baseline2_results = analyze_text(baseline2_text)
    
    # Generate comparative analysis
    def compare_to_baselines(metric: str, section: str) -> Dict[str, Any]:
        """Compare assessment metric to baseline averages"""
        baseline_values = []
        for res in [baseline1_results, baseline2_results]:
            if "error" not in res and metric in res[section]:
                baseline_values.append(res[section][metric])
        
        if not baseline_values:
            return {"value": assess_results[section].get(metric), "baseline": None}
        
        baseline_avg = sum(baseline_values) / len(baseline_values)
        assess_value = assess_results[section].get(metric, 0)
        
        return {
            "value": assess_value,
            "baseline_avg": baseline_avg,
            "difference": assess_value - baseline_avg,
            "percentage_diff": ((assess_value - baseline_avg) / baseline_avg * 100) if baseline_avg != 0 else 0
        }
    
    # Build comprehensive results
    results = {
        "assessment": assess_results,
        "baselines": {
            "baseline1": baseline1_results,
            "baseline2": baseline2_results
        },
        "comparative_analysis": {
            "opening": {
                "sentence_length": compare_to_baselines("avg_sentence_length", "opening"),
                "word_length": compare_to_baselines("avg_word_length", "opening"),
                "readability": compare_to_baselines("readability", "opening"),
                "lexical_diversity": compare_to_baselines("lexical_diversity", "opening"),
                "tone": {
                    "assessment": assess_results["opening"].get("tone"),
                    "baseline1": baseline1_results["opening"].get("tone") if "opening" in baseline1_results else None,
                    "baseline2": baseline2_results["opening"].get("tone") if "opening" in baseline2_results else None
                }
            },
            "closing": {
                "sentence_length": compare_to_baselines("avg_sentence_length", "closing"),
                "word_length": compare_to_baselines("avg_word_length", "closing"),
                "readability": compare_to_baselines("readability", "closing"),
                "lexical_diversity": compare_to_baselines("lexical_diversity", "closing"),
                "tone": {
                    "assessment": assess_results["closing"].get("tone"),
                    "baseline1": baseline1_results["closing"].get("tone") if "closing" in baseline1_results else None,
                    "baseline2": baseline2_results["closing"].get("tone") if "closing" in baseline2_results else None
                }
            },
            "section_consistency": {
                "semantic_similarity": compare_to_baselines("semantic_similarity", "comparison"),
                "tone_consistency": {
                    "assessment": assess_results["comparison"].get("tone_consistency"),
                    "baseline1": baseline1_results["comparison"].get("tone_consistency") if "comparison" in baseline1_results else None,
                    "baseline2": baseline2_results["comparison"].get("tone_consistency") if "comparison" in baseline2_results else None
                },
                "content_continuity": compare_to_baselines("content_continuity", "comparison")
            }
        },
        "recommendations": generate_recommendations(assess_results, baseline1_results, baseline2_results)
    }
    
    return results

def analyze_tone(text: str) -> str:
    """Enhanced tone analyzer with more linguistic features"""
    nlp = analyzer.get_model('lg')
    doc = nlp(text)
    
    # Tone markers
    formal_markers = {"furthermore", "moreover", "however", "thus", "therefore"}
    informal_markers = {"so", "well", "you know", "like", "basically"}
    emotional_markers = {"!", "wow", "unfortunately", "fortunately"}
    
    formal = sum(1 for token in doc if token.text.lower() in formal_markers)
    informal = sum(1 for token in doc if token.text.lower() in informal_markers)
    emotional = sum(1 for token in doc if token.text.lower() in emotional_markers)
    questions = sum(1 for sent in doc.sents if sent.text.endswith("?"))
    
    # Sentence mood analysis
    imperative = sum(1 for sent in doc.sents if any(tok.dep_ == "ROOT" and tok.tag_ == "VB" for tok in sent))
    
    if questions > 1:
        return "interrogative"
    elif imperative > 0:
        return "directive"
    elif emotional > 2:
        return "emotional"
    elif formal > informal:
        return "formal"
    elif informal > formal:
        return "informal"
    return "neutral"

def generate_recommendations(assess: dict, baseline1: dict, baseline2: dict) -> list:
    """Generate writing recommendations based on analysis"""
    recommendations = []
    
    # Check for errors first
    if "error" in assess:
        return ["Assessment text analysis failed: " + assess["error"]]
    
    # Tone consistency recommendations
    if assess["comparison"].get("tone_consistency") is False:
        recommendations.append(
            "Tone shift detected between opening and closing. "
            f"Opening is {assess['opening']['tone']} while closing is {assess['closing']['tone']}. "
            "Consider making the tone more consistent for better flow."
        )
    
    # Compare to baselines
    def get_metric(section: str, metric: str) -> tuple:
        """Helper to get metric values"""
        base1 = baseline1.get(section, {}).get(metric, 0) if "error" not in baseline1 else 0
        base2 = baseline2.get(section, {}).get(metric, 0) if "error" not in baseline2 else 0
        baseline_avg = (base1 + base2) / 2 if (base1 + base2) > 0 else 0
        assess_val = assess.get(section, {}).get(metric, 0)
        return assess_val, baseline_avg
    
    # Opening length comparison
    open_len, base_open_len = get_metric("opening", "avg_sentence_length")
    if base_open_len > 0 and open_len > base_open_len * 1.3:
        recommendations.append(
            "Opening sentences are significantly longer than baseline average "
            f"({open_len:.1f} vs {base_open_len:.1f} words). Consider making them more concise."
        )
    
    # Content continuity recommendation
    if assess["comparison"].get("content_continuity", 0) < 0.4:
        recommendations.append(
            "Low content continuity between opening and closing. Only %.0f%% of key concepts "
            "are maintained. Consider better topic alignment." % (assess["comparison"]["content_continuity"] * 100)
        )
    
    if not recommendations:
        recommendations.append("Text structure and flow are well within expected parameters.")
    
    return recommendations


# Example Usage:
# text = """
# The experiment was conducted. However, the results were unexpected. 
# This suggests methodological issues. Meanwhile, the control group showed normal patterns.

# The data was reanalyzed. That analysis confirmed our hypothesis. 
# These findings demonstrate the need for careful controls.
# """

# results = analyze_semantic_flow(text)
# print(f"Cohesion Score: {results['cohesion']['score']:.2f}")
# print(f"Referential Clarity: {results['referential_clarity']['score']:.2f}")
# print(f"Topic Drift Warnings: {results['topic_drift']['warnings']}")
# print(f"Avg Transition Smoothness: {results['transition_smoothness']['avg_score']:.2f}")
# Output Interpretation:
# Cohesion Score: >0.3 = good, <0.1 = poor

# Referential Clarity: 1 = perfect, <0.7 = needs improvement

# Topic Drift: Any warnings indicate potential issues

# Transition Smoothness: >0.5 = smooth, <0.3 = abrupt

def semantic_flow(text):
    nlp = analyzer.get_model('lg')
    doc = nlp(text)
    sentences = [sent for sent in doc.sents]
    paragraphs = text.split('\n\n') if '\n\n' in text else [text]
    
    analysis = {
        'cohesion': {'score': 0, 'transitions': defaultdict(int)},
        'referential_clarity': {'score': 0, 'references': []},
        'topic_drift': {'warnings': [], 'avg_similarity': 0},
        'transition_smoothness': {'scores': [], 'avg_score': 0}
    }
    
    # 1. Cohesion Scoring
    cohesive_words = ['however', 'therefore', 'thus', 'meanwhile']
    transition_count = 0
    
    for sent in sentences:
        for token in sent:
            if token.text.lower() in cohesive_words:
                transition_count += 1
                analysis['cohesion']['transitions'][token.text.lower()] += 1
    
    analysis['cohesion']['score'] = float(transition_count / len(sentences)) if sentences else 0.0
    
    # 2. Referential Clarity
    pronouns = ['this', 'that', 'these', 'those', 'it', 'they']
    reference_phrases = ['this', 'that', 'these', 'those']
    link_count = 0
    unclear_references = []
    
    for i, sent in enumerate(sentences):
        for token in sent:
            if token.text.lower() in pronouns:
                link_count += 1
                # Check if reference is clear
                if not any(np.dot(token.vector, prev_token.vector) > 0.6 
                          for prev_sent in sentences[:i] 
                          for prev_token in prev_sent):
                    unclear_references.append((i, token.text))
            
            # Check for reference phrases ("this idea")
            if token.text.lower() in reference_phrases and i > 0:
                head_noun = next((child for child in token.children 
                                if child.dep_ in ('attr', 'dobj')), None)
                if head_noun:
                    analysis['referential_clarity']['references'].append(
                        f"{token.text} {head_noun.text}"
                    )
    
    analysis['referential_clarity']['score'] = float((link_count - len(unclear_references)) / len(sentences) if sentences else 0.0)
    
    # 3. Topic Drift
    if len(paragraphs) > 1:
        similarities = []
        for i in range(len(paragraphs)-1):
            doc1 = nlp(paragraphs[i])
            doc2 = nlp(paragraphs[i+1])
            sim = doc1.similarity(doc2)
            similarities.append(float(sim))  # Convert to Python float
            if sim < 0.6:
                analysis['topic_drift']['warnings'].append(
                    f"Low similarity ({float(sim):.2f}) between paragraphs {i+1}-{i+2}"
                )
        analysis['topic_drift']['avg_similarity'] = float(sum(similarities)/len(similarities)) if similarities else 1.0
    
    # 4. Transition Smoothness
    if len(sentences) > 1:
        smoothness_scores = []
        for i in range(len(sentences)-1):
            # Get last word of current sentence and first word of next
            last_word = sentences[i][-1]
            first_word = sentences[i+1][0]
            
            # Calculate cosine similarity between embeddings
            if last_word.has_vector and first_word.has_vector:
                sim = cosine_similarity(
                    last_word.vector.reshape(1, -1), 
                    first_word.vector.reshape(1, -1)
                )[0][0]
                smoothness_scores.append(float(sim))  # Convert to Python float
        
        analysis['transition_smoothness']['scores'] = smoothness_scores
        analysis['transition_smoothness']['avg_score'] = float(sum(smoothness_scores)/len(smoothness_scores)) if smoothness_scores else 1.0
    
    return analysis

def convert_flow_data(flow_data):
    if not isinstance(flow_data, dict):
        return flow_data
        
    for key in ['assess', 'baseline1', 'baseline2']:
        if key in flow_data:
            if 'cohesion' in flow_data[key] and isinstance(flow_data[key]['cohesion']['transitions'], defaultdict):
                flow_data[key]['cohesion']['transitions'] = dict(flow_data[key]['cohesion']['transitions'])
    return flow_data

def analyze_semantic_flow(assess, baseline1, baseline2):
    flow={
        "assess":semantic_flow(assess),
        "baseline1":semantic_flow(baseline1),
        "baseline2":semantic_flow(baseline2)
    }
    return convert_flow_data(flow)




# Disruption Index
def calculate_baseline_stats(text_samples):
    """
    Calculate baseline stylometric statistics from multiple sample texts.
    
    Args:
        text_samples: List of texts representing the author's normal writing style
    
    Returns:
        Dictionary containing baseline statistics
    """
    stats = {
        'sentence_lengths': [],
        'punctuation_freq': defaultdict(list),
        'gpt_phrase_density': [],
        'avg_sentence_length': 0,
        'punctuation_ranges': {},
        'avg_phrase_density': 0
    }
    
    # Process each sample text
    for text in text_samples:
        # Sentence length analysis
        sentences = get_sentences(text)
        stats['sentence_lengths'].extend([len(sent.split()) for sent in sentences])
        
        # Punctuation analysis
        punct_counts = punctuation_counts(text)
        for p, count in punct_counts.items():
            stats['punctuation_freq'][p].append(count / len(text))
        
        # GPT-style phrase analysis
        gpt_phrases = gpt_style_phrases(text)
        stats['gpt_phrase_density'].append(len(gpt_phrases) / max(1, len(sentences)))
    
    # Calculate baseline averages and ranges
    stats['avg_sentence_length'] = np.mean(stats['sentence_lengths'])
    stats['sentence_length_range'] = np.ptp(stats['sentence_lengths'])  # Peak-to-peak range
    
    for p in stats['punctuation_freq']:
        stats['punctuation_ranges'][p] = {
            'avg': np.mean(stats['punctuation_freq'][p]),
            'range': np.ptp(stats['punctuation_freq'][p])
        }
    
    stats['avg_phrase_density'] = np.mean(stats['gpt_phrase_density'])
    
    return stats

def calculate_disruption_index(new_text, baseline_stats, threshold=0.25):
    """
    Calculate stylometric disruption index for new text compared to baseline.
    
    Args:
        new_text: Text to analyze
        baseline_stats: Pre-calculated baseline statistics
        threshold: Deviation threshold for flagging (default 25%)
    
    Returns:
        Dictionary with disruption scores and flags
    """
    results = {
        'metrics': {},
        'deviations': {},
        'flags': []
    }
    
    # Analyze new text
    sentences = get_sentences(new_text)
    new_lengths = [len(sent.split()) for sent in sentences]
    punct_counts = punctuation_counts(new_text)
    gpt_phrases = gpt_style_phrases(new_text)
    
    # Calculate metrics for new text
    results['metrics']['avg_sentence_length'] = np.mean(new_lengths) if new_lengths else 0
    results['metrics']['sentence_length_range'] = np.ptp(new_lengths) if new_lengths else 0
    results['metrics']['punctuation_freq'] = {p: count/len(new_text) for p, count in punct_counts.items()}
    results['metrics']['gpt_phrase_density'] = len(gpt_phrases) / max(1, len(sentences))
    
    # Calculate deviations from baseline
    # Sentence length deviation
    length_dev = abs(results['metrics']['avg_sentence_length'] - baseline_stats['avg_sentence_length']) / baseline_stats['avg_sentence_length']
    results['deviations']['sentence_length'] = length_dev
    
    # Punctuation deviation
    punct_devs = {}
    for p, freq in results['metrics']['punctuation_freq'].items():
        if p in baseline_stats['punctuation_ranges']:
            base_avg = baseline_stats['punctuation_ranges'][p]['avg']
            punct_devs[p] = abs(freq - base_avg) / base_avg
    results['deviations']['punctuation'] = punct_devs
    
    # GPT phrase density deviation
    phrase_dev = abs(results['metrics']['gpt_phrase_density'] - baseline_stats['avg_phrase_density']) / baseline_stats['avg_phrase_density']
    results['deviations']['gpt_phrase_density'] = phrase_dev
    
    # Check for threshold breaches
    if length_dev > threshold:
        results['flags'].append(f"Sentence length deviation {length_dev:.0%} > threshold")
    
    for p, dev in punct_devs.items():
        if dev > threshold:
            results['flags'].append(f"Punctuation '{p}' usage deviation {dev:.0%} > threshold")
    
    if phrase_dev > threshold:
        results['flags'].append(f"GPT-style phrase density deviation {phrase_dev:.0%} > threshold")
    
    # Calculate overall disruption index (weighted average of deviations)
    deviations = [
        length_dev,
        np.mean(list(punct_devs.values())) if punct_devs else 0,
        phrase_dev
    ]
    results['disruption_index'] = np.mean(deviations)
    
    if results['disruption_index'] > threshold:
        results['flags'].append("STYLOMETRIC DISRUPTION DETECTED")
    
    return results

# Example usage
# if __name__ == "__main__":
#     # 1. Establish baseline from sample texts
#     baseline_texts = [
#         "This is a sample text. It shows normal writing style!",
#         "Another example. With typical punctuation? And normal sentence length variation.",
#         "The quick brown fox. Jumps over the lazy dog. Standard stuff."
#     ]
    
#     baseline_stats = calculate_baseline_stats(baseline_texts)
    
#     # 2. Analyze new text
#     new_text = """In conclusion, this means that we should act. Therefore, immediate action is required! 
#                Furthermore, it is important to note the consequences. However, on the other hand..."""
    
#     analysis = calculate_disruption_index(new_text, baseline_stats)
    
#     print("Baseline Statistics:")
#     print(baseline_stats)
#     print("\nDisruption Analysis:")
#     for key, value in analysis.items():
#         print(f"{key}: {value}")





# PGFI: Prufia GPT-Fingerprint Index
# def detect_gpt_patterns(text, baseline_metrics=None):
#     """
#     Detects structural and rhythm-based similarities to GPT-generated content.
    
#     Args:
#         text: Input text to analyze
#         baseline_metrics: Dictionary of baseline metrics for comparison
#                          (if not provided, uses default thresholds)
    
#     Returns:
#         Dictionary containing detection results and flags
#     """
#     # Default baseline thresholds if not provided
#     if baseline_metrics is None:
#         baseline_metrics = {
#             'phrase_repetition_threshold': 3,
#             'semantic_variance_threshold': 0.15,
#             'transition_density_threshold': 0.25
#         }
    
#     # Initialize analysis results
#     analysis = {
#         'gpt_phrases': [],
#         'phrase_repetition_score': 0,
#         'transition_density': 0,
#         'semantic_variance': 0,
#         'flags': []
#     }
    
#     # 1. Detect common GPT-style phrases
#     analysis['gpt_phrases'] = gpt_style_phrases(text)
    
#     # 2. Calculate phrase repetition density
#     phrase_counts = Counter(analysis['gpt_phrases'])
#     total_phrases = len(analysis['gpt_phrases'])
#     analysis['phrase_repetition_score'] = sum(phrase_counts.values()) / max(1, total_phrases)
    
#     # 3. Calculate transition density (using enhanced detection)
#     transitions = [
#         "in conclusion", "therefore", "thus", "hence", "as a result",
#         "furthermore", "moreover", "additionally", "however", "nevertheless",
#         "on the other hand", "in summary", "to sum up", "this means that",
#         "it is important to note"
#     ]
#     words = text.lower().split()
#     transition_words = [word for word in words if word in transitions]
#     analysis['transition_density'] = len(transition_words) / max(1, len(words))
    
#     # 4. Analyze semantic variance (requires spaCy)
#     try:
#         doc = analyzer.get_model('lg')(text)
#         sentence_vectors = [sent.vector for sent in doc.sents if sent.has_vector]
#         if len(sentence_vectors) > 1:
#             # Calculate variance between sentence vectors
#             mean_vector = sum(sentence_vectors) / len(sentence_vectors)
#             variances = [sum((v - mean_vector)**2) for v in sentence_vectors]
#             analysis['semantic_variance'] = sum(variances) / len(variances)
#     except Exception as e:
#         print(f"Semantic analysis error: {e}")
#         analysis['semantic_variance'] = 0
    
#     # 5. Apply detection rules
#     if (analysis['phrase_repetition_score'] > baseline_metrics['phrase_repetition_threshold'] and
#         analysis['semantic_variance'] < baseline_metrics['semantic_variance_threshold']):
#         analysis['flags'].append("FIA-SIG-G05: High phrase repetition with low semantic variance")
    
#     if analysis['transition_density'] > baseline_metrics['transition_density_threshold']:
#         analysis['flags'].append("FIA-SIG-G06: Excessive transition density")
    
#     # 6. Additional rhythm analysis
#     sentence_lengths = [len(sent.split()) for sent in get_sentences(text)]
#     if len(sentence_lengths) > 3:
#         length_variance = sum((x - sum(sentence_lengths)/len(sentence_lengths))**2 for x in sentence_lengths)
#         if length_variance < 5:  # Very consistent sentence lengths
#             analysis['flags'].append("FIA-SIG-G07: Unnaturally consistent sentence rhythm")
    
#     return analysis




def detect_gpt_patterns(text, baseline1=None, baseline2=None):
    """
    Detects structural and rhythm-based similarities to GPT-generated content.
    Uses baseline presets based on string parameters rather than direct metrics.
    
    Args:
        text: Input text to analyze
        baseline1: String indicating primary baseline profile ("strict", "moderate", "lenient")
        baseline2: String indicating secondary baseline profile ("academic", "creative", "technical")
    
    Returns:
        Dictionary containing detection results and flags
    """
    # Define baseline presets
    baseline_presets = {
        # Primary strictness levels
        'strict': {
            'phrase_repetition_threshold': 2,
            'semantic_variance_threshold': 0.1,
            'transition_density_threshold': 0.2
        },
        'moderate': {
            'phrase_repetition_threshold': 3,
            'semantic_variance_threshold': 0.15,
            'transition_density_threshold': 0.25
        },
        'lenient': {
            'phrase_repetition_threshold': 4,
            'semantic_variance_threshold': 0.2,
            'transition_density_threshold': 0.3
        },
        
        # Domain-specific adjustments
        'academic': {
            'transition_density_threshold': 0.3,  # Higher tolerance for transitions
            'semantic_variance_threshold': 0.12    # Lower variance expected
        },
        'creative': {
            'phrase_repetition_threshold': 4,      # More repetition allowed
            'semantic_variance_threshold': 0.25   # Higher variance expected
        },
        'technical': {
            'phrase_repetition_threshold': 2,      # Less repetition allowed
            'transition_density_threshold': 0.2    # Fewer transitions
        }
    }
    
    # Merge baselines if provided
    baseline_metrics = baseline_presets['moderate']  # Default
    
    if baseline1 in baseline_presets:
        baseline_metrics.update(baseline_presets[baseline1])
    
    if baseline2 in baseline_presets:
        baseline_metrics.update(baseline_presets[baseline2])
    
    # Initialize analysis results
    analysis = {
        'gpt_phrases': [],
        'phrase_repetition_score': 0,
        'transition_density': 0,
        'semantic_variance': 0,
        'flags': [],
        'used_baseline': {
            'primary': baseline1 if baseline1 else 'moderate',
            'secondary': baseline2 if baseline2 else 'none'
        }
    }
    
    # 1. Detect common GPT-style phrases
    analysis['gpt_phrases'] = gpt_style_phrases(text)
    
    # 2. Calculate phrase repetition density
    phrase_counts = Counter(analysis['gpt_phrases'])
    total_phrases = len(analysis['gpt_phrases'])
    analysis['phrase_repetition_score'] = sum(phrase_counts.values()) / max(1, total_phrases)
    
    # 3. Calculate transition density
    transitions = [
        "in conclusion", "therefore", "thus", "hence", "as a result",
        "furthermore", "moreover", "additionally", "however", "nevertheless",
        "on the other hand", "in summary", "to sum up", "this means that",
        "it is important to note"
    ]
    words = text.lower().split()
    transition_words = [word for word in words if word in transitions]
    analysis['transition_density'] = len(transition_words) / max(1, len(words))
    
    # 4. Analyze semantic variance (simplified without spaCy)
    sentences = get_sentences(text)
    if len(sentences) > 1:
        # Simple word-based variance approximation
        word_counts = [len(sent.split()) for sent in sentences]
        avg_word_count = sum(word_counts) / len(word_counts)
        analysis['semantic_variance'] = sum(
            (count - avg_word_count)**2 for count in word_counts
        ) / len(word_counts)
    else:
        analysis['semantic_variance'] = 0
    
    # 5. Apply detection rules
    if (analysis['phrase_repetition_score'] > baseline_metrics['phrase_repetition_threshold'] and
        analysis['semantic_variance'] < baseline_metrics['semantic_variance_threshold']):
        analysis['flags'].append("FIA-SIG-G05: High phrase repetition with low semantic variance")
    
    if analysis['transition_density'] > baseline_metrics['transition_density_threshold']:
        analysis['flags'].append("FIA-SIG-G06: Excessive transition density")
    
    # 6. Additional rhythm analysis
    sentence_lengths = [len(sent.split()) for sent in sentences]
    if len(sentence_lengths) > 3:
        length_variance = sum((x - sum(sentence_lengths)/len(sentence_lengths))**2 for x in sentence_lengths)
        if length_variance < 5:
            analysis['flags'].append("FIA-SIG-G07: Unnaturally consistent sentence rhythm")
    
    return analysis

# Helper functions
def gpt_style_phrases(text):
    """Identify common GPT-style phrases"""
    phrases = [
        "as an AI language model",
        "it's important to note that",
        "in the realm of",
        "when it comes to",
        "it's worth mentioning that",
        "in today's world",
        "in this digital age",
        "the intricate tapestry of",
        "on a journey through",
        "delve deeper into",
        "navigate the complexities of",
        "unpack the nuances of"
    ]
    return [phrase for phrase in phrases if phrase in text.lower()]

def get_sentences(text):
    """Basic sentence splitting"""
    import re
    return [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]








def detect_grammar_fixes_only(submission, baseline):
    import difflib
    diff = list(difflib.ndiff(baseline.split(), submission.split()))
    changes = [d for d in diff if d.startswith('+ ') or d.startswith('- ')]
    punctuation = {'.', ',', ';', ':', '?', '!', '"', "'", '(', ')'}
    for change in changes:
        token = change[2:].strip()
        if token.lower() != token.upper() and token not in punctuation:
            return False
    return True
    

def detect_minor_edits(submission, baseline):
    ratio = SequenceMatcher(None, baseline, submission).ratio()
    return 0.8 <= ratio < 0.95

def detect_structural_changes(submission, baseline):
    baseline_sentences = baseline.split('.')
    submission_sentences = submission.split('.')
    return abs(len(baseline_sentences) - len(submission_sentences)) > 2

def detect_major_rewrite(submission, baseline):
    ratio = SequenceMatcher(None, baseline, submission).ratio()
    return ratio < 0.5

def detect_behavioral_inconsistency(submission_behavior, baseline_behavior):
    baseline_speed = baseline_behavior.get('typing_speed', 0)
    submission_speed = submission_behavior.get('typing_speed', 0)
    return abs(baseline_speed - submission_speed) > 30


