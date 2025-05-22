from difflib import SequenceMatcher
from collections import Counter, defaultdict
import math
import spacy
import string
import nltk
from textstat import textstat
from nltk.tokenize import sent_tokenize
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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



# text1 = "hello hello hello"  # Low entropy (repetitive)
# text2 = "the quick brown fox"  # Higher entropy (all unique)
# print(word_entropy(text1))  # Output: 0.0
# print(word_entropy(text2))  # Output: 2.0
def word_entropy(text):
    words = text.split()
    freq = Counter(words)
    total = sum(freq.values())
    return -sum((count/total) * math.log2(count/total) for count in freq.values())


# text = "The book was read by the student. The student enjoyed it."
# print(passive_voice_ratio(text))  # Output: 0.5 (1 passive out of 2 sentences)
def passive_voice_ratio(text):
    nlp = analyzer.get_model('sm')
    doc = nlp(text)
    passive_sentences = [sent for sent in doc.sents if any(tok.dep_ == "nsubjpass" for tok in sent)]
    return len(passive_sentences) / len(list(doc.sents))

# text = "Hello, world! How's it going?"
# punctuation_counts(text)
# Output:{',': 1, '!': 1, "'": 1, '?': 1}
def punctuation_counts(text):
    return {p: text.count(p) for p in string.punctuation if text.count(p) > 0}


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
def analyze_opening_closing(text, num_sentences=3):
    nlp = analyzer.get_model('lg')
    """
    Analyze consistency between opening and closing sections of text.
    
    Args:
        text: Input text to analyze
        num_sentences: Number of sentences to consider as opening/closing
        
    Returns:
        Dictionary containing comparison metrics
    """
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    if len(sentences) < num_sentences*2:
        return {"error": f"Text too short - needs at least {num_sentences*2} sentences"}
    
    opening = sentences[:num_sentences]
    closing = sentences[-num_sentences:]
    
    # Process sections
    def process_section(section):
        combined = " ".join(section)
        section_doc = nlp(combined)
        
        return {
            "avg_sentence_length": sum(len(s.split()) for s in section)/num_sentences,
            "avg_word_length": sum(len(token.text) for token in section_doc)/len(list(section_doc)),
            "pos_distribution": defaultdict(int, 
                [(token.pos_, token.pos) for token in section_doc]),
            "readability": textstat.flesch_reading_ease(combined),
            "lexical_diversity": len(set(token.text.lower() for token in section_doc))/len(list(section_doc)),
            "tone": analyze_tone(combined)
        }
    
    opening_stats = process_section(opening)
    closing_stats = process_section(closing)
    
    # Calculate similarity scores
    def cosine_sim(vec1, vec2):
        return vec1.similarity(vec2) if vec1.has_vector and vec2.has_vector else 0
        
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
            "pos_correlation": sum(
                min(opening_stats["pos_distribution"][pos], closing_stats["pos_distribution"][pos])
                for pos in set(opening_stats["pos_distribution"]) | set(closing_stats["pos_distribution"])
            ) / (len(list(opening_doc)) + len(list(closing_doc)))
        }
    }

def analyze_tone(text):
    """Simple tone analyzer (expand with more sophisticated NLP)"""
    nlp = analyzer.get_model('lg')
    doc = nlp(text)
    
    # Count formal/informal markers
    formal = sum(1 for token in doc if token.text.lower() in ["furthermore", "moreover", "however"])
    informal = sum(1 for token in doc if token.text.lower() in ["so", "well", "you know"])
    
    questions = sum(1 for sent in doc.sents if sent.text.endswith("?"))
    
    if questions > 1:
        return "interrogative"
    elif formal > informal:
        return "formal"
    elif informal > formal:
        return "informal"
    return "neutral"


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

def analyze_semantic_flow(text):
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
    
    analysis['cohesion']['score'] = transition_count / len(sentences) if sentences else 0
    
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
    
    analysis['referential_clarity']['score'] = (link_count - len(unclear_references)) / len(sentences) if sentences else 0
    
    # 3. Topic Drift
    if len(paragraphs) > 1:
        similarities = []
        for i in range(len(paragraphs)-1):
            doc1 = nlp(paragraphs[i])
            doc2 = nlp(paragraphs[i+1])
            sim = doc1.similarity(doc2)
            similarities.append(sim)
            if sim < 0.6:
                analysis['topic_drift']['warnings'].append(
                    f"Low similarity ({sim:.2f}) between paragraphs {i+1}-{i+2}"
                )
        analysis['topic_drift']['avg_similarity'] = sum(similarities)/len(similarities) if similarities else 1
    
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
                smoothness_scores.append(sim)
        
        analysis['transition_smoothness']['scores'] = smoothness_scores
        analysis['transition_smoothness']['avg_score'] = sum(smoothness_scores)/len(smoothness_scores) if smoothness_scores else 1
    
    return analysis

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
def detect_gpt_patterns(text, baseline_metrics=None):
    """
    Detects structural and rhythm-based similarities to GPT-generated content.
    
    Args:
        text: Input text to analyze
        baseline_metrics: Dictionary of baseline metrics for comparison
                         (if not provided, uses default thresholds)
    
    Returns:
        Dictionary containing detection results and flags
    """
    # Default baseline thresholds if not provided
    if baseline_metrics is None:
        baseline_metrics = {
            'phrase_repetition_threshold': 3,
            'semantic_variance_threshold': 0.15,
            'transition_density_threshold': 0.25
        }
    
    # Initialize analysis results
    analysis = {
        'gpt_phrases': [],
        'phrase_repetition_score': 0,
        'transition_density': 0,
        'semantic_variance': 0,
        'flags': []
    }
    
    # 1. Detect common GPT-style phrases
    analysis['gpt_phrases'] = gpt_style_phrases(text)
    
    # 2. Calculate phrase repetition density
    phrase_counts = Counter(analysis['gpt_phrases'])
    total_phrases = len(analysis['gpt_phrases'])
    analysis['phrase_repetition_score'] = sum(phrase_counts.values()) / max(1, total_phrases)
    
    # 3. Calculate transition density (using enhanced detection)
    transitions = [
        "in conclusion", "therefore", "thus", "hence", "as a result",
        "furthermore", "moreover", "additionally", "however", "nevertheless",
        "on the other hand", "in summary", "to sum up", "this means that",
        "it is important to note"
    ]
    words = text.lower().split()
    transition_words = [word for word in words if word in transitions]
    analysis['transition_density'] = len(transition_words) / max(1, len(words))
    
    # 4. Analyze semantic variance (requires spaCy)
    try:
        doc = analyzer.get_model('lg')(text)
        sentence_vectors = [sent.vector for sent in doc.sents if sent.has_vector]
        if len(sentence_vectors) > 1:
            # Calculate variance between sentence vectors
            mean_vector = sum(sentence_vectors) / len(sentence_vectors)
            variances = [sum((v - mean_vector)**2) for v in sentence_vectors]
            analysis['semantic_variance'] = sum(variances) / len(variances)
    except Exception as e:
        print(f"Semantic analysis error: {e}")
        analysis['semantic_variance'] = 0
    
    # 5. Apply detection rules
    if (analysis['phrase_repetition_score'] > baseline_metrics['phrase_repetition_threshold'] and
        analysis['semantic_variance'] < baseline_metrics['semantic_variance_threshold']):
        analysis['flags'].append("FIA-SIG-G05: High phrase repetition with low semantic variance")
    
    if analysis['transition_density'] > baseline_metrics['transition_density_threshold']:
        analysis['flags'].append("FIA-SIG-G06: Excessive transition density")
    
    # 6. Additional rhythm analysis
    sentence_lengths = [len(sent.split()) for sent in get_sentences(text)]
    if len(sentence_lengths) > 3:
        length_variance = sum((x - sum(sentence_lengths)/len(sentence_lengths))**2 for x in sentence_lengths)
        if length_variance < 5:  # Very consistent sentence lengths
            analysis['flags'].append("FIA-SIG-G07: Unnaturally consistent sentence rhythm")
    
    return analysis



detect_gpt_patterns("But the fox brown jumps over lazy dog")


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


