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

def get_sentences(text):
    return sent_tokenize(text)

def gpt_style_phrases(text):
    patterns = [
        "in conclusion", "as a result", "this means that",
        "it is important to note", "therefore", "in summary",
        "on the other hand", "to sum up"
    ]
    return [phrase for phrase in patterns if phrase in text.lower()]

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
    print("analysis",analysis)
    return analysis



analysis_results = detect_gpt_patterns(
    "Therefore, I am writing to express my interest in the Senior Full-Stack Engineer at signer.com. In Summary, This letter is a test submission to demonstrate my communication skills and attention to detail. Nevertheless, Thank you for considering my application. Therefore, I look forward to the opportunity to discuss how my skills and experience align with your needs.Sincerely,"
)

# Print formatted results
# print("\nGPT Pattern Detection Results:")
# print("="*50)

# # 1. Display basic metrics
# print("\nKey Metrics:")
# print(f"- GPT-style phrases found: {len(analysis_results['gpt_phrases'])}")
# print(f"- Phrase repetition score: {analysis_results['phrase_repetition_score']:.2f}")
# print(f"- Transition density: {analysis_results['transition_density']:.2%}")
# print(f"- Semantic variance: {analysis_results['semantic_variance']:.4f}")

# 2. Show detected GPT phrases if any
# if analysis_results['gpt_phrases']:
#     print("\nDetected GPT-style Phrases:")
#     for phrase in set(analysis_results['gpt_phrases']):  # Remove duplicates
#         print(f"- '{phrase}'")

# 3. Display any flags/warnings
# if analysis_results['flags']:
#     print("\nDetection Flags:")
#     for flag in analysis_results['flags']:
#         print(f"- ⚠️ {flag}")
# else:
#     print("\nNo strong GPT patterns detected")

# print("\n" + "="*50)
