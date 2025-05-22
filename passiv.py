from difflib import SequenceMatcher
from collections import Counter, defaultdict
import math
import spacy
import string
import nltk
from textstat import textstat
from nltk.tokenize import sent_tokenize, word_tokenize
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

def passive_voice_ratio(text):
    """Calculate the ratio of passive voice sentences to total sentences"""
    nlp = analyzer.get_model('sm')
    doc = nlp(text)
    sentences = list(doc.sents)
    
    if not sentences:
        return 0.0
    
    passive_count = sum(1 for sent in sentences if any(tok.dep_ == "nsubjpass" for tok in sent))
    return passive_count / len(sentences)

def passive_voice_analysis(assess_text, baseline1_text, baseline2_text):
    """
    Compare passive voice usage in assessment text against two baselines
    Returns a dictionary with analysis results and recommendations
    """
    results = {}
    
    # Calculate ratios
    assess_ratio = passive_voice_ratio(assess_text)
    baseline1_ratio = passive_voice_ratio(baseline1_text)
    baseline2_ratio = passive_voice_ratio(baseline2_text)
    
    # Store raw ratios
    results['passive_voice_ratios'] = {
        'assessment': assess_ratio,
        'baseline1': baseline1_ratio,
        'baseline2': baseline2_ratio
    }
    
    # Calculate comparison metrics
    baseline_avg = (baseline1_ratio + baseline2_ratio) / 2
    deviation_from_avg = assess_ratio - baseline_avg
    
    results['comparison'] = {
        'baseline_average': baseline_avg,
        'deviation_from_average': deviation_from_avg,
        'deviation_percentage': (deviation_from_avg / baseline_avg) * 100 if baseline_avg != 0 else float('inf')
    }
    
    # Generate recommendations
    recommendations = []
    if assess_ratio > baseline_avg * 1.5:  # 50% more than average
        recommendations.append("The text uses significantly more passive voice than the baselines. Consider revising passive constructions to active voice where possible.")
    elif assess_ratio > baseline_avg * 1.2:  # 20% more than average
        recommendations.append("The text uses moderately more passive voice than the baselines. Review if some passive constructions could be made active.")
    elif assess_ratio < baseline_avg * 0.8:  # 20% less than average
        recommendations.append("The text uses less passive voice than the baselines. This is generally good for clarity.")
    else:
        recommendations.append("The passive voice usage is within normal range compared to the baselines.")
    
    results['recommendations'] = recommendations
    
    return results

# Example usage:
if __name__ == "__main__":
    # Example texts
    assess_text = "The book was read by the student. The student enjoyed it. The results were analyzed by the team."
    baseline1_text = "The student read the book. The student enjoyed it. The team analyzed the results."
    baseline2_text = "The experiment was conducted by researchers. Data was collected over several weeks."
    
    analysis = passive_voice_analysis(assess_text, baseline1_text, baseline2_text)
    print(analysis)
    print("Passive Voice Ratios:", analysis['passive_voice_ratios'])
    print("Comparison:", analysis['comparison'])
    print("Recommendations:", analysis['recommendations'])