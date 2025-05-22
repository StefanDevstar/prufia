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

# def analyze_semantic_flow(text):
#     nlp = analyzer.get_model('lg')
#     doc = nlp(text)
#     sentences = [sent for sent in doc.sents]
#     paragraphs = text.split('\n\n') if '\n\n' in text else [text]
    
#     analysis = {
#         'cohesion': {'score': 0, 'transitions': defaultdict(int)},
#         'referential_clarity': {'score': 0, 'references': []},
#         'topic_drift': {'warnings': [], 'avg_similarity': 0},
#         'transition_smoothness': {'scores': [], 'avg_score': 0}
#     }
    
#     # 1. Cohesion Scoring
#     cohesive_words = ['however', 'therefore', 'thus', 'meanwhile']
#     transition_count = 0
    
#     for sent in sentences:
#         for token in sent:
#             if token.text.lower() in cohesive_words:
#                 transition_count += 1
#                 analysis['cohesion']['transitions'][token.text.lower()] += 1
    
#     analysis['cohesion']['score'] = transition_count / len(sentences) if sentences else 0
    
#     # 2. Referential Clarity
#     pronouns = ['this', 'that', 'these', 'those', 'it', 'they']
#     reference_phrases = ['this', 'that', 'these', 'those']
#     link_count = 0
#     unclear_references = []
    
#     for i, sent in enumerate(sentences):
#         for token in sent:
#             if token.text.lower() in pronouns:
#                 link_count += 1
#                 # Check if reference is clear
#                 if not any(np.dot(token.vector, prev_token.vector) > 0.6 
#                           for prev_sent in sentences[:i] 
#                           for prev_token in prev_sent):
#                     unclear_references.append((i, token.text))
            
#             # Check for reference phrases ("this idea")
#             if token.text.lower() in reference_phrases and i > 0:
#                 head_noun = next((child for child in token.children 
#                                 if child.dep_ in ('attr', 'dobj')), None)
#                 if head_noun:
#                     analysis['referential_clarity']['references'].append(
#                         f"{token.text} {head_noun.text}"
#                     )
    
#     analysis['referential_clarity']['score'] = (link_count - len(unclear_references)) / len(sentences) if sentences else 0
    
#     # 3. Topic Drift
#     if len(paragraphs) > 1:
#         similarities = []
#         for i in range(len(paragraphs)-1):
#             doc1 = nlp(paragraphs[i])
#             doc2 = nlp(paragraphs[i+1])
#             sim = doc1.similarity(doc2)
#             similarities.append(sim)
#             if sim < 0.6:
#                 analysis['topic_drift']['warnings'].append(
#                     f"Low similarity ({sim:.2f}) between paragraphs {i+1}-{i+2}"
#                 )
#         analysis['topic_drift']['avg_similarity'] = sum(similarities)/len(similarities) if similarities else 1
    
#     # 4. Transition Smoothness
#     if len(sentences) > 1:
#         smoothness_scores = []
#         for i in range(len(sentences)-1):
#             # Get last word of current sentence and first word of next
#             last_word = sentences[i][-1]
#             first_word = sentences[i+1][0]
            
#             # Calculate cosine similarity between embeddings
#             if last_word.has_vector and first_word.has_vector:
#                 sim = cosine_similarity(
#                     last_word.vector.reshape(1, -1), 
#                     first_word.vector.reshape(1, -1)
#                 )[0][0]
#                 smoothness_scores.append(sim)
        
#         analysis['transition_smoothness']['scores'] = smoothness_scores
#         analysis['transition_smoothness']['avg_score'] = sum(smoothness_scores)/len(smoothness_scores) if smoothness_scores else 1
    
#     return analysis


from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def analyze_discourse_flow(assess_text, baseline1_text, baseline2_text):
    """
    Compare discourse flow characteristics between assessment text and baselines
    Analyzes transition patterns, referential chains, and logical flow
    Returns comparative analysis with metrics and warnings
    """
    nlp = analyzer.get_model('lg')
    
    def analyze_text(text):
        """Core analysis function for individual texts"""
        if not text or not isinstance(text, str):
            return {"error": "Invalid text input"}
            
        doc = nlp(text)
        sentences = [sent for sent in doc.sents]
        
        analysis = {
            'transition_quality': {
                'explicit_transitions': defaultdict(int),
                'implicit_transitions': [],
                'score': 0
            },
            'referential_chains': {
                'pronoun_resolution': [],
                'entity_continuity': [],
                'score': 0
            },
            'dependency_chains': {
                'logical_connectors': [],
                'argument_flow': [],
                'score': 0
            },
            'coherence_metrics': {
                'paragraph_similarity': 0,
                'sentence_flow': 0,
                'topic_consistency': 0
            }
        }
        
        # 1. Transition Analysis
        explicit_markers = {
            'contrast': ['however', 'although', 'nevertheless'],
            'cause': ['because', 'since', 'therefore'],
            'addition': ['furthermore', 'moreover', 'additionally']
        }
        
        implicit_transitions = 0
        for i in range(1, len(sentences)):
            prev_sent = sentences[i-1]
            curr_sent = sentences[i]
            
            # Check for explicit transition markers
            found_marker = False
            for marker_type in explicit_markers:
                for marker in explicit_markers[marker_type]:
                    if marker in curr_sent.text.lower():
                        analysis['transition_quality']['explicit_transitions'][marker_type] += 1
                        found_marker = True
                        break
            
            # Analyze implicit transitions using semantic similarity
            if not found_marker:
                sim = prev_sent.similarity(curr_sent)
                analysis['transition_quality']['implicit_transitions'].append(sim)
                if sim > 0.6:
                    implicit_transitions += 1
        
        analysis['transition_quality']['score'] = (
            sum(analysis['transition_quality']['explicit_transitions'].values()) + implicit_transitions
        ) / len(sentences) if sentences else 0
        
        # 2. Referential Chain Analysis
        entities = defaultdict(list)
        unresolved_pronouns = []
        
        for i, sent in enumerate(sentences):
            for ent in sent.ents:
                entities[ent.label_].append((i, ent.text))
            
            for token in sent:
                if token.pos_ == 'PRON' and token.text.lower() in ['it', 'they', 'this', 'that']:
                    referent_found = False
                    # Look for antecedents in previous sentences
                    for j in range(i-1, max(-1, i-3), -1):
                        if any(np.dot(token.vector, t.vector) > 0.7 for t in sentences[j]):
                            referent_found = True
                            break
                    if not referent_found:
                        unresolved_pronouns.append((i, token.text))
        
        analysis['referential_chains']['pronoun_resolution'] = unresolved_pronouns
        analysis['referential_chains']['score'] = 1 - (len(unresolved_pronouns) / sum(1 for s in sentences for t in s if t.pos_ == 'PRON')) if sentences else 1
        
        # 3. Dependency Chain Analysis
        logical_connectors = []
        argument_flow = []
        
        for sent in sentences:
            for token in sent:
                if token.dep_ in ['mark', 'cc'] and token.head.dep_ in ['ROOT', 'conj']:
                    logical_connectors.append((token.text, token.head.text))
                
                # Track subject-verb-object chains
                if token.dep_ in ['nsubj', 'dobj']:
                    argument_flow.append((token.text, token.head.text))
        
        analysis['dependency_chains']['logical_connectors'] = logical_connectors
        analysis['dependency_chains']['argument_flow'] = argument_flow
        analysis['dependency_chains']['score'] = len(logical_connectors) / len(sentences) if sentences else 0
        
        # 4. Coherence Metrics
        if len(sentences) > 1:
            sentence_sims = []
            for i in range(len(sentences)-1):
                sentence_sims.append(sentences[i].similarity(sentences[i+1]))
            analysis['coherence_metrics']['sentence_flow'] = np.mean(sentence_sims) if sentence_sims else 1
        
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) > 1:
            para_sims = []
            for i in range(len(paragraphs)-1):
                para_sims.append(nlp(paragraphs[i]).similarity(nlp(paragraphs[i+1])))
            analysis['coherence_metrics']['paragraph_similarity'] = np.mean(para_sims) if para_sims else 1
        
        return analysis
    
    # Analyze all three texts
    assess_results = analyze_text(assess_text)
    baseline1_results = analyze_text(baseline1_text)
    baseline2_results = analyze_text(baseline2_text)
    
    # Comparative analysis
    def compare_metric(metric_path):
        """Helper to compare metrics across texts"""
        assess_val = assess_results
        base1_val = baseline1_results
        base2_val = baseline2_results
        
        for key in metric_path.split('.'):
            assess_val = assess_val.get(key, 0) if isinstance(assess_val, dict) else 0
            base1_val = base1_val.get(key, 0) if isinstance(base1_val, dict) else 0
            base2_val = base2_val.get(key, 0) if isinstance(base2_val, dict) else 0
        
        baseline_avg = (base1_val + base2_val) / 2 if (base1_val + base2_val) > 0 else 0
        return {
            'assessment': assess_val,
            'baseline_avg': baseline_avg,
            'difference': assess_val - baseline_avg,
            'z_score': (assess_val - baseline_avg) / np.std([base1_val, base2_val]) if np.std([base1_val, base2_val]) != 0 else 0
        }
    
    results = {
        'assessment': assess_results,
        'baselines': {
            'baseline1': baseline1_results,
            'baseline2': baseline2_results
        },
        'comparison': {
            'transition_quality': compare_metric('transition_quality.score'),
            'referential_chains': compare_metric('referential_chains.score'),
            'dependency_chains': compare_metric('dependency_chains.score'),
            'sentence_coherence': compare_metric('coherence_metrics.sentence_flow'),
            'paragraph_coherence': compare_metric('coherence_metrics.paragraph_similarity')
        },
        'anomalies': [],
        'recommendations': []
    }
    
    # Detect anomalies
    transition_z = results['comparison']['transition_quality']['z_score']
    if transition_z < -1.5:
        results['anomalies'].append(
            "Low transition quality (z = %.2f) - Possible lack of logical connectors" % transition_z
        )
    
    ref_z = results['comparison']['referential_chains']['z_score']
    if ref_z < -2:
        results['anomalies'].append(
            "Poor referential chains (z = %.2f) - Many unclear pronoun references" % ref_z
        )
    
    # Generate recommendations
    if len(assess_results.get('referential_chains', {}).get('pronoun_resolution', [])) > 3:
        results['recommendations'].append(
            "Reduce ambiguous pronouns - %d unclear references detected" % 
            len(assess_results['referential_chains']['pronoun_resolution'])
        )
    
    if results['comparison']['dependency_chains']['difference'] < -0.15:
        results['recommendations'].append(
            "Strengthen argument flow - Add more logical connectors between clauses"
        )
    
    if not results['recommendations']:
        results['recommendations'].append("Discourse flow appears natural and well-structured")
    
    return results





baseline1="Machine learning algorithms require careful tuning to achieve optimal performance. This process, known as hyperparameter optimization, involves systematically testing different configurations. Although time-consuming, these adjustments often yield significant improvements in model accuracy.For example, when training a neural network, one must consider both learning rate and batch size. These parameters interact in complex ways that affect training stability. Consequently, practitioners typically use validation sets to guide their tuning decisions."
baseline2="Data preprocessing is a crucial step in any machine learning pipeline. First, you need to handle missing values through imputation or removal. Next comes feature scaling to normalize numerical ranges.Categorical variables require special treatment too. One-hot encoding is commonly used but can lead to dimensionality issues. Sometimes feature engineering provides better solutions than direct encoding."
ssessment="Deep learning models have many layers. The layers process information sequentially. Training requires large datasets. Backpropagation updates the weights. It can be computationally expensive. They often need GPUs. This helps speed up training. The results are sometimes hard to interpret. It remains popular despite this."

print(analyze_discourse_flow(ssessment,baseline1,baseline2))