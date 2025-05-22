from collections import defaultdict
import textstat
import nltk
from typing import Dict, Any
from functools import lru_cache
import spacy


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

assess="Artificial intelligence is transforming modern business operations at an unprecedented pace. ompanies across industries are adopting AI solutions to gain competitive advantages. This technological shift raises important questions about workforce adaptation.[Several paragraphs about AI applications...]In conclusion, businesses must carefully valuate their AI adoption strategies. Are we preparing employees adequately for these changes? The future of work will undoubtedly look very different due to AI."

baseline1="The impact of automation on employment has been widely debated in recent years. Many economists predict significant job market disruptions in the coming decade. This transition period requires thoughtful policy interventions.[Content about historical automation examples...]To summarize, while automation creates new opportunities, it also presents challenges. Policymakers should focus on retraining programs and social safety nets. The human dimension of technological change must remain central to these discussions."

baseline2="Digital transformation initiatives are reshaping organizational structures globally. From small startups to large corporations, technology adoption is accelerating. This trend shows no signs of slowing down in the foreseeable future.[Content about digital tools and platforms...]Ultimately, the pace of change demands agile responses from all stakeholders. How can companies balance innovation with stability? Leaders must navigate this complex landscape with both vision and caution."

print(compare_opening_closing(assess,baseline1, baseline2))