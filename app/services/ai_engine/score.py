from difflib import SequenceMatcher

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