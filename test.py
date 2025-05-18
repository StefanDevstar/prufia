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

def run_tests():
    # Grammar Fixes Only
    assert detect_grammar_fixes_only("The quick brown fox jumps over the lazy dog.", "The quick brown fox jumps over the lazy dog") == True
    assert detect_grammar_fixes_only("She doesn’t like it when it rains", "She dont like it when it rains") == True
    assert detect_grammar_fixes_only("The cat chased the dog through the garden", "The dog chased the cat through the garden") == False
    assert detect_grammar_fixes_only("He went to the store earlier", "He is going to the market now") == False

    # Minor Edits
    assert detect_minor_edits("The weather is nice today for walking", "The weather is good today for walking") == True
    assert detect_minor_edits("She completed the report and sent it to her manager", "She wrote the report and emailed it to her boss") == True
    assert detect_minor_edits("They never show up on time for their shift", "He always arrives early for work") == False
    assert detect_minor_edits("The moon orbits the Earth and reflects sunlight", "Technology helps us communicate faster than before") == False

    # Structural Changes
    assert detect_structural_changes("We walked to the park; birds were singing on that nice day.", "We walked to the park. It was a nice day. Birds were singing.") == True
    assert detect_structural_changes("He had a list and bought everything when he went shopping.", "He wanted to go shopping. He had a list. He bought everything.") == True
    assert detect_structural_changes("The stock market rose sharply today amid strong earnings reports.", "He stayed home. He was tired. He went to bed early.") == False
    assert detect_structural_changes("Writing structure shows sentence samples. This is a sample writing.", "This is a writing sample. It shows sentence structure.") == False

    # Major Rewrite
    assert detect_major_rewrite("Yesterday's storm damaged several homes, leaving many without power.", "The economy is expected to grow steadily due to increased consumer spending.") == True
    assert detect_major_rewrite("Freedom of expression is essential in every democratic society.", "Students should submit their assignments on time to receive full credit.") == True
    assert detect_major_rewrite("The cat sat on the window ledge and watched the birds fly past.", "The cat sat on the windowsill watching the birds fly by.") == False
    assert detect_major_rewrite("Their plan was to go to the museum later in the day.", "They planned to visit the museum in the afternoon.") == False

    # Behavioral Inconsistency
    assert detect_behavioral_inconsistency({"typing_speed": 95}, {"typing_speed": 45}) == True
    assert detect_behavioral_inconsistency({"typing_speed": 20}, {"typing_speed": 60}) == True
    assert detect_behavioral_inconsistency({"typing_speed": 58}, {"typing_speed": 55}) == False
    assert detect_behavioral_inconsistency({"typing_speed": 73}, {"typing_speed": 70}) == False

    print("All Prufia scoring function tests passed successfully.")

if __name__ == '__main__':
    run_tests()