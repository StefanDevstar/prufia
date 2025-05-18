from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def read_texts(folder):
    return [open(os.path.join(folder, f), encoding='utf-8').read()
            for f in os.listdir(folder) if f.endswith('.txt')]

def calculate_stylometry_score(baseline_folder, new_file):
    base = read_texts(baseline_folder)
    new = open(new_file, encoding='utf-8').read()
    tfidf = TfidfVectorizer()
    mat = tfidf.fit_transform(base + [new])
    return round(cosine_similarity(mat[-1], mat[:-1]).mean(), 4)

def calculate_keystroke_score(baseline_folder, new_file):
    return 0.75  # placeholder

def calculate_combined_score(baseline_folder, new_file):
    s = calculate_stylometry_score(baseline_folder, new_file)
    k = calculate_keystroke_score(baseline_folder, new_file)
    return round((s + k) / 2, 4)
