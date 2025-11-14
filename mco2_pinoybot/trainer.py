import csv
import pickle
import os
import re
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

MODEL_PATH = 'pinoybot_model.pkl'
VECTORIZER_PATH = 'pinoybot_vectorizer.pkl'
TARGET_LABELS = ['ENG', 'FIL', 'OTH']

FIL_PREFIXES = (
    'ipag', 'pang', 'mapag', 'pin', 'kin', 'ipa', 'pa', 'ka', 'ma',
    'na', 'tag', 'sang', 'pina', 'naka', 'maka',
    'nakaka', 'mapang', 'mang', 'nang', 'um'
)

FIL_SUFFIXES = (
    'han', 'hin', 'in', 'an', 'ng', 'ong', 'nin', 'non'
)

ENG_PREFIXES = (
    'un', 're', 'dis', 'over', 'mis', 'out', 'pre',
    'under', 'anti', 'de', 'fore', 'inter', 'mid',
    'sub', 'super', 'trans', 'semi', 'auto',
    'co', 'ex', 'extra', 'hyper', 'micro', 'post'
)

ENG_SUFFIXES = (
    'ing', 'ed', 'ly', 'tion', 'sion', 'ness', 'ment',
    'ful', 'less', 'able', 'ible', 'ous', 'ious', 'al',
    'ical', 'er', 'est', 'ize', 'ise', 'ity', 'ty'
)

MIN_ROOT_LEN = 3

ENG_STRONG_LETTERS = set("qxz")
ENG_SOFT_LETTERS = set("vfjc")

OTH_ABBREVIATION_CLUES = {
    'IMO', 'LOL', 'ASAP', 'AFAIK', 'IDK', 'BTW', 'JK', 'BRB', 'DIY', 'TLDR'
}

OTH_EXPRESSION_CLUES = {
    'haha', 'hehe', 'hihi', 'hoho', 'grr', 'argh', 'ugh', 'wow', 'omg',
    'lolz', 'yay', 'huy', 'ay', 'naku', 'ooh', 'ahh', 'ehh', 'heck'
}

FIL_DOUBLE_VOWELS_CLUES = {'aa', 'ii', 'uu'}

ENG_BIGRAM_CLUES = {
    'th', 'sh', 'ch', 'ph', 'wh', 'ck', 'st', 'tr', 'dr', 'pr',
    'pl', 'cl', 'gl', 'fl', 'bl', 'br', 'cr', 'gr'
}

FIL_BIGRAM_CLUES = {
    'ng', 'mg' # you can add more if needed
}

# FEATURES
def check_if_name_entity(word: str, position_in_sentence: int = 0) -> Optional[str]:
    if not word:
        return None
    if position_in_sentence == 0:
        return None
    if word[0].isupper():
        if len(word) > 1:
            return 'OTH'

    return None

def check_if_other(word: str) -> bool:
    if not word:
        return True
    word = word.strip()
    if not word:
        return True
    wl, wu = word.lower(), word.upper()
    if not any(ch.isalnum() for ch in word):
        return True
    if any(ch.isdigit() for ch in word):
        return True
    if wl in OTH_EXPRESSION_CLUES:
        return True
    if wu in OTH_ABBREVIATION_CLUES:
        return True
    if word.isupper() and len(word) >= 2 and word.isalpha():
        return True
    return False

def check_if_filipino(word: str) -> Optional[str]:
    if not word:
        return None
    wl = word.lower()
    if '-' in wl:
        parts = wl.split('-')
        if len(parts) == 2 and parts[0] == parts[1] and len(parts[0]) >= 2:
            return 'FIL'
    for dv in FIL_DOUBLE_VOWELS_CLUES:
        if dv in wl:
            return 'FIL'
    return None

def check_if_english_letters(word: str) -> Optional[str]:
    if not word:
        return None
    wl = word.lower()
    if any(ch in ENG_STRONG_LETTERS for ch in wl):
        return 'ENG'
    if any(ch in ENG_SOFT_LETTERS for ch in wl) and len(wl) > 3:
        return 'ENG'
    return None

def check_if_filipino_bigrams(word: str) -> Optional[str]:
    if not word or len(word) < 2:
        return None
    w = word.lower()
    for bg in FIL_BIGRAM_CLUES:
        if bg in w:
            return 'FIL'
    return None

def check_if_english(word: str) -> Optional[str]:
    if not word or len(word) < 3:
        return None
    wl = word.lower()
    for p in ENG_PREFIXES:
        if wl.startswith(p) and len(wl) > len(p) + 2:
            return 'ENG'
    for s in ENG_SUFFIXES:
        if wl.endswith(s) and len(wl) > len(s) + 2:
            return 'ENG'
    return None

def check_if_english_bigrams(word: str) -> Optional[str]:
    if not word or len(word) < 3:
        return None
    w = word.lower()
    for bg in ENG_BIGRAM_CLUES:
        if bg in w:
            return 'ENG'
    return None

def apply_rules(word: str, position_in_sentence: int = 0) -> Optional[str]:
    if not word:
        return None
    word = word.strip()
    if not word:
        return None
    if check_if_other(word):
        return 'OTH'
    name_entity = check_if_name_entity(word, position_in_sentence)
    if name_entity:
        return name_entity
    fil_bg = check_if_filipino_bigrams(word)
    if fil_bg:
        return fil_bg
    bigram = check_if_english_bigrams(word)
    if bigram:
        return bigram
    fil = check_if_filipino(word)
    if fil:
        return fil
    eng = check_if_english_letters(word)
    if eng:
        return eng
    return None


def map_label_to_target(raw_label: str) -> str:
    label = raw_label.upper().strip()
    tags = label.split('-')
    if 'FIL' in tags:
        return 'FIL'
    if 'ENG' in tags:
        return 'ENG'
    return 'OTH'

def load_data_from_csv(data_path: str, apply_rule_corrections: bool = True) -> Tuple[List[str], List[str]]:
    X, y = [], []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)
            corrections = 0
            for row in csv_reader:
                if len(row) > 3:
                    word, raw = (row[2] or "").strip(), (row[3] or "").strip()
                    if word and raw:
                        tag = map_label_to_target(raw)
                        if apply_rule_corrections:
                            rule = apply_rules(word)
                            if rule:
                                if rule != tag:
                                    corrections += 1
                                tag = rule
                        X.append(word)
                        y.append(tag)
        print(f"\nLoaded {len(X)} samples")
        print(f"Applied {corrections} rule corrections")
    except Exception as e:
        print(f"Error loading data: {e}")
    return X, y

def create_and_save_model(data_path='final_annotations.csv', use_rule_corrections=True):
    X_words, y_tags = load_data_from_csv(data_path, apply_rule_corrections=use_rule_corrections)
    if not X_words:
        print("No data loaded. Exiting.")
        return

    print("\nExtracting features...")
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=2000)
    X_features = vectorizer.fit_transform(X_words)

    X_train, X_temp, y_train, y_temp = train_test_split(X_features, y_tags, test_size=0.3, random_state=42, stratify=y_tags)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    best_acc = 0
    best_model = None
    best_depth = None
    for depth in [10, 20, 30, 40, 50, None]:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)
        print(f"Depth={depth} → Val Accuracy={acc:.4f}")
        if acc > best_acc:
            best_acc, best_model, best_depth = acc, clf, depth

    print(f"\nBest Decision Tree depth: {best_depth} (Val Accuracy={best_acc:.4f})")

    y_pred = best_model.predict(X_test)
    print("\nTest Set Performance:")
    print(classification_report(y_test, y_pred, zero_division=0, digits=4))
    cm = confusion_matrix(y_test, y_pred, labels=TARGET_LABELS)
    print("\nConfusion Matrix:\n", cm)

    model = best_model
    try:
        feature_names = vectorizer.get_feature_names_out()
    except:
        feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]

    plt.figure(figsize=(14, 8))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=model.classes_,
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title("PinoyBot Decision Tree (Gini Impurity)")
    plt.show()

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)

    print(f"\nDecision Tree model saved → {MODEL_PATH}")
    print(f"Vectorizer saved → {VECTORIZER_PATH}")

if __name__ == "__main__":
    create_and_save_model(use_rule_corrections=True)
