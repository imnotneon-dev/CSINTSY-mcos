import csv
import pickle
import os
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ==============================
# File paths
# ==============================
MODEL_PATH = 'pinoybot_model.pkl'
VECTORIZER_PATH = 'pinoybot_vectorizer.pkl'
TARGET_LABELS = ['ENG', 'FIL', 'OTH']

# ==============================
# RULE-BASED PATTERNS
# ==============================

MIN_ROOT_LEN = 3

# Letters not common in Filipino
ENG_STRONG_LETTERS = set("qxz")        # Most likely English
ENG_SOFT_LETTERS   = set("vfjc")       # Might appear in Filipino words

# Common English bigrams
ENG_BIGRAM_CLUES = {
    'th', 'sh', 'ch', 'ph', 'wh', 'ck', 'st', 'tr', 'dr', 'pr',
    'pl', 'cl', 'gl', 'fl', 'bl', 'br', 'cr', 'gr'
}

FIL_BIGRAM_CLUES = {
    'ng', 'mg'  # you can add more if needed
}

# Expressions/interjections classified as OTH
OTH_EXPRESSION_CLUES = {
    'haha', 'hehe', 'hihi', 'hoho', 'grr', 'argh', 'ugh', 'wow', 'omg', 
    'lolz', 'yay', 'huy', 'ay', 'naku', 'ooh', 'ahh', 'ehh', 'heck'
}

# Abbreviations classified as OTH
OTH_ABBREVIATION_CLUES = {
    'IMO', 'LOL', 'ASAP', 'AFAIK', 'IDK', 'BTW', 'JK', 'BRB', 'DIY', 'TLDR'
}

# Filipino clues: double vowels
FIL_DOUBLE_VOWELS_CLUES = {'aa', 'ii', 'uu'}

# High-confidence Filipino words
FIL_KNOWN_WORDS = {
    'sige', 'opo', 'po', 'naman', 'kasi', 'grabe', 'hay', 'naku', 'talaga', 
    'pala', 'daw', 'din', 'rin', 'lang', 'ulit', 'muna', 'nga', 'tayo', 'sila', 
    'siya', 'ako', 'ikaw', 'mo', 'ko'
}

# High-confidence English words
ENG_KNOWN_WORDS = {
    'project', 'is', 'a', 'study', 'happy', 'sad', 'love', 'hate', 'after', 'before', 
    'today', 'tomorrow', 'yesterday', 'always', 'never', 'because', 'maybe'
}

# ==============================
# RULE FUNCTIONS
# ==============================

def check_if_known(word: str) -> Optional[str]:
    word_lower = word.lower()
    if word_lower in FIL_KNOWN_WORDS:
        return 'FIL'
    if word_lower in ENG_KNOWN_WORDS:
        return 'ENG'
    return None

def check_if_other(word: str) -> bool:
    if not word:
        return True
    word = word.strip()
    if not word:
        return True
    word_upper = word.upper()
    word_lower = word.lower()
    
    # Only punctuation
    if not any(ch.isalnum() for ch in word):
        return True
    # Contains digits
    if any(ch.isdigit() for ch in word):
        return True
    # Expressions
    if word_lower in OTH_EXPRESSION_CLUES:
        return True
    # Known abbreviations
    if word_upper in OTH_ABBREVIATION_CLUES:
        return True
    # All caps abbreviation
    if word.isupper() and len(word) >= 2 and word.isalpha():
        return True
    return False

def check_if_filipino(word: str) -> Optional[str]:
    """Detect Filipino words via reduplication or double vowels."""
    if not word:
        return None
    word_lower = word.lower()
    
    # Hyphenated reduplication: bili-bili
    if '-' in word_lower:
        parts = word_lower.split('-')
        if len(parts) == 2 and parts[0] == parts[1] and len(parts[0]) >= 2:
            return 'FIL'
    
    # Double vowels
    if any(dv in word_lower for dv in FIL_DOUBLE_VOWELS_CLUES):
        return 'FIL'
    
    return None

def check_if_filipino_bigrams(word: str) -> Optional[str]:
    if not word or len(word) < 2:
        return None
    w = word.lower()
    for bg in FIL_BIGRAM_CLUES:
        if bg in w:
            return 'FIL'
    return None

def check_if_english_letters(word: str) -> Optional[str]:
    if not word:
        return None
    w = word.lower()
    # Strong clue
    if any(ch in ENG_STRONG_LETTERS for ch in w):
        return 'ENG'
    # Softer clue
    if any(ch in ENG_SOFT_LETTERS for ch in w) and len(w) > 3:
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

def apply_rules(word: str) -> Optional[str]:
    """Apply rule-based classification."""
    if not word:
        return None
    word = word.strip()
    if not word:
        return None
    
    # 1. Other (highest priority)
    if check_if_other(word):
        return 'OTH'
    # 2. Known words
    known = check_if_known(word)
    if known:
        return known
    # 3. Filipino clues: double vowels or reduplication
    fil = check_if_filipino(word)
    if fil:
        return fil
    # 3b. Filipino bigrams
    fil_bg = check_if_filipino_bigrams(word)
    if fil_bg:
        return fil_bg
    # 4. English bigrams
    bigram = check_if_english_bigrams(word)
    if bigram:
        return bigram
    # 5. English letters
    eng = check_if_english_letters(word)
    if eng:
        return eng
    return None

# ==============================
# DATA LOADING
# ==============================

def map_label_to_target(raw_label: str) -> str:
    label = raw_label.upper().strip()
    tags = label.split('-')
    if 'FIL' in tags:
        return 'FIL'
    if 'ENG' in tags:
        return 'ENG'
    return 'OTH'

def load_data_from_csv(data_path: str, apply_rule_corrections: bool = True) -> Tuple[List[str], List[str]]:
    X_words = []
    y_tags = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            corrections_count = 0
            for row in reader:
                if len(row) > 3:
                    word = (row[2] or "").strip()
                    raw_tag = (row[3] or "").strip()
                    if word and raw_tag:
                        target_tag = map_label_to_target(raw_tag)
                        if apply_rule_corrections:
                            rule_tag = apply_rules(word)
                            if rule_tag and rule_tag != target_tag:
                                corrections_count += 1
                                target_tag = rule_tag
                        X_words.append(word)
                        y_tags.append(target_tag)
    except Exception as e:
        print(f"Error loading data: {e}")
        return [], []
    
    from collections import Counter
    label_counts = Counter(y_tags)
    print(f"\nLoaded {len(X_words)} samples")
    if apply_rule_corrections:
        print(f"🔧 Applied {corrections_count} rule-based corrections ({corrections_count/len(X_words)*100:.1f}%)")
    for label in TARGET_LABELS:
        count = label_counts[label]
        pct = (count/len(y_tags)*100) if y_tags else 0
        print(f"   {label}: {count} ({pct:.1f}%)")
    return X_words, y_tags

# ==============================
# MODEL TRAINING
# ==============================

def create_and_save_model(data_path='final_annotations.csv', use_rule_corrections=True):
    X_words, y_tags = load_data_from_csv(data_path, apply_rule_corrections=use_rule_corrections)
    if not X_words:
        print("No data loaded. Exiting.")
        return
    print("\nExtracting features and tuning hyperparameters...")

    vectorizer_configs = [
        {"ngram_range": (2,4), "max_features": 1000},
        {"ngram_range": (2,5), "max_features": 1500},
        {"ngram_range": (3,5), "max_features": 2000},
    ]
    alphas = [0.1,0.5,1.0,1.5,2.0]

    best_acc = 0
    best_model = None
    best_vectorizer = None
    best_config = None

    for config in vectorizer_configs:
        print(f"\nTesting config: ngram_range={config['ngram_range']}, max_features={config['max_features']}")
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=config["ngram_range"], max_features=config["max_features"])
        X_features = vectorizer.fit_transform(X_words)

        X_train, X_temp, y_train, y_temp = train_test_split(
            X_features, y_tags, test_size=0.3, random_state=42, stratify=y_tags)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        for alpha in alphas:
            model = MultinomialNB(alpha=alpha)
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_val_pred)
            print(f"   α={alpha:.2f} → Validation Accuracy={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_vectorizer = vectorizer
                best_config = (config, alpha)

    print("\nBest configuration:")
    print(f"   ngram_range={best_config[0]['ngram_range']}, max_features={best_config[0]['max_features']}, alpha={best_config[1]}")
    print(f"   Validation Accuracy={best_acc:.4f}")

    # Evaluate on full data
    X_test_features = best_vectorizer.transform(X_words)
    y_pred = best_model.predict(X_test_features)
    print(classification_report(y_tags, y_pred, zero_division=0, digits=4))
    cm = confusion_matrix(y_tags, y_pred, labels=TARGET_LABELS)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)
    per_class_acc = cm.diagonal()/cm.sum(axis=1)
    for i,label in enumerate(TARGET_LABELS):
        print(f"   {label}: {per_class_acc[i]*100:.2f}%")

    with open(MODEL_PATH,'wb') as f:
        pickle.dump(best_model,f)
    with open(VECTORIZER_PATH,'wb') as f:
        pickle.dump(best_vectorizer,f)
    print(f"\nModel saved → {MODEL_PATH}")
    print(f"Vectorizer saved → {VECTORIZER_PATH}")
    print("Training complete!")

if __name__ == "__main__":
    create_and_save_model(use_rule_corrections=True)
