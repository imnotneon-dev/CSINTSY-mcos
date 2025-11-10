import csv
import pickle
import os
import re
import numpy as np
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Define file paths
MODEL_PATH = 'pinoybot_model.pkl'
VECTORIZER_PATH = 'pinoybot_vectorizer.pkl'
TARGET_LABELS = ['ENG', 'FIL', 'OTH']

# ==============================
# RULE-BASED PATTERNS
# ==============================

# Filipino prefixes
FIL_PREFIXES = (
    'mag', 'nag', 'pag', 'ipag', 'pang', 'mapag',
    'pin', 'kin', 'ipa', 'pa', 'ka', 'ma', 
    'na', 'tag', 'sang', 'pina', 'naka', 'maka',
    'nakaka', 'mapang', 'mang', 'nang', 'um'
)

# Filipino suffixes
FIL_SUFFIXES = (
    'han', 'hin', 'in', 'an', 'ng', 'ong',
    'nin', 'non'
)

# Common English prefixes
ENG_PREFIXES = (
    'un', 're', 'dis', 'over', 'mis', 'out', 'pre',
    'under', 'anti', 'de', 'fore', 'inter', 'mid',
    'sub', 'super', 'trans', 'semi', 'auto',
    'co', 'ex', 'extra', 'hyper', 'micro', 'post'
)

# Common English suffixes
ENG_SUFFIXES = (
    'ing', 'ed', 'ly', 'tion', 'sion', 'ness', 'ment',
    'ful', 'less', 'able', 'ible', 'ous', 'ious', 'al',
    'ical', 'er', 'est', 'ize', 'ise', 'ity', 'ty'
)

OTH_ABBREVIATION_CLUES = {
    'IMO', 'LOL', 'ASAP', 'AFAIK', 'IDK', 'BTW', 'JK', 'BRB', 'DIY', 'TLDR'
}

# Specific expressions/interjections that should be OTH (add more as needed)
OTH_EXPRESSION_CLUES = {
    'haha', 'hehe', 'hihi', 'hoho', 'grr', 'argh', 'ugh', 'wow', 'omg', 
    'lolz', 'yay', 'huy', 'ay', 'naku', 'ooh', 'ahh', 'ehh', 'heck'
}

# Filipino common double vowels (for rule check)
FIL_DOUBLE_VOWELS_CLUES = {'aa', 'ii', 'oo', 'uu', 'ee'}

# Words that are definitively Filipino (high-confidence set)
FIL_KNOWN_WORDS = {
    'sige', 'opo', 'po', 'naman', 'kasi', 'grabe', 'hay', 'naku', 'talaga', 
    'pala', 'daw', 'din', 'rin', 'lang', 'ulit', 'muna', 'nga', 'tayo', 'sila', 
    'siya', 'ako', 'ikaw', 'mo', 'ko' # Add more words you are certain of here
}

# Words that are definitively English (high-confidence set)
ENG_KNOWN_WORDS = {
    'project', 'study', 'happy', 'sad', 'love', 'hate', 'after', 'before', 
    'today', 'tomorrow', 'yesterday', 'always', 'never', 'because', 'maybe'
}

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
    
    word_upper = word.upper()
    word_lower = word.lower()
    
    # 1. Check if word is composed ONLY of punctuation
    if all(char in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' for char in word):
        return True
    
    # 2. Has digits (Numbers)
    if any(char.isdigit() for char in word):
        return True
    
    # 3. Expressions (Checking the list)
    if word_lower in OTH_EXPRESSION_CLUES:
        return True
    
    # 4. Abbreviations (Check if ALL CAPS AND in the list OR check for standard patterns)
    if word_upper in OTH_ABBREVIATION_CLUES:
        return True
        
    # Simple, non-list, ALL CAPS abbreviations check (e.g., USA, UP, BDO)
    if word.isupper() and len(word) >= 2 and word.isalpha():
        return True

    return False


def check_if_filipino(word: str) -> Optional[str]:
    """
    Check if word has Filipino characteristics using list checks.
    Returns 'FIL' if detected, None otherwise.
    """
    if not word or len(word) < 2:
        return None
    
    word_lower = word.lower()
    
    # Check for Filipino prefixes
    if any(word_lower.startswith(prefix) for prefix in FIL_PREFIXES) and len(word_lower) > 3:
        return 'FIL'
    
    # Check for Filipino suffixes
    if any(word_lower.endswith(suffix) for suffix in FIL_SUFFIXES) and len(word_lower) > 3:
        return 'FIL'
    
    # Check for double vowels (common in Filipino)
    if any(dv in word_lower for dv in FIL_DOUBLE_VOWELS_CLUES):
        return 'FIL'
    
    # Check for reduplication (e.g., bili-bili)
    if '-' in word:
        parts = word.split('-')
        if len(parts) == 2 and parts[0].lower() == parts[1].lower() and len(parts[0]) >= 2:
             return 'FIL'
        
        # Check for hyphenated code-switching patterns (nag-lunch, etc.)
        first_part = parts[0].lower()
        if any(first_part.startswith(prefix) for prefix in FIL_PREFIXES):
            return 'FIL'
    
    return None


def check_if_english(word: str) -> Optional[str]:
    """
    Check if word has English characteristics using list checks.
    Returns 'ENG' if detected, None otherwise.
    """
    if not word or len(word) < 3:
        return None
    
    word_lower = word.lower()
    
    # Check for English prefixes (ensure minimum word length after prefix)
    for prefix in ENG_PREFIXES:
        if word_lower.startswith(prefix) and len(word_lower) > len(prefix) + 2:
            # Rule to prevent easy Filipino prefix overlap
            if not any(word_lower.startswith(fp) for fp in FIL_PREFIXES):
                return 'ENG'
    
    # Check for English suffixes (ensure minimum word length before suffix)
    for suffix in ENG_SUFFIXES:
        if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
            # Rule to prevent easy Filipino suffix overlap
            if not any(word_lower.endswith(fs) for fs in FIL_SUFFIXES):
                return 'ENG'
    
    return None


def apply_rules(word: str) -> Optional[str]:
    """
    Apply rule-based classification.
    Returns label if confident, None if uncertain.
    Priority: OTH > KNOWN WORD > FIL MORPHOLOGY > ENG MORPHOLOGY
    """
    # 1. Check if it's OTH (highest priority)
    if check_if_other(word):
        return 'OTH'
        
    # 2. Check KNOWN WORDS (Overrides morphology if present)
    known_result = check_if_known(word)
    if known_result:
        return known_result
    
    # 3. Then check Filipino Morphology (medium priority)
    fil_result = check_if_filipino(word)
    if fil_result:
        return 'FIL'
    
    # 4. Finally check English Morphology (lowest priority)
    eng_result = check_if_english(word)
    if eng_result:
        return 'ENG'
    
    # No rule matched
    return None


# ==============================
# DATA LOADING WITH LABEL MAPPING
# ==============================

def map_label_to_target(raw_label: str) -> str:
    """Map detailed annotations to 3-class system."""
    label = raw_label.upper().strip()
    tags = label.split('-')
    
    # Filipino + Code-switched
    if 'FIL' in tags:
        return 'FIL'
    
    # English
    if 'ENG' in tags:
        return 'ENG'
    
    # Others (Sym, Num, Expr, Unk, Abb, NE)
    return 'OTH'


def load_data_from_csv(data_path: str, apply_rule_corrections: bool = True) -> Tuple[List[str], List[str]]:
    """
    Loads word and label data, mapping all labels to 3-class system.
    If apply_rule_corrections=True, overrides labels with rule-based predictions where confident.
    """
    X_words = []
    y_tags = []
    
    try:
        with open(data_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # Skip header row

            corrections_count = 0
            for row in csv_reader:
                if len(row) > 3: 
                    word = (row[2] or "").strip()
                    raw_tag = (row[3] or "").strip()

                    if word and raw_tag:
                        # Map to target label
                        target_tag = map_label_to_target(raw_tag)
                        
                        # Apply rule-based correction if enabled
                        if apply_rule_corrections:
                            rule_tag = apply_rules(word)
                            if rule_tag:
                                if rule_tag != target_tag:
                                    corrections_count += 1
                                target_tag = rule_tag
                        
                        X_words.append(word)
                        y_tags.append(target_tag)

    except FileNotFoundError:
        print(f"❌ Error: Data file '{data_path}' not found.")
        return [], []
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return [], []
    
    # Print statistics
    from collections import Counter
    label_counts = Counter(y_tags)
    print(f"\n✅ Loaded {len(X_words)} samples")
    if apply_rule_corrections:
        print(f"🔧 Applied {corrections_count} rule-based corrections ({corrections_count/len(X_words)*100:.1f}%)")
    print(f"📊 Label distribution:")
    for label in TARGET_LABELS:
        count = label_counts[label]
        pct = (count / len(y_tags) * 100) if y_tags else 0
        print(f"   {label}: {count:,} ({pct:.1f}%)")
    
    return X_words, y_tags


# ==============================
# MODEL TRAINING
# ==============================

def create_and_save_model(data_path='final_annotations.csv', use_rule_corrections=True):
    """
    Train and optimize a Naive Bayes model using TF-IDF features.
    Uses rule-based corrections and hyperparameter search for best performance.
    """
    X_words, y_tags = load_data_from_csv(data_path, apply_rule_corrections=use_rule_corrections)
    if not X_words:
        print("No data loaded. Exiting.")
        return

    print("\nExtracting features and tuning hyperparameters...")

    # Candidate TF-IDF configurations
    vectorizer_configs = [
        {"ngram_range": (2, 4), "max_features": 1000},
        {"ngram_range": (2, 5), "max_features": 1500},
        {"ngram_range": (3, 5), "max_features": 2000},
    ]

    alphas = [0.1, 0.5, 1.0, 1.5, 2.0]

    best_acc = 0
    best_model = None
    best_vectorizer = None
    best_config = None

    for config in vectorizer_configs:
        print(f"\nTesting config: ngram_range={config['ngram_range']}, max_features={config['max_features']}")

        vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=config["ngram_range"],
            max_features=config["max_features"]
        )
        X_features = vectorizer.fit_transform(X_words)

        # Stratified split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_features, y_tags, test_size=0.3, random_state=42, stratify=y_tags
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

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

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    X_test = best_vectorizer.transform(X_words)
    y_pred = best_model.predict(X_test)
    print(classification_report(y_tags, y_pred, zero_division=0, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_tags, y_pred, labels=TARGET_LABELS)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, label in enumerate(TARGET_LABELS):
        print(f"   {label}: {per_class_acc[i]*100:.2f}%")

    # Save the best model and vectorizer
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(best_vectorizer, f)

    print(f"\nModel saved → {MODEL_PATH}")
    print(f"Vectorizer saved → {VECTORIZER_PATH}")
    print("\nTraining complete! Model optimized and saved successfully.")

if __name__ == "__main__":
    create_and_save_model(use_rule_corrections=True)