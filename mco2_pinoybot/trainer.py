# trainer.py

import csv
import pickle
import re
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer, DictVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer


# --- Change this line: ---
# changed it to this
from sklearn.svm import LinearSVC
# -------------------------

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Define file paths (MODEL_PATH and VECTORIZER_PATH remain the same)
MODEL_PATH = 'pinoybot_model.pkl'            # classifier only
VECTORIZER_PATH = 'pinoybot_vectorizer.pkl'  # feature extractor only
PIPELINE_PATH = 'pinoybot_pipeline.pkl'      # for debugging

# --- The 3 Mandatory Target Classes ---
TARGET_LABELS = ['ENG', 'FIL', 'OTH']


def load_data_from_csv(data_path: str) -> Tuple[List[str], List[str]]:
    """Loads word and label data, using only the primary tag (column 4)."""
    X_words: List[str] = []
    y_tags: List[str] = []

    try:
        with open(data_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader, None)  # Skip header row incase

            for row in csv_reader:
                # Word is at index 2, Primary Tag is at index 3
                if len(row) > 3:
                    word = (row[2] or '').strip()
                    primary_tag = (row[3] or '').strip().upper()

                    if primary_tag in TARGET_LABELS and word:
                        X_words.append(word)       # keep case; handle lowercase
                        y_tags.append(primary_tag)

    except FileNotFoundError:
        print(f"Error: Data file '{data_path}' not found. Please ensure it is named 'final_annotations.csv'.")
        return [], []
    except Exception as e:
        print(f"An unexpected error occurred while reading the CSV: {e}")
        return [], []

    print(f"Loaded {len(X_words)} data points for training, using tags: {', '.join(TARGET_LABELS)}")
    return X_words, y_tags


# --- Handcrafted (dictionary-free) morphology/orthography features ---
FIL_PREFIXES = ("mag", "nag", "pag", "ipag", "pang", "pin", "kin", "ipa", "pa", "ka", "ma", "na")
FIL_INFIXES  = ("um", "in")
FIL_SUFFIXES = ("han", "hin", "in", "an")

ONLY_PUNCT_RE = re.compile(r"^[^\w\s]+$")
HAS_DIGIT_RE  = re.compile(r".*\d.*")
REDUP_RE      = re.compile(r"^([a-z]{2,})\1$", re.I)

def _word_shape(w: str) -> str:
    # Example shapes: Xx, xxxx, XXX, ddd, Xxx-dd
    out = []
    for ch in w:
        if ch.isupper():   out.append("X")
        elif ch.islower(): out.append("x")
        elif ch.isdigit(): out.append("d")
        elif ch in "-_":   out.append("-")
        else:              out.append("p")
    return "".join(out)

def _has_fil_prefix(w: str) -> bool:
    wl = w.lower().lstrip("'")
    return any(wl.startswith(p) for p in FIL_PREFIXES)

def _has_fil_suffix(w: str) -> bool:
    wl = w.lower().rstrip("'")
    return any(wl.endswith(s) for s in FIL_SUFFIXES)

def _has_fil_infix(w: str) -> bool:
    wl = w.lower()
    return ("um" in wl[1:-1]) or ("in" in wl[1:-1])

def custom_feature_dicts(words: List[str]) -> List[dict]:
    feats = []
    for w in words:
        feats.append({
            "len": len(w),
            "shape": _word_shape(w),
            "has_hyphen": 1 if "-" in w else 0,
            "has_digit": 1 if HAS_DIGIT_RE.match(w) else 0,
            "only_punct": 1 if ONLY_PUNCT_RE.match(w) else 0,
            "all_caps": 1 if w.isupper() else 0,
            "title_case": 1 if w.istitle() else 0,
            "redup": 1 if REDUP_RE.match(w or "") else 0,
            "fil_prefix": 1 if _has_fil_prefix(w) else 0,
            "fil_infix": 1 if _has_fil_infix(w) else 0,
            "fil_suffix": 1 if _has_fil_suffix(w) else 0,
        })
    return feats


def create_and_save_model(data_path: str = 'final_annotations.csv'):
    # --- 1. LOAD & SPLIT DATA ---
    X_words, y_tags = load_data_from_csv(data_path)
    if not X_words:
        return

    X_train_words, X_test_words, y_train, y_test = train_test_split(
        X_words, y_tags, test_size=0.30, random_state=42, stratify=y_tags
    )

    # --- 2. FEATURE EXTRACTION ---
    # handles misspellings, affixes, mixed scripts
    char_vect = TfidfVectorizer(
        analyzer='char',
        ngram_range=(1, 5),
        lowercase=True,
        sublinear_tf=True,
        min_df=1
    )
 
    word_vect = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 1),
        lowercase=True,
        min_df=2,
        max_features=5000
    )
    # Handcrafted morphology/orthography features (dictionary-free)
    handcrafted = Pipeline(steps=[
        ("to_dicts", FunctionTransformer(custom_feature_dicts, validate=False)),
        ("dict_vect", DictVectorizer(sparse=True)),
    ])

    # Combine all channels into one "vectorizer" that your pinoybot.py can .transform()
    vectorizer = FeatureUnion(transformer_list=[
        ("char", char_vect),
        ("word", word_vect),
        ("handcrafted", handcrafted),
    ])

    # Fit the vectorizer on TRAIN words only, then transform train/test
    X_train = vectorizer.fit_transform(X_train_words)
    X_test  = vectorizer.transform(X_test_words)

    # --- 3. MODEL: LINEAR SVM  ---
    #basically it helps with rarer things spotted in training so it prevents always guessing from common labels
    model = LinearSVC(class_weight="balanced", C=1.0, random_state=42)
    model.fit(X_train, y_train)

    # --- 4. EVALUATION ---
    y_pred = model.predict(X_test)
    print("\n--- LinearSVC Model Training & Evaluation Report ---")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred, labels=TARGET_LABELS))

    # --- 5. SAVE THE MODEL AND VECTORIZER ---
    # Save separate artifacts to match your existing pinoybot.py loader
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer successfully saved as {VECTORIZER_PATH}")

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Trained model successfully saved as {MODEL_PATH}")

    # (Optional) also save a full pipeline for convenience/debugging
    full_pipeline = Pipeline([("features", vectorizer), ("clf", model)])
    with open(PIPELINE_PATH, 'wb') as f:
        pickle.dump(full_pipeline, f)
    print(f"(Optional) Full pipeline saved as {PIPELINE_PATH}")


if __name__ == "__main__":
    create_and_save_model()
