import csv
import pickle
import re
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "pinoybot_model.pkl"
VECTORIZER_PATH = "pinoybot_vectorizer.pkl"
PIPELINE_PATH = "pinoybot_pipeline.pkl"
TARGET_LABELS = ["ENG", "FIL", "OTH"]


# ==============================
# LOAD DATA
# ==============================
def load_data_from_csv(data_path: str) -> Tuple[List[str], List[str]]:
    X_words, y_tags = [], []

    try:
        with open(data_path, "r", encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader, None)  # skip header

            for row in csv_reader:
                if len(row) > 3:
                    word = (row[2] or "").strip()
                    tag = (row[3] or "").strip().upper()
                    if word and tag in TARGET_LABELS:
                        X_words.append(word)
                        y_tags.append(tag)
    except FileNotFoundError:
        print(f"❌ File '{data_path}' not found.")
        return [], []
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return [], []

    print(f"✅ Loaded {len(X_words)} samples for training.")
    return X_words, y_tags


# ==============================
# HANDCRAFTED FEATURES
# ==============================
FIL_PREFIXES = ("mag", "nag", "pag", "ipag", "pang", "pin", "kin", "ipa", "pa", "ka", "ma", "na")
FIL_SUFFIXES = ("han", "hin", "in", "an")
ONLY_PUNCT_RE = re.compile(r"^[^\w\s]+$")
HAS_DIGIT_RE = re.compile(r".*\d.*")
REDUP_RE = re.compile(r"^([a-z]{2,})\1$", re.I)


def _word_shape(w: str) -> str:
    out = []
    for ch in w:
        if ch.isupper():
            out.append("X")
        elif ch.islower():
            out.append("x")
        elif ch.isdigit():
            out.append("d")
        elif ch in "-_":
            out.append("-")
        else:
            out.append("p")
    return "".join(out)


def _has_fil_prefix(w: str) -> bool:
    wl = w.lower().lstrip("'")
    return any(wl.startswith(p) for p in FIL_PREFIXES)


def _has_fil_suffix(w: str) -> bool:
    wl = w.lower().rstrip("'")
    return any(wl.endswith(s) for s in FIL_SUFFIXES)


def custom_feature_dicts(words: List[str]) -> List[dict]:
    feats = []
    for w in words:
        feats.append(
            {
                "len": len(w),
                "shape": _word_shape(w),
                "has_hyphen": int("-" in w),
                "has_digit": int(HAS_DIGIT_RE.match(w) is not None),
                "only_punct": int(ONLY_PUNCT_RE.match(w) is not None),
                "all_caps": int(w.isupper()),
                "title_case": int(w.istitle()),
                "redup": int(REDUP_RE.match(w or "") is not None),
                "fil_prefix": int(_has_fil_prefix(w)),
                "fil_suffix": int(_has_fil_suffix(w)),
            }
        )
    return feats


# ==============================
# CREATE AND SAVE MODEL
# ==============================
def create_and_save_model(data_path="final_annotations.csv", model_type="svm"):
    """
    model_type: 'svm' or 'nb'
    """

    X_words, y_tags = load_data_from_csv(data_path)
    if not X_words:
        return

    # Split data
    X_train_words, X_test_words, y_train, y_test = train_test_split(
        X_words, y_tags, test_size=0.3, random_state=42, stratify=y_tags
    )

    # Feature extractors
    char_vect = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=True, sublinear_tf=True)
    word_vect = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True, max_features=5000)
    handcrafted = Pipeline(
        steps=[
            ("to_dicts", FunctionTransformer(custom_feature_dicts, validate=False)),
            ("dict_vect", DictVectorizer(sparse=True)),
        ]
    )

    # Combine all
    vectorizer = FeatureUnion(
        transformer_list=[
            ("char", char_vect),
            ("word", word_vect),
            ("handcrafted", handcrafted),
        ]
    )

    X_train = vectorizer.fit_transform(X_train_words)
    X_test = vectorizer.transform(X_test_words)

    # ==============================
    # MODEL SELECTION
    # ==============================
    if model_type.lower() == "nb":
        print("⚙️ Training Naive Bayes...")
        best_alpha, best_acc = 1.0, 0
        for alpha in [0.1, 0.5, 1.0, 2.0]:
            model = MultinomialNB(alpha=alpha)
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            print(f"  α={alpha} → Accuracy={acc:.4f}")
            if acc > best_acc:
                best_alpha, best_acc = alpha, acc
        model = MultinomialNB(alpha=best_alpha)
        model.fit(X_train, y_train)
        print(f"✅ Best α={best_alpha} with Accuracy={best_acc:.4f}")

    else:
        print("⚙️ Training Linear SVM...")
        model = LinearSVC(class_weight="balanced", C=1.0, random_state=42)
        model.fit(X_train, y_train)

    # ==============================
    # EVALUATION
    # ==============================
    y_pred = model.predict(X_test)
    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=TARGET_LABELS))

    # ==============================
    # SAVE ARTIFACTS
    # ==============================
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"\n💾 Saved model → {MODEL_PATH}")
    print(f"💾 Saved vectorizer → {VECTORIZER_PATH}")

    full_pipeline = Pipeline([("features", vectorizer), ("clf", model)])
    with open(PIPELINE_PATH, "wb") as f:
        pickle.dump(full_pipeline, f)
    print(f"💾 Full pipeline saved as {PIPELINE_PATH}")


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    # You can change 'svm' to 'nb' if you want to train the Naive Bayes variant
    create_and_save_model("final_annotations.csv", model_type="svm")
