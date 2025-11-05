import csv
import pickle
import os
import re
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Define file paths (MODEL_PATH and VECTORIZER_PATH remain the same)
MODEL_PATH = 'pinoybot_model.pkl'
VECTORIZER_PATH = 'pinoybot_vectorizer.pkl'
TARGET_LABELS = ['ENG', 'FIL', 'OTH']

# Define file paths
MODEL_PATH = 'pinoybot_model.pkl'
VECTORIZER_PATH = 'pinoybot_vectorizer.pkl'

# --- The 3 Mandatory Target Classes ---
TARGET_LABELS = ['ENG', 'FIL', 'OTH']

def load_data_from_csv(data_path: str) -> Tuple[List[str], List[str]]:
    """Loads word and label data, using only the primary tag (column 4)."""
    X_words = []
    y_tags = []
    
    try:
        with open(data_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader) # Skip header row

            for row in csv_reader:
                # Word is at index 2, Primary Tag is at index 3
                if len(row) > 3: 
                    word = row[2]
                    primary_tag = row[3].upper() # Read tag and uppercase it

                    if primary_tag in TARGET_LABELS and word:
                        X_words.append(preprocess_word(word))
                        y_tags.append(primary_tag)

    except FileNotFoundError:
        print(f"Error: Data file '{data_path}' not found. Please ensure it is named 'final_annotations.csv'.")
        return [], []
    except Exception as e:
        print(f"An unexpected error occurred while reading the CSV: {e}")
        return [], []
        
    print(f"Loaded {len(X_words)} data points for training, using tags: {', '.join(TARGET_LABELS)}")
    return X_words, y_tags


def create_and_save_model(data_path='final_annotations.csv'):
    
    X_words, y_tags = load_data_from_csv(data_path)
    
    # --- 2. FEATURE EXTRACTION: CHARACTER N-GRAM TF-IDF ---
    # Keeping the existing powerful feature set
    vectorizer = TfidfVectorizer(
        analyzer='char',      
        ngram_range=(2, 5),   
        max_features=1000     
    )
    X_features = vectorizer.fit_transform(X_words)

# --- 3. TRAIN/VALIDATION/TEST SPLIT (70/15/15) ---
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_features, y_tags, test_size=0.30, random_state=42, stratify=y_tags
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

   # Hyperparameter tuning (alpha)
    best_alpha, best_acc = 1.0, 0
    for alpha in [0.1, 0.5, 1.0, 2.0]:
        model = MultinomialNB(alpha=alpha)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)
        print(f"Alpha={alpha:.1f} | Validation Accuracy={acc:.4f}")
        if acc > best_acc:
            best_alpha, best_acc = alpha, acc

    print(f"\n✅ Best alpha found: {best_alpha} (Val Accuracy={best_acc:.4f})")

    # Train final model
    model = MultinomialNB(alpha=best_alpha)
    model.fit(X_train, y_train)

    # --- 4. EVALUATION ---
    y_pred = model.predict(X_test)
    print("\n--- Naive Bayes Model Training & Evaluation Report ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    # --- 5. SAVE THE MODEL AND VECTORIZER ---
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Trained model successfully saved as {MODEL_PATH}")

    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer successfully saved as {VECTORIZER_PATH}")

if __name__ == "__main__":
    create_and_save_model()