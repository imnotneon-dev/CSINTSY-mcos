import csv
import pickle
import os
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
# --- Change this line: ---
from sklearn.naive_bayes import MultinomialNB
# -------------------------
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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
                        X_words.append(word.lower())
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

    # --- 3. TRAIN/TEST SPLIT AND MODEL TRAINING (70/30 for now) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_tags, test_size=0.30, random_state=42, stratify=y_tags
    )
    
    model = MultinomialNB(alpha=1.0)
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