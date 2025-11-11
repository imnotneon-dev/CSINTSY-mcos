"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import os
import pickle
import numpy as np
from typing import List, Any
from trainer import apply_rules

# define file paths (just constant strings here)
MODEL_PATH = 'pinoybot_model.pkl'
VECTORIZER_PATH = 'pinoybot_vectorizer.pkl'

# global variables to store the loaded model and vectorizer
_MODEL: Any = None
_VECTORIZER: Any = None

# helper function to load model and vectorizer
def _load_resources():
    global _MODEL, _VECTORIZER
    
    if _MODEL is not None and _VECTORIZER is not None:
        return
    
    # loading model
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                _MODEL = pickle.load(f)
            print(f"Successfully loaded model from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            
    # loading vectorizer
    if os.path.exists(VECTORIZER_PATH):
        try:
            with open(VECTORIZER_PATH, 'rb') as f:
                _VECTORIZER = pickle.load(f)
            print(f"Successfully loaded vectorizer from {VECTORIZER_PATH}")
        except Exception as e:
            print(f"Error loading vectorizer: {e}")
            
    if _MODEL is None or _VECTORIZER is None:
        print("WARNING: Model or vectorizer did not load properly.")

# Main tagging function
def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.
    Args:
        tokens: List of word tokens (strings).
    Returns:
        tags: List of predicted tags ("ENG", "FIL", or "OTH"), one per token.
    """
    # 1. Load your trained model from disk (e.g., using pickle or joblib)
    #    Example: with open('trained_model.pkl', 'rb') as f: model = pickle.load(f)
    #    (Replace with your actual model loading code)

    _load_resources()
    if _MODEL is None or _VECTORIZER is None:
        return ['OTH' for _ in tokens]

    if not tokens:
        return []
    
    # to store results
    predicted_tags = []
    model_tokens = []        # Tokens sent to the model
    model_features = None    # TF-IDF vectors
    model_predictions = []   # Model’s predicted labels

    # apply rules first (from trainer.py)
    for token in tokens:
        rule_tag = apply_rules(token)

        # confident rule classification
        if rule_tag in ('OTH', 'FIL', 'ENG'):
            predicted_tags.append(rule_tag)
        else:
            # model handles uncertain tags
            model_tokens.append(token)
            predicted_tags.append(None)  # placeholder
    
    # 2. Extract features from the input tokens to create the feature matrix
    #    Example: features = ... (your feature extraction logic here)
    # predictions for remaining tokens
    
    if model_tokens:
        tokens_lower = [t.lower() for t in tokens]
        features = _VECTORIZER.transform(tokens_lower)
        model_predictions = _MODEL.predict(features)

        # Fill in predictions where rule_tag was None
        pred_index = 0
        for i in range(len(predicted_tags)):
            if predicted_tags[i] is None:
                predicted_tags[i] = model_predictions[pred_index]
                pred_index += 1

        # debugging purposes: show active features for each token
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        features_dense = features.toarray()
        feature_names = _VECTORIZER.get_feature_names_out()

        print("\n--- Feature Analysis (Active N-Grams and Predictions) ---")
        print(f"Total Words Tested: {len(tokens)}")
        print(f"Total Features Learned: {len(feature_names)}\n")

        for word, vector, tag in zip(tokens, features_dense, predicted_tags):
            non_zero_indices = np.nonzero(vector)[0]

            print("=========================================================")
            print(f"| TOKEN: '{word}' | PREDICTED CLASS: {tag}")
            print("=========================================================")

            if len(non_zero_indices) == 0:
                print("  [No active features found among the learned N-grams.]")
            else:
                print("  [Active Features (N-Grams) and their TF-IDF Scores]:")
                for i in non_zero_indices:
                    feature_name = feature_names[i]
                    feature_score = vector[i]
                    print(f"  -> '{feature_name}' (Score: {feature_score:.4f})")

        print("-------------------------------------------------\n")
        np.set_printoptions(threshold=1000, linewidth=75)
    
    predicted = _MODEL.predict(features)
    # Convert numpy string objects to normal Python strings
    predicted_tags = [str(tag) for tag in predicted_tags]
    
    return predicted_tags

if __name__ == "__main__":
    _load_resources()

    print("\n-------------------------------------------------\n")
    print("TEST CASE 1:")

    example_tokens_1 = ["Gusto", "ko", "mag-compute", "ng", "2024", "."]
    tags_1 = tag_language(example_tokens_1)
    print(f"Tokens 1: {example_tokens_1}")
    print(f"Expected Tags 1: ['FIL', 'FIL', 'OTH', 'FIL', 'OTH', 'OTH']")
    print(f"Predicted Tags 1: {tags_1}")

    print("\n-------------------------------------------------\n")
    print("TEST CASE 2:")

    example_tokens_2 = ["Si", "Dr", "dela", "Cruz", "is", "a", "good", "Filipino", "teacher", "."]
    tags_2 = tag_language(example_tokens_2)
    print(f"\nTokens 2: {example_tokens_2}")
    print("Expected Tags 2: ['FIL', 'ENG', 'OTH', 'OTH', 'ENG', 'ENG', 'ENG', 'ENG', 'ENG', 'OTH']")
    print(f"Predicted Tags 2: {tags_2}")

    print("\n-------------------------------------------------\n")
    print("TEST CASE 3:")
    example_tokens_3 = ["Hello", "kumusta", "ka", "ngayon", "?"]
    tags_3 = tag_language(example_tokens_3)
    print(f"\nTokens 3: {example_tokens_3}")
    print("Expected Tags 3: ['ENG', 'FIL', 'FIL', 'FIL', 'OTH']")
    print(f"Predicted Tags 3: {tags_3}")
