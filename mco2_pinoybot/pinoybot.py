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

MODEL_PATH = 'pinoybot_model.pkl'
VECTORIZER_PATH = 'pinoybot_vectorizer.pkl'

_MODEL: Any = None
_VECTORIZER: Any = None

def _load_resources():
    global _MODEL, _VECTORIZER
    
    if _MODEL is not None and _VECTORIZER is not None:
        return
    
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                _MODEL = pickle.load(f)
            print(f"Successfully loaded model from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            
    if os.path.exists(VECTORIZER_PATH):
        try:
            with open(VECTORIZER_PATH, 'rb') as f:
                _VECTORIZER = pickle.load(f)
            print(f"Successfully loaded vectorizer from {VECTORIZER_PATH}")
        except Exception as e:
            print(f"Error loading vectorizer: {e}")
            
    if _MODEL is None or _VECTORIZER is None:
        print("WARNING: Model or vectorizer did not load properly.")

def tag_language(tokens: List[str]) -> List[str]:
    # 1. Load your trained model from disk (e.g., using pickle or joblib)
    #    Example: with open('trained_model.pkl', 'rb') as f: model = pickle.load(f)
    #    (Replace with your actual model loading code)

    _load_resources()
    if _MODEL is None or _VECTORIZER is None:
        return ['OTH' for _ in tokens]

    if not tokens:
        return []
    
    predicted_tags = []
    model_tokens = []       
    model_features = None    
    model_predictions = []  

    for token in tokens:
        rule_tag = apply_rules(token)
        if rule_tag in ('OTH', 'FIL', 'ENG'):
            predicted_tags.append(rule_tag)
        else:
            model_tokens.append(token)
            predicted_tags.append(None) 
    
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
        # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        # features_dense = features.toarray()
        # feature_names = _VECTORIZER.get_feature_names_out()

        # print("\n--- Feature Analysis (Active N-Grams and Predictions) ---")
        # print(f"Total Words Tested: {len(tokens)}")
        # print(f"Total Features Learned: {len(feature_names)}\n")

        # for word, vector, tag in zip(tokens, features_dense, predicted_tags):
        #     non_zero_indices = np.nonzero(vector)[0]

        #     print("=========================================================")
        #     print(f"| TOKEN: '{word}' | PREDICTED CLASS: {tag}")
        #     print("=========================================================")

        #     if len(non_zero_indices) == 0:
        #         print("  [No active features found among the learned N-grams.]")
        #     else:
        #         print("  [Active Features (N-Grams) and their TF-IDF Scores]:")
        #         for i in non_zero_indices:
        #             feature_name = feature_names[i]
        #             feature_score = vector[i]
        #             print(f"  -> '{feature_name}' (Score: {feature_score:.4f})")

        # print("-------------------------------------------------\n")
        # np.set_printoptions(threshold=1000, linewidth=75)
    
    # predicted = _MODEL.predict(features)
    # Convert numpy string objects to normal Python strings
    predicted_tags = [str(tag) for tag in predicted_tags]
    
    return predicted_tags

if __name__ == "__main__":
    _load_resources()
