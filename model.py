

from datetime import datetime

import numpy as np
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler

from features import (
    build_tfidf_vectorizer,
    extract_handcrafted_features,
    handcrafted_features_to_vector,
    HANDCRAFTED_FEATURE_NAMES,
)
from preprocess import clean_text, combine_subject_body

DEFAULT_MODEL_PATH = Path(__file__).parent / "phishing_model.joblib"


class PhishingDetectionModel:
    """
    A self-contained model object: text vectorizer + numeric-feature
    scaler + classifier, trained together and saved/loaded as one unit.
    """

    def __init__(self, algorithm="random_forest", max_tfidf_features=3000):
        self.algorithm = algorithm
        self.vectorizer = build_tfidf_vectorizer(max_features=max_tfidf_features)
        self.scaler = StandardScaler()
        self.classifier = self._build_classifier(algorithm)
        self.version = None
        self.is_trained = False

    @staticmethod
    def _build_classifier(algorithm):
        if algorithm == "logistic_regression":
            return LogisticRegression(max_iter=1000, class_weight="balanced")
        # default: random forest
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
        )

    # ------------------------------------------------------------------
    # Feature building
    # ------------------------------------------------------------------
    def _build_feature_matrix(self, texts, fit=False):
        """
        texts: list of raw (subject+body combined, uncleaned) strings.
        Returns a combined sparse matrix of TF-IDF + scaled handcrafted
        features, and the list of handcrafted feature dicts (useful for
        storing in the database).
        """
        cleaned = [clean_text(t) for t in texts]

        if fit:
            tfidf_matrix = self.vectorizer.fit_transform(cleaned)
        else:
            tfidf_matrix = self.vectorizer.transform(cleaned)

        handcrafted_dicts = [extract_handcrafted_features(t) for t in texts]
        handcrafted_matrix = np.array(
            [handcrafted_features_to_vector(d) for d in handcrafted_dicts]
        )

        if fit:
            handcrafted_scaled = self.scaler.fit_transform(handcrafted_matrix)
        else:
            handcrafted_scaled = self.scaler.transform(handcrafted_matrix)

        combined = hstack([tfidf_matrix, csr_matrix(handcrafted_scaled)])
        return combined, handcrafted_dicts

    # ------------------------------------------------------------------
    # Training / evaluation
    # ------------------------------------------------------------------
    def train(self, subjects, bodies, labels, test_size=0.25, random_state=42):
        """
        subjects, bodies, labels: parallel lists.
        labels are strings: 'phishing' or 'legitimate'.
        Returns a dict of evaluation metrics computed on a held-out test split.
        """
        texts = [combine_subject_body(s, b) for s, b in zip(subjects, bodies)]
        y = np.array([1 if lbl == "phishing" else 0 for lbl in labels])

        X_train_text, X_test_text, y_train, y_test = train_test_split(
            texts, y, test_size=test_size, random_state=random_state, stratify=y
        )

        X_train, _ = self._build_feature_matrix(X_train_text, fit=True)
        X_test, _ = self._build_feature_matrix(X_test_text, fit=False)

        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        self.version = datetime.utcnow().strftime("v%Y%m%d-%H%M%S")

        y_pred = self.classifier.predict(X_test)

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, target_names=["legitimate", "phishing"], zero_division=0
            ),
            "test_size": len(y_test),
            "train_size": len(y_train),
            "algorithm": self.algorithm,
            "model_version": self.version,
        }
        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, subject, body):
        """
        Predict a single email. Returns:
            (label:str, confidence:float, feature_dict:dict)
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained/loaded yet.")

        text = combine_subject_body(subject, body)
        X, handcrafted_dicts = self._build_feature_matrix([text], fit=False)

        pred = self.classifier.predict(X)[0]
        proba = None
        if hasattr(self.classifier, "predict_proba"):
            proba = self.classifier.predict_proba(X)[0]
            confidence = float(proba[pred])
        else:
            confidence = 1.0

        label = "phishing" if pred == 1 else "legitimate"
        return label, confidence, handcrafted_dicts[0]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path=DEFAULT_MODEL_PATH):
        joblib.dump(self, path)

    @staticmethod
    def load(path=DEFAULT_MODEL_PATH):
        return joblib.load(path)
