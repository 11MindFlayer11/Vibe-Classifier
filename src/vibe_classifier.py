from transformers import pipeline
import spacy
from typing import List, Dict, Any
import re


class VibeClassifier:
    def __init__(self):
        """Initialize the vibe classifier with pre-trained models."""
        # Load zero-shot classification pipeline
        self.classifier = pipeline(
            "zero-shot-classification", model="facebook/bart-large-mnli"
        )

        # Load spaCy for text preprocessing
        self.nlp = spacy.load("en_core_web_sm")

        # Define fashion vibes
        self.vibes = [
            "Coquette",
            "Clean Girl",
            "Cottagecore",
            "Streetcore",
            "Y2K",
            "Boho",
            "Party Glam",
        ]

        # Define vibe keywords and associations
        self.vibe_keywords = {
            "Coquette": [
                "feminine",
                "cute",
                "girly",
                "bows",
                "lace",
                "pink",
                "dainty",
                "sweet",
                "romantic",
            ],
            "Clean Girl": [
                "minimal",
                "sleek",
                "neutral",
                "classic",
                "effortless",
                "polished",
                "simple",
            ],
            "Cottagecore": [
                "rustic",
                "floral",
                "vintage",
                "pastoral",
                "natural",
                "romantic",
                "countryside",
            ],
            "Streetcore": [
                "urban",
                "edgy",
                "street",
                "casual",
                "cool",
                "trendy",
                "sporty",
            ],
            "Y2K": [
                "retro",
                "2000s",
                "nostalgic",
                "colorful",
                "bold",
                "futuristic",
                "playful",
            ],
            "Boho": [
                "bohemian",
                "free-spirited",
                "earthy",
                "layered",
                "ethnic",
                "artistic",
            ],
            "Party Glam": [
                "glamorous",
                "sparkly",
                "elegant",
                "luxurious",
                "shiny",
                "dressy",
                "formal",
            ],
        }

    def classify_vibes(self, text: str, top_k: int = 3) -> List[Dict[str, float]]:
        """Classify the fashion vibes in a text using zero-shot classification.

        Args:
            text (str): Input text (caption, hashtags, etc.)
            top_k (int): Number of top vibes to return

        Returns:
            List[Dict[str, float]]: Top-k vibes with confidence scores
        """
        # Preprocess text
        text = self._preprocess_text(text)

        # Perform zero-shot classification
        result = self.classifier(
            text,
            candidate_labels=self.vibes,
            hypothesis_template="This outfit has a {} style.",
            multi_label=True,
        )

        # Format results
        vibe_scores = []
        for vibe, score in zip(result["labels"], result["scores"]):
            vibe_scores.append({"vibe": vibe, "confidence": float(score)})

        # Sort by confidence and return top-k
        vibe_scores.sort(key=lambda x: x["confidence"], reverse=True)
        return vibe_scores[:top_k]

    def analyze_keywords(self, text: str) -> Dict[str, float]:
        """Analyze text for vibe-related keywords and calculate vibe scores.

        Args:
            text (str): Input text

        Returns:
            Dict[str, float]: Vibe scores based on keyword matching
        """
        text = text.lower()
        scores = {vibe: 0.0 for vibe in self.vibes}

        # Count keyword occurrences for each vibe
        for vibe, keywords in self.vibe_keywords.items():
            count = sum(1 for keyword in keywords if keyword.lower() in text)
            if count > 0:
                scores[vibe] = count / len(keywords)  # Normalize by keyword count

        return scores

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text by cleaning and normalizing.

        Args:
            text (str): Input text

        Returns:
            str: Preprocessed text
        """
        # Convert hashtags to regular words
        text = re.sub(r"#(\w+)", r"\1", text)

        # Process with spaCy
        doc = self.nlp(text)

        # Remove stopwords and lemmatize
        tokens = [token.lemma_ for token in doc if not token.is_stop]

        return " ".join(tokens)

    def get_final_vibes(self, text: str, top_k: int = 3) -> List[str]:
        """Combine zero-shot classification and keyword analysis for final vibe prediction.

        Args:
            text (str): Input text
            top_k (int): Number of top vibes to return

        Returns:
            List[str]: Top-k vibes
        """
        # Get vibes from zero-shot classification
        zs_vibes = self.classify_vibes(text, top_k=len(self.vibes))
        zs_scores = {v["vibe"]: v["confidence"] for v in zs_vibes}

        # Get vibes from keyword analysis
        kw_scores = self.analyze_keywords(text)

        # Combine scores (70% zero-shot, 30% keywords)
        final_scores = {}
        for vibe in self.vibes:
            final_scores[vibe] = 0.7 * zs_scores.get(vibe, 0) + 0.3 * kw_scores.get(
                vibe, 0
            )

        # Sort and return top-k vibes
        sorted_vibes = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [vibe for vibe, _ in sorted_vibes[:top_k]]
