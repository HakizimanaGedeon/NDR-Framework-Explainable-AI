import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from rdflib import Graph, URIRef, Literal, Namespace

# -------------------------------------------
# Knowledge Base
# -------------------------------------------
class KnowledgeBase:
    def __init__(self):
        self.graph = Graph()
        self.namespace = Namespace("http://example.org/ontology/")
        self.graph.bind("ex", self.namespace)

    def add_or_update_law(self, law_uri, law_description):
        law = URIRef(self.namespace[law_uri])
        self.graph.set((law, self.namespace.description, Literal(law_description)))

    def add_or_update_antecedent(self, antecedent_uri, antecedent_description):
        antecedent = URIRef(self.namespace[antecedent_uri])
        self.graph.set((antecedent, self.namespace.description, Literal(antecedent_description)))

    def get_laws(self):
        return list(self.graph.subjects(predicate=self.namespace.description))

# -------------------------------------------
# Inference Engine (NDR Core Logic)
# -------------------------------------------
class InferenceEngine:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base

    def infer(self, features):
        conclusions = []
        if features['mean_radius'] > 15:
            conclusions.append("Law1: Large mean radius indicates higher likelihood of malignancy.")
        if features['smoothness'] > 0.1:
            conclusions.append("Law2: High smoothness indicates possible malignancy.")
        if features['mean_texture'] > 20:
            conclusions.append("Law3: High texture indicates a higher chance of malignancy.")
        return conclusions

# -------------------------------------------
# Explanation Generator
# -------------------------------------------
class ExplanationGenerator:
    def __init__(self, knowledge_base, reasoning_engine):
        self.knowledge_base = knowledge_base
        self.reasoning_engine = reasoning_engine

    def generate_explanation(self, features):
        reasoning_results = self.reasoning_engine.infer(features)
        return self._format_explanation(reasoning_results)

    def _format_explanation(self, reasoning_results):
        return f"Explanation: {' '.join(reasoning_results)}"

# -------------------------------------------
# NDR Framework Integration
# -------------------------------------------
class NDRFramework:
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.inference_engine = InferenceEngine(self.knowledge_base)
        self.explanation_generator = ExplanationGenerator(self.knowledge_base, self.inference_engine)

    def add_law(self, law_uri, law_description):
        self.knowledge_base.add_or_update_law(law_uri, law_description)

    def add_antecedent(self, antecedent_uri, antecedent_description):
        self.knowledge_base.add_or_update_antecedent(antecedent_uri, antecedent_description)

    def generate_explanation(self, features):
        return self.explanation_generator.generate_explanation(features)

# -------------------------------------------
# Feature Extraction using InceptionV3
# -------------------------------------------
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def extract_features(image_path):
    img_preprocessed = load_and_preprocess_image(image_path)
    features = base_model.predict(img_preprocessed)
    return features.flatten()  # Flatten to a 1D array

# -------------------------------------------
# Example Usage (Testing the Framework)
# -------------------------------------------
if __name__ == "__main__":
    # Initialize NDR framework
    ndr = NDRFramework()
    
    # Adding basic laws and antecedents based on dataset features
    ndr.add_law("Law1", "If Mean Radius is high, the likelihood of cancer increases.")
    ndr.add_law("Law2", "If Smoothness is high, the likelihood of cancer is higher.")
    ndr.add_law("Law3", "If Mean Texture is high, the likelihood of cancer increases.")
    ndr.add_antecedent("Antecedent1", "Mean Radius > 15")
    ndr.add_antecedent("Antecedent2", "Smoothness > 0.1")
    ndr.add_antecedent("Antecedent3", "Mean Texture > 20")

    # Example: Choose a random test sample (instead of using the model to predict, we extract the features directly)
    features = {'mean_radius': 16, 'smoothness': 0.2, 'mean_texture': 22}

    # Generate explanation
    explanation = ndr.generate_explanation(features)
    print(f"Generated Explanation: {explanation}")
