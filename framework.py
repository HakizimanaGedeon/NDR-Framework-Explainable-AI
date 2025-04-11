from .knowledge_base import KnowledgeBase
from .inference_engine import InferenceEngine
from .explanation_generator import ExplanationGenerator

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
