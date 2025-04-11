class ExplanationGenerator:
    def __init__(self, knowledge_base, reasoning_engine):
        self.knowledge_base = knowledge_base
        self.reasoning_engine = reasoning_engine

    def generate_explanation(self, features):
        reasoning_results = self.reasoning_engine.infer(features)
        return self._format_explanation(reasoning_results)

    def _format_explanation(self, reasoning_results):
        return f"Explanation: {' '.join(reasoning_results)}"
