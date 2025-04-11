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
