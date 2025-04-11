from rdflib import Graph, URIRef, Literal, Namespace

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
