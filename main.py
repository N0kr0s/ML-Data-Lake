import time
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np

# === Real NER using spaCy ===
import spacy
nlp = spacy.load("en_core_web_sm")

def perform_ner(text):
    doc = nlp(text)
    print("=== NER ===")
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    for ent_text, label in entities:
        print(f"Entity: {ent_text}, Label: {label}")
    return entities

# === NEL (Simple dictionary) ===
knowledge_base = {
    "Apple": {"id": 1, "aliases": ["Apple Inc."], "desc": "Technology company."},
    "OpenAI": {"id": 2, "aliases": ["Open AI"], "desc": "Artificial intelligence research lab."},
    "John Smith": {"id": 3, "aliases": ["J. Smith"], "desc": "Software engineer."},
    "Microsoft": {"id": 4, "aliases": ["MSFT"], "desc": "Software and hardware company."}
}

def simple_nel(entities):
    linked_entities = []
    for ent_text, ent_label in entities:
        for key, val in knowledge_base.items():
            if ent_text == key or ent_text in val["aliases"]:
                linked_entities.append((ent_text, val["id"]))
                break
    print("\n=== NEL ===")
    print(linked_entities)
    return linked_entities

# === Relation extraction (simplified) ===
def extract_relations(text):
    relations = []
    sentences = text.split('.')
    for sent in sentences:
        if "acquired" in sent:
            relations.append(("Apple", "acquired", "OpenAI"))
        if "left" in sent and "to work for" in sent:
            relations.append(("John Smith", "left", "Microsoft"))
            relations.append(("John Smith", "joined", "OpenAI"))
    print("\n=== Extracted Relations ===")
    for r in relations:
        print(r)
    return relations

# === Graph builder ===
def build_graph(relations):
    G = nx.DiGraph()
    for src, rel, tgt in relations:
        src_id = knowledge_base.get(src, {}).get("id")
        tgt_id = knowledge_base.get(tgt, {}).get("id")
        if src_id and tgt_id:
            G.add_edge(src_id, tgt_id, label=rel, weight=1)
    return G

# === Co-occurrence ===
def cooccurrence_score(linked_ents):
    co_matrix = defaultdict(lambda: defaultdict(int))
    for i in range(len(linked_ents)):
        for j in range(i+1, len(linked_ents)):
            e1, e2 = linked_ents[i][1], linked_ents[j][1]
            co_matrix[e1][e2] += 1
            co_matrix[e2][e1] += 1
    return dict(co_matrix)

# === Cosine similarity ===
def cosine_similarity_score(id_to_desc):
    ids = list(id_to_desc.keys())
    corpus = [id_to_desc[i] for i in ids]
    vectorizer = CountVectorizer().fit_transform(corpus)
    vectors = vectorizer.toarray()
    sim_matrix = cosine_similarity(vectors)
    return {(ids[i], ids[j]): sim_matrix[i][j] for i in range(len(ids)) for j in range(len(ids)) if i < j}

# === Graph Distance ===
def graph_distance(G):
    return dict(nx.all_pairs_shortest_path_length(G))

# === PageRank ===
def pagerank_score(G):
    return nx.pagerank(G)

# === Fake embeddings ===
def fake_embedding_score(id_to_desc):
    np.random.seed(42)
    emb = {eid: np.random.rand(5) for eid in id_to_desc}
    result = {}
    ids = list(id_to_desc.keys())
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            sim = np.dot(emb[ids[i]], emb[ids[j]]) / (
                np.linalg.norm(emb[ids[i]]) * np.linalg.norm(emb[ids[j]]))
            result[(ids[i], ids[j])] = sim
    return result

# === Main ===
text = "Apple acquired OpenAI. John Smith left Microsoft to work for OpenAI."
entities = perform_ner(text)
linked_entities = simple_nel(entities)
id_to_desc = {v['id']: v['desc'] for v in knowledge_base.values() if v['id'] in [eid for _, eid in linked_entities]}
relations = extract_relations(text)
G = build_graph(relations)

for name, method in [
    ("Co-occurrence", lambda: cooccurrence_score(linked_entities)),
    ("Cosine Similarity", lambda: cosine_similarity_score(id_to_desc)),
    ("Graph Distance", lambda: graph_distance(G)),
    ("PageRank", lambda: pagerank_score(G)),
    ("Embedding Distance (Simulated)", lambda: fake_embedding_score(id_to_desc))]:
    start = time.time()
    result = method()
    end = time.time()
    print(f"\n=== {name} ===")
    for k, v in result.items():
        print(f"{k}: {v}")
    print(f"Time: {end - start:.4f} sec")
