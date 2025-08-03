import time
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np

from SPARQLWrapper import SPARQLWrapper, JSON

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


# === NEL using Wikidata ===
def wikidata_nel(entities):
    """
    Performs Named Entity Linking using Wikidata.
    :param entities: List of tuples (entity_text, entity_label) from spaCy NER.
    :return: List of tuples (entity_text, wikidata_qid).
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    linked_entities = []

    for ent_text, ent_label in entities:
        # Базовый запрос для поиска элемента по метке
        query = f"""
        SELECT ?item ?itemLabel WHERE {{
          ?item rdfs:label "{ent_text}"@en.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        LIMIT 5
        """

        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
            bindings = results.get("results", {}).get("bindings", [])
            if bindings:
                # Простой выбор: берем первый результат.
                best_match = bindings[0]
                qid = best_match["item"]["value"].split("/")[-1]  # Извлекаем QID из URI
                linked_entities.append((ent_text, qid))
                print(
                    f"Linked '{ent_text}' to Wikidata item: {qid} ({best_match.get('itemLabel', {}).get('value', 'No Label')})")
            else:
                print(f"No Wikidata item found for '{ent_text}'")
        except Exception as e:
            print(f"Error querying Wikidata for '{ent_text}': {e}")
            # Можно добавить кортеж с None
            # linked_entities.append((ent_text, None))

    print("\n=== NEL (Wikidata) ===")
    print(linked_entities)
    return linked_entities


def get_wikidata_descriptions(linked_entities):
    """
    Получает краткие описания для списка QID из Wikidata.
    :param linked_entities: Список кортежей (entity_text, qid).
    :return: Словарь {qid: description}.
    """
    if not linked_entities:
        return {}

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    descriptions = {}
    qids = [qid for _, qid in linked_entities if qid]

    if not qids:
        return descriptions

    # Создаем VALUES блок для запроса нескольких QID одновременно
    values_clause = " ".join([f"wd:{qid}" for qid in qids])
    query = f"""
    SELECT ?item ?itemDescription WHERE {{
      VALUES ?item {{ {values_clause} }}
      OPTIONAL {{
        ?item schema:description ?itemDescription.
        FILTER(LANG(?itemDescription) = "en")
      }}
    }}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])
        for binding in bindings:
            qid_uri = binding["item"]["value"]
            qid = qid_uri.split("/")[-1]
            description = binding.get("itemDescription", {}).get("value", "No description available")
            descriptions[qid] = description
    except Exception as e:
        print(f"Error fetching descriptions: {e}")

    return descriptions


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


# === Graph builder for Wikidata ===
def build_graph_wikidata(relations, linked_entities_map):
    """
    Строит граф, используя QID из Wikidata.
    :param relations: Список кортежей (source_name, relation, target_name).
    :param linked_entities_map: Словарь {entity_name: qid}.
    :return: NetworkX DiGraph.
    """
    G = nx.DiGraph()
    for src_name, rel, tgt_name in relations:
        src_qid = linked_entities_map.get(src_name)
        tgt_qid = linked_entities_map.get(tgt_name)
        if src_qid and tgt_qid:
            # Можно использовать QID как ID узлов
            G.add_edge(src_qid, tgt_qid, label=rel, weight=1)
        else:
            print(f"Warning: Could not find QID for {src_name} or {tgt_name}. Relation not added.")
    return G


# === Co-occurrence ===
def cooccurrence_score(linked_ents):
    co_matrix = defaultdict(lambda: defaultdict(int))
    for i in range(len(linked_ents)):
        for j in range(i + 1, len(linked_ents)):
            e1, e2 = linked_ents[i][1], linked_ents[j][1]
            co_matrix[e1][e2] += 1
            co_matrix[e2][e1] += 1
    return dict(co_matrix)


# === Cosine similarity ===
def cosine_similarity_score(id_to_desc):
    if not id_to_desc:
        return {}
    ids = list(id_to_desc.keys())
    corpus = [id_to_desc[i] for i in ids]
    if not corpus:
        return {}
    vectorizer = CountVectorizer().fit_transform(corpus)
    vectors = vectorizer.toarray()
    sim_matrix = cosine_similarity(vectors)
    return {(ids[i], ids[j]): sim_matrix[i][j] for i in range(len(ids)) for j in range(len(ids)) if i < j}


# === Graph Distance ===
def graph_distance(G):
    if len(G.nodes()) == 0:
        return {}
    return dict(nx.all_pairs_shortest_path_length(G))


# === PageRank ===
def pagerank_score(G):
    if len(G.nodes()) == 0:
        return {}
    return nx.pagerank(G)


# === Fake embeddings ===
def fake_embedding_score(id_to_desc):
    if not id_to_desc:
        return {}
    np.random.seed(42)
    emb = {eid: np.random.rand(5) for eid in id_to_desc}
    result = {}
    ids = list(id_to_desc.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            sim = np.dot(emb[ids[i]], emb[ids[j]]) / (
                    np.linalg.norm(emb[ids[i]]) * np.linalg.norm(emb[ids[j]]))
            result[(ids[i], ids[j])] = sim
    return result


# === Main ===
text = """Elon Musk Donated $15 Million To Trump's MAGA And GOP Just 3 Days Before His Third Party Bid

Elon Musk, the CEO of Tesla Inc. and SpaceX, donated $15 million to President Donald Trump and the Republican Party, days before he called for the formation of a third party.
Musk’s Donations Disclosed In Campaign Finance Report

Musk made the donations in late June. The donations were disclosed in the campaign finance reports of the pro-Trump MAGA Inc., Senate Leadership Fund, and Congressional Leadership Fund super PACs, reported CNN.

On June 27, Musk contributed $5 million each to pro-Trump MAGA Inc., the Senate Leadership Fund and the Congressional Leadership Fund. The donations were disclosed in the campaign finance report of each entity. However, on June 30, he threatened to establish a third party, the America Party, if Congress passed the “big, beautiful bill.” Musk also donated over $45 million to the America PAC in the first half of 2025.

Don’t Miss:

    Trending: 7,000+ investors have joined Timeplast's mission to eliminate microplastics—now it's your turn to invest in the future of sustainable plastic before time runs out.

    This AI-Powered Trading Platform Has 5,000+ Users, 27 Pending Patents, and a $43.97M Valuation — You Can Become an Investor for Just $500.25

Trump's MAGA Inc. Raises Nearly $200 Million For 2026 Midterms

The Republican Party, under Trump’s leadership, has been aggressively fundraising for the 2026 midterm elections. Trump’s super PAC, MAGA Inc., raised $177 million in the first half of the year, leaving it with nearly $200 million in available funds.

Besides Musk, the donor list featured billionaire Wall Street trader Jeffrey Yass with a $16 million contribution, cosmetics heir Ronald Lauder with $5 million, and Silicon Valley investor Marc Andreessen, who donated $3 million.
Trump-Musk Rift Complicates SpaceX's Role In US Defense

Musk’s relationship with the Trump administration has been tumultuous since he left in late May. Musk’s donations to the Republican Party and his subsequent change of heart are noteworthy in the context of the current political landscape.

Musk has been publicly critical of Trump’s policies, such as the axing of subsidies on renewable energy. This criticism has been accompanied by Musk’s assertion that SpaceX won NASA contracts by delivering superior performance at lower costs.

Meanwhile, the Trump administration’s reported consideration of other partners, including Amazon’s (NASDAQ:AMZN) Project Kuiper, for the Golden Dome missile defense system has raised questions about SpaceX’s future role in national defense projects.

Read Next:

    $100k+ in investable assets? Match with a fiduciary advisor for free to learn how you can maximize your retirement and save on taxes – no cost, no obligation.

    Bezos' Favorite Real Estate Platform Launches A Way To Ride The Ongoing Private Credit Boom

Image via Shutterstock

"ACTIVE INVESTORS' SECRET WEAPON" Supercharge Your Stock Market Game with the #1 "news & everything else" trading tool: Benzinga Pro - Click here to start Your 14-Day Trial Now!

Get the latest stock analysis from Benzinga?

This article Elon Musk Donated $15 Million To Trump's MAGA And GOP Just 3 Days Before His Third Party Bid originally appeared on Benzinga.com

© 2025 Benzinga.com. Benzinga does not provide investment advice. All rights reserved."""

# 1. Выполняем NER
entities = perform_ner(text)

# 2. Выполняем NEL с Wikidata
linked_entities = wikidata_nel(entities)

# 3. Получаем описания для Wikidata сущностей
id_to_desc = {}
if linked_entities:
    wikidata_descriptions = get_wikidata_descriptions(linked_entities)
    # id_to_desc теперь будет использовать QID как ключи
    id_to_desc = {qid: wikidata_descriptions.get(qid, "No description") for _, qid in linked_entities if qid}

# 4. Извлекаем отношения
relations = extract_relations(text)

# 5. Строим граф с Wikidata QID
linked_entities_map = dict(linked_entities)
G = build_graph_wikidata(relations, linked_entities_map)

# 6. Выполняем различные оценки
for name, method in [
    ("Co-occurrence", lambda: cooccurrence_score(linked_entities)),
    ("Cosine Similarity", lambda: cosine_similarity_score(id_to_desc)),
    ("Graph Distance", lambda: graph_distance(G)),
    ("PageRank", lambda: pagerank_score(G)),
    ("Embedding Distance (Simulated)", lambda: fake_embedding_score(id_to_desc))]:
    start = time.time()
    try:
        result = method()
        end = time.time()
        print(f"\n=== {name} ===")
        for k, v in result.items():
            print(f"{k}: {v}")
        print(f"Time: {end - start:.4f} sec")
    except Exception as e:
        end = time.time()
        print(f"\n=== {name} ===")
        print(f"Error: {e}")
        print(f"Time: {end - start:.4f} sec")