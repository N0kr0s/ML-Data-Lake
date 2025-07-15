import spacy
from kb import entity_kb
from rapid import build_name_index, resolve_entity_id
from graph import visualize_kb

nlp = spacy.load("en_core_web_sm")

# индексация (name и aliases => id)
name_to_id = build_name_index(entity_kb)

def perform_ner(text):
    doc = nlp(text)
    print("=== NER ===")
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    for ent_text, label in entities:
        print(f"Entity: {ent_text}, Label: {label}")
    return entities

def show_entity_info_by_id(ent_id, visited=None, level=0):
    if visited is None:
        visited = set()
    if ent_id in visited:
        return
    visited.add(ent_id)

    entity = entity_kb[ent_id]
    indent = "  " * level
    print(f"{indent}Entity: {entity['name']}")
    print(f"{indent}  Description: {entity['description']}")
    print(f"{indent}  Wikipedia: {entity['wikipedia']}")

    for linked_id in entity.get("linked", []):
        if linked_id in entity_kb:
            show_entity_info_by_id(linked_id, visited, level + 1)

def perform_nel(entities):
    print("=== NEL (Entity Linking) ===")
    for name, label in entities:
        ent_id = resolve_entity_id(name, name_to_id)
        if ent_id:
            show_entity_info_by_id(ent_id)
        else:
            print(f"Entity: {name} - no link found")


# === Демонстрация ===

def print_result(text):
    input()
    print("="*50)
    print(text)
    print("="*50)
    entities = perform_ner(text)
    perform_nel(entities)

if __name__ == '__main__':
    text1 = "Elonn Mask and Vlademir Poutin met in the US"
    text2 = "WASHINGTON (AP) — Elon Musk, the billionaire owner of major government contractor SpaceX and a key ally of Republican presidential nominee Donald Trump, has been in regular contact with Russian President Vladimir Putin for the last two years, The Wall Street Journal reported. A person familiar with the situation, who spoke on condition of anonymity to discuss the sensitive matter, confirmed to The Associated Press that Musk and Putin have had contact through calls. The person didn’t provide additional details about the frequency of the calls, when they occurred or their content. Musk, the world’s richest man who also owns Tesla and the social platform X, has emerged as a leading voice on the American right. He’s poured millions of dollars into Trump’s presidential bid and turned the platform once known as Twitter into a site popular with Trump supporters, as well as conspiracy theorists, extremists and Russian propagandists. "

    visualize_kb(entity_kb)

    print_result(text1)
    print_result(text2)
