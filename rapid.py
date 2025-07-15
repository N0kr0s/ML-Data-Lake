from rapidfuzz import process


def build_name_index(kb):
    name_to_id = {}
    for eid, data in kb.items():
        name_to_id[data["name"]] = eid
        for alias in data.get("aliases", []):
            name_to_id[alias] = eid
    return name_to_id


def resolve_entity_id(name, name_to_id):
    if name in name_to_id:
        return name_to_id[name]

    best_match = process.extractOne(name, name_to_id.keys(), score_cutoff=80)
    if best_match:
        return name_to_id[best_match[0]]

    return None
