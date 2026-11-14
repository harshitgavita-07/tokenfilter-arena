"""A small controlled world: entities, relations, ground-truth values.

The experiment needs factual statements whose truth we fully control, so
the world is synthetic and self-contained. 300 train entities + 60 held
out test entities across 6 relation types. Values are shared per
relation so "corruption" (swapping in a wrong-but-plausible value) is
meaningful.
"""
from __future__ import annotations

import random

RELATIONS = {
    "city_country": {"entity": "city", "question": "What country is {e} in?",
                     "values": ["France", "Japan", "Brazil", "Canada", "Egypt", "India", "Spain", "Kenya"]},
    "person_birthyear": {"entity": "scientist", "question": "In what year was {e} born?",
                         "values": [str(y) for y in range(1850, 1951, 4)]},
    "company_industry": {"entity": "company", "question": "What industry is {e} in?",
                         "values": ["aviation", "pharmaceuticals", "shipping", "telecom", "agriculture", "mining"]},
    "river_continent": {"entity": "river", "question": "On what continent is the {e}?",
                        "values": ["Africa", "Asia", "Europe", "South America", "North America", "Australia"]},
    "language_family": {"entity": "language", "question": "What language family does {e} belong to?",
                        "values": ["Romance", "Germanic", "Slavic", "Semitic", "Bantu", "Dravidian"]},
    "peak_range": {"entity": "mountain", "question": "What range is {e} part of?",
                   "values": ["Andes", "Alps", "Atlas", "Urals", "Zagros", "Sierra Madre"]},
}

# pronounceable synthetic entity names, deterministic per index
_SYL = ["ka", "lor", "mi", "den", "sha", "vol", "tri", "ben", "qu", "az",
        "nor", "fel", "is", "tan", "ru", "pel", "ga", "mon", "eth", "ur"]


def entity_name(i: int) -> str:
    r = random.Random(1000 + i)
    name = "".join(_SYL[r.randrange(len(_SYL))] for _ in range(3))
    return name.capitalize()


def build_world(n_train: int = 300, n_test: int = 60, seed: int = 7) -> dict:
    """Return {'train': [(rel, entity, value)], 'test': [...]} with
    disjoint entities; values cycle through the relation's value set."""
    rng = random.Random(seed)
    rels = sorted(RELATIONS)
    train, test = [], []
    seen: set[str] = set()
    for i in range(n_train + n_test):
        rel = rels[i % len(rels)]
        values = RELATIONS[rel]["values"]
        value = values[(i // len(rels) + rng.randrange(0)) % len(values)] if False else values[(i // len(rels)) % len(values)]
        name = entity_name(i)
        while name in seen:  # synthetic names can collide; guarantee uniqueness
            name = entity_name(i + 100000 + len(seen))
        seen.add(name)
        row = (rel, name, value)
        (train if i < n_train else test).append(row)
    return {"train": train, "test": test}


def declarative(rel: str, entity: str, value: str) -> str:
    q = RELATIONS[rel]["question"].format(e=entity)
    # "What country is X in?" -> "X is in France." style, relation-generic:
    subj = RELATIONS[rel]["entity"]
    return f"{entity} is a {subj}. {q} {value}."


def qa(rel: str, entity: str, value: str) -> tuple[str, str]:
    return RELATIONS[rel]["question"].format(e=entity), value
