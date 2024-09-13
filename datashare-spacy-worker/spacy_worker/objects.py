from __future__ import annotations

import re
from collections import defaultdict
from enum import Enum, unique
from hashlib import md5
from typing import Dict, Iterable, List

from icij_common.pydantic_utils import LowerCamelCaseModel
from pydantic import Field

from spacy_worker.es import DOC_ROOT_ID, ID_, SOURCE


@unique
class Category(str, Enum):
    PER = "PERSON"
    ORG = "ORGANIZATION"
    LOC = "LOCATION"
    DATE = "DATE"
    MONEY = "MONEY"
    NUM = "NUMBER"
    UNK = "UNKNOWN"


class NamedEntity_(LowerCamelCaseModel):
    start: int
    mention: str
    category: Category

    @classmethod
    def from_spacy(cls, entity: NamedEntity, text: str) -> NamedEntity_:
        mention = text[entity.start : entity.end]
        return cls(start=entity.start, mention=mention, category=entity.category)


class NamedEntity(LowerCamelCaseModel):
    start: int
    end: int
    category: Category


class DSNamedEntity(LowerCamelCaseModel):
    id: str
    category: Category
    mention: str
    mention_norm: str
    document_id: str
    root_document: str
    extractor: str = Field(default="SPACY", const=True)
    extractor_language: str

    @classmethod
    def from_tags(
        cls, tags: Iterable[NamedEntity_], doc: DSDoc, language: str
    ) -> List[DSNamedEntity]:
        entities = defaultdict(list)
        for tag in tags:
            entities[(tag.mention, tag.category)].append(tag.start)
        ents = []
        for (mention, category), offsets in entities.items():
            # TODO: add a validator for this
            hasher = md5()
            offsets = sorted(offsets)
            mention_norm = _normalize(mention)
            hashed = (doc.id, str(offsets), "SPACY", mention_norm)
            for h in hashed:
                hasher.update(h.encode())
            ne_id = hasher.hexdigest()
            ents.append(
                DSNamedEntity(
                    id=ne_id,
                    category=category,
                    mention=mention,
                    mention_norm=mention_norm,
                    document_id=doc.id,
                    root_document=doc.root_id,
                    extractor_language=language,
                )
            )
        return ents


_CONSECUTIVE_SPACES_RE = re.compile(r"\s+")


def _normalize(s: str) -> str:
    s = s.strip()
    return _CONSECUTIVE_SPACES_RE.sub(s, " ").lower()


@unique
class SpacySize(str, Enum):
    SMALL = "sm"
    MEDIUM = "md"
    LARGE = "lg"
    TRANSFORMER = "trf"


class DSDoc(LowerCamelCaseModel):
    id: str
    project: str
    root_id: str

    @classmethod
    def from_es(cls, es_doc: Dict, project: str) -> DSDoc:
        return cls(project=project, id=es_doc[ID_], root_id=es_doc[SOURCE][DOC_ROOT_ID])
