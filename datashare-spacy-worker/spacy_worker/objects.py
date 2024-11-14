from __future__ import annotations

import re
from collections import defaultdict
from enum import Enum, unique
from hashlib import md5

from icij_common.es import (
    DOC_CONTENT,
    DOC_CONTENT_LENGTH,
    DOC_LANGUAGE,
    DOC_ROOT_ID,
    ID_,
    SOURCE,
)
from icij_common.pydantic_utils import LowerCamelCaseModel, safe_copy
from pydantic import Field, root_validator
from typing_extensions import Self

from spacy_worker.utils import lower_camel_to_snake_case


@unique
class SpacySize(str, Enum):
    SMALL = "sm"
    MEDIUM = "md"
    LARGE = "lg"
    TRANSFORMER = "trf"


@unique
class Category(str, Enum):
    PER = "PERSON"
    ORG = "ORGANIZATION"
    LOC = "LOCATION"
    DATE = "DATE"
    MONEY = "MONEY"
    NUM = "NUMBER"
    UNK = "UNKNOWN"


class BatchDocument(LowerCamelCaseModel):
    id: str
    root_document: str
    project: str
    language: str

    @classmethod
    def from_es(cls, es_doc: dict, project: str) -> Self:
        sources = es_doc[SOURCE]
        return cls(
            project=project,
            id=es_doc[ID_],
            root_id=sources[DOC_ROOT_ID],
            language=sources[DOC_LANGUAGE],
        )


class Document(BatchDocument):
    content: str
    content_length: int

    @classmethod
    def from_es(cls, es_doc: dict, project: str) -> Self:
        sources = es_doc[SOURCE]
        return cls(
            project=project,
            id=es_doc[ID_],
            root_id=sources[DOC_ROOT_ID],
            language=sources[DOC_LANGUAGE],
            content=sources[DOC_CONTENT],
            content_length=sources[DOC_CONTENT_LENGTH],
        )


class NlpTag(LowerCamelCaseModel):
    start: int
    mention: str
    category: Category

    def with_offset(self, offset: int) -> Self:
        return safe_copy(self, update={"start": self.start + offset})


SPACY_PIPELINE_NAME = "SPACY"
_ID_PLACEHOLDER, _MENTION_NORM_PLACEHOLDER = "", ""


class NamedEntity(LowerCamelCaseModel):
    id: str = _ID_PLACEHOLDER
    category: Category
    mention: str
    mention_norm: str = _ID_PLACEHOLDER
    document_id: str
    root_document: str
    extractor: str = Field(default=SPACY_PIPELINE_NAME, const=True)
    type: str = Field(default="NamedEntity", const=True)
    extractor_language: str
    offsets: list[int]

    @root_validator(pre=True)
    def generate_id(cls, values: dict) -> dict:
        values = {lower_camel_to_snake_case(k): v for k, v in values.items()}
        provided_mention_norm = values.get("mention_norm")
        hasher = md5()
        offsets = values.get("offsets", [])
        offsets = sorted(offsets)
        mention = values.get("mention", "")
        mention_norm = _normalize(mention)
        if provided_mention_norm is not None and provided_mention_norm != mention_norm:
            msg = (
                f'provided mention_norm ("{provided_mention_norm}") differs from '
                f'computed mention norm ("{mention_norm}")'
            )
            raise ValueError(msg)
        values["mention_norm"] = mention_norm
        doc_id = values.get("document_id", "")
        hashed = (doc_id, str(offsets), SPACY_PIPELINE_NAME, mention_norm)
        for h in hashed:
            hasher.update(h.encode())
        ne_id = hasher.hexdigest()
        provided_id = values.get("id")
        if provided_id is not None and provided_id != ne_id:
            msg = f'provided id ("{provided_id}") differs from computed id ("{ne_id}")'
            raise ValueError(msg)
        values["id"] = ne_id
        return values

    @classmethod
    def from_tags(cls, tags: list[NlpTag], doc: BatchDocument) -> list[Self]:
        entities = defaultdict(list)
        for tag in tags:
            entities[(tag.mention, tag.category)].append(tag.start)
        ents = []
        for (mention, category), offsets in entities.items():
            ents.append(
                NamedEntity(
                    category=category,
                    mention=mention,
                    document_id=doc.id,
                    root_document=doc.root_document,
                    extractor_language=doc.language,
                    offsets=offsets,
                )
            )
        return ents


_CONSECUTIVE_SPACES_RE = re.compile(r"\s+")


def _normalize(s: str) -> str:
    s = s.strip()
    return _CONSECUTIVE_SPACES_RE.sub(" ", s).lower()
