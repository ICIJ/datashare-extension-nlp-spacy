from typing import List, Optional

import pytest
import spacy

from spacy_worker.objects import Category, NamedEntity
from spacy_worker.tasks import get_n_process, spacy_ner

_doc_0 = """In this first sentence I'm speaking about a person named Dan.

Then later I'm speaking about Paris.

To finish I'm speaking about a company named Intel.
"""

_doc_1 = "some short document"


@pytest.mark.parametrize(
    "model,expected_n_process", [("en_core_web_sm", 665), ("en_core_web_trf", 1)]
)
def test_get_n_process(model: str, expected_n_process: int):
    # Given
    max_process = 666
    language = spacy.load(model)
    # When
    n_process = get_n_process(language, max_process)
    # Then
    assert n_process == expected_n_process


@pytest.mark.parametrize(
    "categories,expected_entities",
    [
        (
            None,
            [
                [
                    NamedEntity(start=57, end=60, category=Category.PER),
                    NamedEntity(start=93, end=98, category=Category.LOC),
                    NamedEntity(start=146, end=151, category=Category.ORG),
                ],
                [],
            ],
        ),
        (
            ["LOCATION", "PERSON", "ORGANIZATION"],
            [
                [
                    NamedEntity(start=57, end=60, category=Category.PER),
                    NamedEntity(start=93, end=98, category=Category.LOC),
                    NamedEntity(start=146, end=151, category=Category.ORG),
                ],
                [],
            ],
        ),
        (
            ["LOCATION"],
            [
                [
                    NamedEntity(start=93, end=98, category=Category.LOC),
                ],
                [],
            ],
        ),
    ],
)
async def test_spacy_ner_task(
    categories: Optional[List[str]],
    expected_entities: List[List[NamedEntity]],
    app_lifetime_deps,  # pylint: disable=unused-argument
):
    # Given
    texts = [_doc_0, _doc_1]
    language = "en"

    # When
    entities = await spacy_ner(texts, language, categories=categories)

    # Then
    assert entities == expected_entities
