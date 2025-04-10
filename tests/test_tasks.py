import pytest
from icij_common.es import HITS, SOURCE, ESClient, has_type
from pydantic import parse_obj_as

from datashare_spacy_worker.objects import (
    BatchDocument,
    Category,
    Document,
    NamedEntity,
    SpacySize,
)
from datashare_spacy_worker.tasks import spacy_ner_task
from tests.conftest import TEST_PROJECT


# TODO: this one would deserve a finer testing due to the multiple corner cases due to
#  interactions between batch_size, buffer_size and so on
@pytest.mark.integration
@pytest.mark.parametrize(
    ("categories", "expected_entities"),
    [
        (
            None,
            [
                NamedEntity(
                    document_id="doc-0",
                    root_document="root-0",
                    mention="Dan",
                    category=Category.PER,
                    offsets=[57],
                    extractor_language="ENGLISH",
                ),
                NamedEntity(
                    document_id="doc-0",
                    root_document="root-0",
                    mention="Paris",
                    category=Category.LOC,
                    offsets=[93, 103],
                    extractor_language="ENGLISH",
                ),
                NamedEntity(
                    document_id="doc-0",
                    root_document="root-0",
                    mention="Intel",
                    category=Category.ORG,
                    offsets=[161],
                    extractor_language="ENGLISH",
                ),
            ],
        ),
        (
            ["LOCATION", "PERSON", "ORGANIZATION"],
            [
                NamedEntity(
                    document_id="doc-0",
                    root_document="root-0",
                    mention="Dan",
                    category=Category.PER,
                    offsets=[57],
                    extractor_language="ENGLISH",
                ),
                NamedEntity(
                    document_id="doc-0",
                    root_document="root-0",
                    mention="Paris",
                    category=Category.LOC,
                    offsets=[93, 103],
                    extractor_language="ENGLISH",
                ),
                NamedEntity(
                    document_id="doc-0",
                    root_document="root-0",
                    mention="Intel",
                    category=Category.ORG,
                    offsets=[161],
                    extractor_language="ENGLISH",
                ),
            ],
        ),
        (
            ["LOCATION"],
            [
                NamedEntity(
                    document_id="doc-0",
                    root_document="root-0",
                    mention="Paris",
                    category=Category.LOC,
                    offsets=[93, 103],
                    extractor_language="ENGLISH",
                ),
            ],
        ),
    ],
)
async def test_spacy_ner_task_int(
    categories: list[Category] | None,
    expected_entities: list[NamedEntity],
    test_es_client: ESClient,
    populate_es: list[Document],  # noqa: ARG001
    batch_docs: list[BatchDocument],
) -> None:
    # Given
    docs = [d.model_dump() for d in batch_docs]
    model_size = SpacySize.SMALL
    max_length = 62

    # When
    n_docs = await spacy_ner_task(
        docs, model_size=model_size, categories=categories, max_length=max_length
    )
    body = {"query": has_type(type_field="type", type_value="NamedEntity")}
    sort = ["offsets:asc"]
    index_ents = []
    async for hits in test_es_client.poll_search_pages(
        index=TEST_PROJECT, body=body, sort=sort
    ):
        index_ents += [e[SOURCE] for e in hits[HITS][HITS]]

    # Then
    assert n_docs == 2
    index_ents = parse_obj_as(list[NamedEntity], index_ents)
    assert index_ents == expected_entities
