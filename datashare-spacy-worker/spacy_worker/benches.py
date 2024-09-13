import asyncio
import logging
from collections.abc import AsyncIterable, Callable
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Generator,
    List,
    Tuple,
    TypeVar,
    cast,
)
from uuid import uuid4

from aiostream.stream import chain
from elasticsearch._async.helpers import async_bulk
from icij_common.logging_utils import log_elapsed_time
from icij_common.pydantic_utils import jsonable_encoder
from icij_worker import Task, TaskManager, TaskState
from pydantic import parse_obj_as

from spacy_worker.constants import DS_SPACY_NER_TASK, SPACY_NER_TASK
from spacy_worker.es import (
    DOC_CONTENT,
    DOC_LANGUAGE,
    DOC_ROOT_ID,
    ESClient,
    ID_,
    KEEP_ALIVE,
    SOURCE,
    make_document_query,
)
from spacy_worker.objects import DSDoc, DSNamedEntity, NamedEntity_
from spacy_worker.utils import iter_async

_SORT = [{"language": "asc"}]
logger = logging.getLogger(__name__)

READY_STATES = {TaskState.DONE, TaskState.ERROR, TaskState.CANCELLED}
ERROR_STATES = {TaskState.ERROR, TaskState.CANCELLED}

_ELAPSED_MSG = "performed NER in {elapsed}"


async def extract_nlp_task(
    project: str,
    task_manager: TaskManager,
    es_client: ESClient,
    *,
    batch_size: int,
    max_content_length: int,
    nlp_parallelism: int,
):
    # We poll documents from the next task, while workers are busy
    max_tasks = nlp_parallelism * 2
    with log_elapsed_time(logger, level=logging.INFO, output_msg=_ELAPSED_MSG):
        # Here
        docs = doc_stream(es_client, project)
        async for language, language_docs in _language_docs_it(docs):
            async for res in nlp_pipeline(
                language_docs,
                language,
                task_manager,
                project=project,
                max_tasks=max_tasks,
                batch_size=batch_size,
                max_content_length=max_content_length,
            ):
                tags = parse_obj_as(List[NamedEntity_], res["tags"])
                named_entities = DSNamedEntity.from_tags(
                    tags, res["doc"], language=language
                )
                # TODO: we could deal with that in a bulk async fashion...
                #  we could also batch to avoid too many ES round-trips
                await _add_doc_ne(es_client, named_entities)


# This one lives in the extension
async def nlp_pipeline(
    docs: AsyncIterable[Dict],
    language: str,
    task_manager: TaskManager,
    *,
    project: str,
    max_tasks: int,
    batch_size: int,
    max_content_length: int,
) -> AsyncGenerator[Dict, None]:
    # TODO: polling could even be done in an async fashion
    current_tasks = []
    batch = []
    results = dict()
    async for doc in docs:
        doc_content = doc[SOURCE][DOC_CONTENT]
        doc_length = len(doc_content)
        results[doc[ID_]] = {
            "n_batches": doc_length // batch_size,
            "named_entities": [],
            "root_id": doc[SOURCE][DOC_ROOT_ID],
        }
        for offset in range(0, doc_length, max_content_length):
            chunk = doc_content[offset : offset + max_content_length]
            batch_item = (doc[ID_], offset, chunk)
            if len(batch) >= batch_size:
                if len(current_tasks) >= max_tasks:
                    ready_task = await _poll_tasks(current_tasks, task_manager)
                    batch_ne = await task_manager.get_task_result(ready_task)
                    batch_ne = cast(List[Dict], batch_ne.result)
                    for res in batch_ne:
                        results[res["doc_id"]]["named_entities"].append(res["tags"])
                    for res in _ready_res(results, project):
                        yield res
                t = await _enqueue_task(task_manager, batch, language)
                current_tasks.append(t)
                batch.clear()
            batch.append(batch_item)
    if batch:
        t = await _enqueue_task(task_manager, batch, language)
        current_tasks.append(t)
        batch.clear()
    while current_tasks:
        ready_task = await _poll_tasks(current_tasks, task_manager)
        batch_ne = await task_manager.get_task_result(ready_task)
        batch_ne = cast(List[Dict], batch_ne.result)
        for res in batch_ne:
            results[res["doc_id"]]["named_entities"].append(res["tags"])
        for res in _ready_res(results, project):
            yield res


def _ready_res(results: Dict, project: str) -> Generator[Dict, None, None]:
    for doc_id, res in results.items():
        if len(res["named_entities"]) == res["n_batches"]:
            yield {
                "doc": DSDoc(id=doc_id, project=project, root_id=res["root_id"]),
                "tags": sum(res["named_entities"], []),
            }


async def _enqueue_task(
    task_manager: TaskManager, task_docs: List[Tuple[str, int, str]], language: str
) -> str:
    task_name = SPACY_NER_TASK
    task_id = f"{task_name}-{uuid4().hex}"
    args = {
        "docs": [
            {"doc_id": doc_id, "offset": offset, "text": text}
            for doc_id, offset, text in task_docs
        ],
        "language": language,
    }
    task = Task.create(task_id=task_id, task_name=task_name, args=args)
    await task_manager.save_task(task)
    await task_manager.enqueue(task)
    return task.id


# This one lives ExtractNLPTask
async def extract_nlp_task_ds(
    project: str,
    task_manager: TaskManager,
    es_client: ESClient,
    *,
    doc_batch_size: int,
    nlp_parallelism: int,
):
    # We poll documents from the next task, while workers are busy
    max_tasks = nlp_parallelism * 2
    with log_elapsed_time(logger, level=logging.INFO, output_msg=_ELAPSED_MSG):
        # Here
        docs = doc_stream(es_client, project)
        async for language, language_docs in _language_docs_it(docs):
            async for res in nlp_pipeline_ds(
                language_docs,
                language,
                task_manager,
                project=project,
                max_tasks=max_tasks,
                doc_batch_size=doc_batch_size,
            ):
                doc = DSDoc.from_es(res["doc"], project=project)
                tags = parse_obj_as(List[NamedEntity_], res["tags"])
                named_entities = DSNamedEntity.from_tags(tags, doc, language=language)
                # TODO: we could deal with that in a bulk async fashion...
                #  we could also batch to avoid too many ES round-trips
                await _add_doc_ne(es_client, named_entities)


# This one lives in the extension
async def nlp_pipeline_ds(
    docs: AsyncIterable[Dict],
    language: str,
    task_manager: TaskManager,
    *,
    project: str,
    max_tasks: int,
    doc_batch_size: int,
) -> AsyncGenerator[Dict, None]:
    # TODO: polling could even be done in an async fashion
    current_tasks = []
    task_docs = []
    async for doc in docs:
        if len(task_docs) >= doc_batch_size:
            while len(current_tasks) >= max_tasks:
                # Wait for some task to be ready
                ready_task = await _poll_tasks(current_tasks, task_manager)
                docs_ne = await task_manager.get_task_result(ready_task)
                docs_ne = cast(List[Dict], docs_ne.result)
                for res in docs_ne:
                    yield res
                # Pop the task
                current_tasks = [t for t in current_tasks if t != ready_task]
            t = await _enqueue_ds_task(task_manager, task_docs, language)
            current_tasks.append(t)
            task_docs.clear()
        task_docs.append(DSDoc.from_es(doc, project=project))
    if task_docs:
        t = await _enqueue_ds_task(task_manager, task_docs, language)
        current_tasks.append(t)
        task_docs.clear()
    # Wait for all task to be done
    while current_tasks:
        ready_task = await _poll_tasks(current_tasks, task_manager)
        docs_ne = await task_manager.get_task_result(ready_task)
        docs_ne = cast(List[Dict], docs_ne.result)
        for res in docs_ne:
            yield res
        current_tasks = [t for t in current_tasks if t != ready_task]


async def doc_stream(es_client: ESClient, project: str) -> AsyncGenerator[Dict, None]:
    query = make_document_query(None, sources=[DOC_ROOT_ID, DOC_CONTENT])
    async with es_client.pit(index=project) as pit:
        if pit is not None:
            pit[KEEP_ALIVE] = es_client.keep_alive
        async for doc in es_client.poll_search_pages(
            query, index=project, pit=pit, sort=_SORT
        ):
            yield doc


async def _enqueue_ds_task(
    task_manager: TaskManager, task_docs: List[DSDoc], language: str
) -> str:
    task_name = DS_SPACY_NER_TASK
    task_id = f"{task_name}-{uuid4().hex}"
    args = {"docs": task_docs, "language": language}
    task = Task.create(task_id=task_id, task_name=task_name, args=args)
    await task_manager.save_task(task)
    await task_manager.enqueue(task)
    return task.id


async def _poll_tasks(
    tasks: List[str], task_manager: TaskManager, interval_ms: float = 0.2
) -> str:
    while "I'm waiting for a task be done":
        for t in tasks:
            task = await task_manager.get_task(t)
            if task.state in ERROR_STATES:
                msg = f'Task(id="{t}") has state {task.state}, exiting...'
                raise ValueError(msg)
            if task.state is TaskState.DONE:
                return t
        await asyncio.sleep(interval_ms)


async def _add_doc_ne(es_client: ESClient, named_entities: List[DSNamedEntity]):
    named_entities = iter_async(jsonable_encoder(named_entities))
    await async_bulk(es_client, named_entities)


async def _language_docs_it(
    docs: AsyncIterator[Dict],
) -> AsyncGenerator[Tuple[str, AsyncGenerator[Dict, None]], None]:
    try:
        cur_doc = await anext(docs)
    except StopIteration:
        return
    current_language = cur_doc[DOC_LANGUAGE]
    while "I can poll docs":
        iter_while = lambda d: d[DOC_LANGUAGE] == current_language
        lang_docs, docs = before_and_after(docs, iter_while)
        yield current_language, lang_docs
        try:
            cur_doc = await anext(docs)
        except StopIteration:
            return
        current_language = cur_doc[DOC_LANGUAGE]


T = TypeVar("T")


def before_and_after(
    it: AsyncIterator[T], predicate: Callable[[T], bool]
) -> Tuple[AsyncIterator[T], AsyncIterator[T]]:
    it = aiter(it)
    transition = []

    async def true_iterator():
        async for elem in it:
            if predicate(elem):
                yield elem
            else:
                transition.append(elem)
                return

    remainder_iterator = chain(iter_async(transition), it)
    return true_iterator(), remainder_iterator
