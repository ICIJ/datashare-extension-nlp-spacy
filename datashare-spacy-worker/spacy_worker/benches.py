import asyncio
import logging
import time
from collections.abc import AsyncIterable, Callable
from copy import copy
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

import pycountry
import spacy
from aiohttp import BasicAuth
from aiostream.stream import chain
from elasticsearch._async.helpers import async_bulk
from icij_common.logging_utils import log_elapsed_time, log_elapsed_time_cm
from icij_common.pydantic_utils import jsonable_encoder
from icij_worker import (
    AMQPTaskManager,
    PostgresStorageConfig,
    Task,
    TaskManager,
    TaskState,
)
from icij_worker.utils import run_deps
from icij_worker.utils.amqp import AMQPManagementClient
from pydantic import parse_obj_as

from spacy_worker.app import app
from spacy_worker.constants import DS_SPACY_NER_TASK, SPACY_NER_TASK
from spacy_worker.es import (
    ASC,
    DOC_CONTENT,
    DOC_LANGUAGE,
    DOC_ROOT_ID,
    ESClient,
    HITS,
    ID_,
    KEEP_ALIVE,
    SHARD_DOC_,
    SORT,
    SOURCE,
    make_document_query,
    with_pit,
)
from spacy_worker.objects import DSDoc, DSNamedEntity, NamedEntity_, SpacySize
from spacy_worker.tasks import lifespan_es_client
from spacy_worker.tasks.dependencies import (
    es_client_enter,
    es_client_exit,
    load_app_config,
)
from spacy_worker.utils import iter_async

_SORT = f"{DOC_LANGUAGE}:{ASC},{SHARD_DOC_}:{ASC}"
logger = logging.getLogger(__name__)

READY_STATES = {TaskState.DONE, TaskState.ERROR, TaskState.CANCELLED}
ERROR_STATES = {TaskState.ERROR, TaskState.CANCELLED}

_ELAPSED_MSG = "performed NER in {elapsed}"

_SUPPORTED_LANGUAGES = set(l.split("_")[0] for l in spacy.info()["pipelines"].keys())


async def extract_nlp_task(
    project: str,
    task_manager: TaskManager,
    es_client: ESClient,
    *,
    batch_size: int,
    max_content_length: int,
    nlp_parallelism: int,
    size: SpacySize,
):
    # We poll documents from the next task, while workers are busy
    max_tasks = nlp_parallelism * 2
    with log_elapsed_time_cm(logger, level=logging.INFO, output_msg=_ELAPSED_MSG):
        start = time.process_time()
        task_buffer = []
        res_buffer = dict()
        can_consume = asyncio.Event()
        can_produce = asyncio.Event()
        can_produce.set()
        produce_tasks = produce_ner_tasks(
            project,
            task_manager,
            es_client,
            task_buffer,
            res_buffer,
            can_consume=can_consume,
            can_produce=can_produce,
            batch_size=batch_size,
            max_tasks=max_tasks,
            max_content_length=max_content_length,
            size=size,
        )
        produce_tasks = asyncio.create_task(produce_tasks)
        await can_consume.wait()
        consume_results = consume_ner_results(
            project,
            produce_tasks,
            task_manager,
            es_client,
            task_buffer,
            res_buffer,
            max_tasks,
            can_produce=can_produce,
            start=start,
        )
        consume_results = asyncio.create_task(consume_results)
        await produce_tasks
        await consume_results


async def produce_ner_tasks(
    project: str,
    task_manager: TaskManager,
    es_client: ESClient,
    task_buffer: List[str],
    res_buffer: Dict[str, Dict],
    *,
    can_produce: asyncio.Event,
    can_consume: asyncio.Event,
    batch_size: int,
    max_tasks: int,
    max_content_length: int,
    size: SpacySize,
):
    # Here
    docs = doc_stream(es_client, project)
    async for language, language_docs in _language_docs_it(docs):
        await nlp_pipeline(
            language_docs,
            language,
            task_manager,
            task_buffer,
            res_buffer,
            can_produce=can_produce,
            can_consume=can_consume,
            batch_size=batch_size,
            max_content_length=max_content_length,
            max_tasks=max_tasks,
            size=size,
        )


async def consume_ner_results(
    project: str,
    produce_task: asyncio.Task,
    task_manager: TaskManager,
    es_client: ESClient,
    task_buffer: List[str],
    res_buffer: Dict[str, Dict],
    max_tasks: int,
    can_produce: asyncio.Event,
    start: float,
):
    ne_buffer = []
    n_docs = 0
    n_tokens = 0
    while not produce_task.done():
        ready_task = await _poll_tasks(task_buffer, task_manager)
        if len(task_buffer) <= max_tasks:
            can_produce.set()
        else:
            can_produce.clear()
        batch_ne = await task_manager.get_task_result(ready_task)
        batch_ne = cast(List[Dict], batch_ne.result)
        for res in batch_ne:
            res_buffer[res["doc_id"]]["named_entities"].append(res["tags"])
        for res in _ready_res(res_buffer, project):
            n_docs += 1
            n_tokens += res["n_tokens"]
            tags = parse_obj_as(List[NamedEntity_], res["tags"])
            named_entities = DSNamedEntity.from_tags(
                tags, res["doc"], language=res["doc"].language
            )
            ne_buffer.extend(named_entities)
            # TODO: update to 1000
            if len(ne_buffer) >= 1000:
                elapsed = time.process_time() - start
                tok_speed = int(n_tokens / elapsed)
                doc_speed = int(n_docs / elapsed)
                await _add_doc_ne(es_client, ne_buffer, project)
                logger.info("saved %s docs", n_docs)
                logger.info("%s tokens/s, %s doc/s", tok_speed, doc_speed)
                ne_buffer.clear()
            res_buffer.pop(res["doc"].id)
    if ne_buffer:
        await _add_doc_ne(es_client, ne_buffer, project)


# This one lives in the extension
async def nlp_pipeline(
    docs: AsyncIterable[Dict],
    language: str,
    task_manager: TaskManager,
    task_buffer: List[str],
    res_buffer: Dict[str, Dict],
    can_produce: asyncio.Event,
    can_consume: asyncio.Event,
    *,
    max_tasks: int,
    batch_size: int,
    max_content_length: int,
    size: SpacySize,
):
    # TODO: polling could even be done in an async fashion
    batch = []
    async for doc in docs:
        doc_content = doc[SOURCE][DOC_CONTENT]
        doc_length = len(doc_content)
        res_buffer[doc[ID_]] = {
            "n_batches": doc_length // batch_size + 1,
            "named_entities": [],
            "root_id": doc[SOURCE][DOC_ROOT_ID],
            "language": language,
            "n_tokens": doc_length,
        }
        for offset in range(0, doc_length, max_content_length):
            chunk = doc_content[offset : offset + max_content_length]
            batch_item = (doc[ID_], offset, chunk)
            if len(batch) >= batch_size:
                if not can_consume.is_set():
                    can_consume.set()
                await can_produce.wait()
                t_id = await _enqueue_task(task_manager, batch, language, size)
                task_buffer.append(t_id)
                if len(task_buffer) >= max_tasks:
                    can_produce.clear()
                elif not can_consume.is_set():
                    can_produce.set()
                batch.clear()
            batch.append(batch_item)
    if batch:
        await can_produce.wait()
        t_id = await _enqueue_task(task_manager, batch, language, size)
        task_buffer.append(t_id)
        batch.clear()


def _ready_res(results: Dict, project: str) -> Generator[Dict, None, None]:
    for doc_id, res in copy(results).items():
        if len(res["named_entities"]) == res["n_batches"]:
            yield {
                "doc": DSDoc(
                    id=doc_id,
                    project=project,
                    root_id=res["root_id"],
                    language=res["language"],
                ),
                "tags": sum(res["named_entities"], []),
                "n_tokens": res["n_tokens"],
            }


async def _enqueue_task(
    task_manager: TaskManager,
    task_docs: List[Tuple[str, int, str]],
    language: str,
    size: SpacySize,
) -> str:
    task_name = SPACY_NER_TASK
    task_id = f"{task_name}-{uuid4().hex}"
    language = pycountry.languages.get(name=language).alpha_2
    args = {
        "docs": [
            {"doc_id": doc_id, "offset": offset, "text": text}
            for doc_id, offset, text in task_docs
        ],
        "language": language,
        "size": size,
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
    size: SpacySize,
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
                size=size,
            ):
                doc = DSDoc.from_es(res["doc"], project=project)
                tags = parse_obj_as(List[NamedEntity_], res["tags"])
                named_entities = DSNamedEntity.from_tags(tags, doc, language=language)
                # TODO: we could deal with that in a bulk async fashion...
                #  we could also batch to avoid too many ES round-trips
                await _add_doc_ne(es_client, named_entities, project)


# This one lives in the extension
async def nlp_pipeline_ds(
    docs: AsyncIterable[Dict],
    language: str,
    task_manager: TaskManager,
    *,
    project: str,
    max_tasks: int,
    doc_batch_size: int,
    size: SpacySize,
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
            t = await _enqueue_ds_task(task_manager, task_docs, language, size)
            current_tasks.append(t)
            task_docs.clear()
        task_docs.append(DSDoc.from_es(doc, project=project))
    if task_docs:
        t = await _enqueue_ds_task(task_manager, task_docs, language, size)
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
    query = make_document_query(dict())
    async with es_client.pit(index=project) as pit:
        if pit is not None:
            pit[KEEP_ALIVE] = es_client.keep_alive
        query = with_pit(query, pit)
        async for doc in es_client.poll_search_pages(
            query,
            sort=_SORT,
            _source_includes=[DOC_ROOT_ID, DOC_CONTENT, DOC_LANGUAGE, SORT],
        ):
            for d in doc[HITS][HITS]:
                language = pycountry.languages.get(name=d[SOURCE][DOC_LANGUAGE]).alpha_2
                if language in _SUPPORTED_LANGUAGES:
                    yield d


async def _enqueue_ds_task(
    task_manager: TaskManager, task_docs: List[DSDoc], language: str, size: SpacySize
) -> str:
    task_name = DS_SPACY_NER_TASK
    task_id = f"{task_name}-{uuid4().hex}"
    task_docs = jsonable_encoder(task_docs)
    language = pycountry.languages.get(name=language).alpha_2
    args = {"docs": task_docs, "language": language, "size": size}
    task = Task.create(task_id=task_id, task_name=task_name, args=args)
    await task_manager.save_task(task)
    await task_manager.enqueue(task)
    return task.id


async def _poll_tasks(
    tasks: List[str], task_manager: TaskManager, interval_s: float = 2.0
) -> str:
    while "I'm waiting for a task be done":
        for t in tasks:
            task = await task_manager.get_task(t)
            if task.state in ERROR_STATES:
                msg = f'Task(id="{t}") has state {task.state}, exiting...'
                raise ValueError(msg)
            if task.state is TaskState.DONE:
                tasks.remove(t)
                return t
        await asyncio.sleep(interval_s)


async def _add_doc_ne(
    es_client: ESClient, named_entities: List[DSNamedEntity], project: str
):
    named_entities = jsonable_encoder(named_entities)
    actions = (
        {
            "_op_type": "update",
            "_index": project,
            ID_: ne["id"],
            "doc": ne,
            "doc_as_upsert": True,
        }
        for ne in named_entities
    )
    success, _ = await async_bulk(es_client, actions, raise_on_error=True)


async def _language_docs_it(
    docs: AsyncIterator[Dict],
) -> AsyncGenerator[Tuple[str, AsyncGenerator[Dict, None]], None]:
    try:
        cur_doc = await anext(docs)
    except StopIteration:
        return
    current_language = cur_doc[SOURCE][DOC_LANGUAGE]
    while "I can poll docs":
        iter_while = lambda d: cur_doc[SOURCE][DOC_LANGUAGE] == current_language
        lang_docs, docs = before_and_after(docs, iter_while)
        yield current_language, lang_docs
        try:
            cur_doc = await anext(docs)
        except StopIteration:
            return
        current_language = cur_doc[SOURCE][DOC_LANGUAGE]


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


async def main(
    *, batch_size: int, max_content_length: int, nlp_parallelism: int, size: SpacySize
):
    deps = [
        ("loading async app configuration", load_app_config, None),
        ("es client", es_client_enter, es_client_exit),
    ]
    task_storage = PostgresStorageConfig(port=5435).to_storage(None)
    async with run_deps(deps, ""):
        async with task_storage:
            management_client = AMQPManagementClient(
                "http://localhost:15672",
                rabbitmq_vhost="%2F",
                rabbitmq_auth=BasicAuth("guest", "guest"),
            )
            broker_url = "http://localhost:5672"
            tm = AMQPTaskManager(
                app, task_storage, management_client, broker_url=broker_url
            )
            es_client = lifespan_es_client()
            async with tm:
                await extract_nlp_task(
                    "local-datashare",
                    task_manager=tm,
                    es_client=es_client,
                    batch_size=batch_size,
                    max_content_length=max_content_length,
                    nlp_parallelism=nlp_parallelism,
                    size=size,
                )


if __name__ == "__main__":
    import spacy_worker
    import icij_worker
    from icij_common.logging_utils import setup_loggers

    names = [__name__, spacy_worker.__name__]
    setup_loggers(names, level=logging.DEBUG)
    setup_loggers([icij_worker.__name__], level=logging.INFO)
    nlp_parallelism = 10
    # batch_size = 4 * 1024
    batch_size = 1 * 1024
    max_content_length = 1024
    spacy_size = SpacySize.MEDIUM
    asyncio.run(
        main(
            batch_size=batch_size,
            max_content_length=max_content_length,
            nlp_parallelism=nlp_parallelism,
            size=spacy_size,
        )
    )
    # asyncio.run(main_ds())
