from typing import AsyncGenerator, Sequence, TypeVar

T = TypeVar("T")


async def iter_async(seq: Sequence[T]) -> AsyncGenerator[T, None]:
    for t in seq:
        yield t
