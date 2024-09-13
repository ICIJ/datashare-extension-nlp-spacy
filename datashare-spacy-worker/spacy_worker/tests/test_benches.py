from spacy_worker.benches import before_and_after
from spacy_worker.utils import iter_async


async def test_before_and_after():
    # Given
    async_it = iter_async(range(7))
    module_3_predicate = lambda x: not (x % 3)
    not_module_3_predicate = lambda x:  x % 3

    # When
    before, async_it = before_and_after(async_it, module_3_predicate)
    before = [b async for b in before]
    expected_before = [0]
    assert before == expected_before

    before, async_it = before_and_after(async_it, not_module_3_predicate)
    before = [b async for b in before]
    expected_before = [1, 2]
    assert before == expected_before

    before, async_it = before_and_after(async_it, module_3_predicate)
    before = [b async for b in before]
    expected_before = [3]
    assert before == expected_before

    before, async_it = before_and_after(async_it, not_module_3_predicate)
    before = [b async for b in before]
    expected_before = [4, 5]
    assert before == expected_before

    after = [a async for a in async_it]
    assert after == [6]
