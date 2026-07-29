from frame_cache import FrameCache


class FakeEmbedding:
    def __init__(self, nbytes):
        self.nbytes = nbytes


def test_missing_excludes_cached_and_pending():
    cache = FrameCache(max_frames_per_task=10, max_task_mb=10, max_global_mb=10)
    try:
        cache.submit("task", [1], lambda _idx: FakeEmbedding(10))
        cache.ensure_encoded("task", 1, lambda _idx: FakeEmbedding(10), timeout=1)
        cache.submit("task", [2], lambda _idx: FakeEmbedding(20))

        assert cache.missing("task", [1, 2, 3]) == [3]
    finally:
        cache._pool.shutdown(wait=True)


def test_replacing_embedding_keeps_byte_accounting_correct():
    cache = FrameCache(max_frames_per_task=10, max_task_mb=10, max_global_mb=10)
    try:
        task = cache._get_or_create("task")
        cache._encode_and_store("task", 1, lambda _idx: FakeEmbedding(10))
        cache._encode_and_store("task", 1, lambda _idx: FakeEmbedding(25))

        assert task.bytes_used == 25
        assert cache.stats()["bytes_total"] == 25
    finally:
        cache._pool.shutdown(wait=True)
