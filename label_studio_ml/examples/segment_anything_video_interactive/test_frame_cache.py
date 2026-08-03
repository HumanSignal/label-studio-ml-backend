import threading
import time

from frame_cache import FrameCache


class FakeEmbedding:
    def __init__(self, nbytes):
        self.nbytes = nbytes


def test_schedule_missing_excludes_cached_and_pending():
    cache = FrameCache(max_frames_per_task=10, max_task_mb=10, max_global_mb=10)
    started = threading.Event()
    release = threading.Event()

    def encode_batch(indices):
        started.set()
        assert release.wait(timeout=2)
        return {idx: FakeEmbedding(10) for idx in indices}

    try:
        task = cache._get_or_create("task")
        cache._store_embeddings(task, {1: FakeEmbedding(10)}, [1])
        cached1, pending1 = cache.schedule_missing("task", [2], encode_batch)
        assert started.wait(timeout=1)
        cached2, pending2 = cache.schedule_missing("task", [1, 2, 3], encode_batch)
    finally:
        release.set()
        cache._pool.shutdown(wait=True)

    assert cached1 == []
    assert pending1 == [2]
    assert cached2 == [1]
    assert pending2 == [2, 3]


def test_schedule_missing_reserves_before_batch_encode():
    cache = FrameCache(max_frames_per_task=10, max_task_mb=10, max_global_mb=10)
    started = threading.Event()
    release = threading.Event()
    calls = []

    def encode_batch(indices):
        calls.append(tuple(indices))
        started.set()
        assert release.wait(timeout=2)
        return {idx: FakeEmbedding(10) for idx in indices}

    try:
        cached1, pending1 = cache.schedule_missing("task", [1, 2, 3], encode_batch)
        assert started.wait(timeout=1)
        cached2, pending2 = cache.schedule_missing("task", [1, 2, 3], encode_batch)
    finally:
        release.set()
        cache._pool.shutdown(wait=True)

    assert cached1 == []
    assert pending1 == [1, 2, 3]
    assert cached2 == []
    assert pending2 == [1, 2, 3]
    assert calls == [(1, 2, 3)]


def test_replacing_embedding_keeps_byte_accounting_correct():
    cache = FrameCache(max_frames_per_task=10, max_task_mb=10, max_global_mb=10)
    try:
        task = cache._get_or_create("task")
        cache._store_embeddings(task, {1: FakeEmbedding(10)}, [1])
        cache._store_embeddings(task, {1: FakeEmbedding(25)}, [1])

        assert task.bytes_used == 25
        assert cache.stats()["bytes_total"] == 25
    finally:
        cache._pool.shutdown(wait=True)


def test_expire_idle_drops_empty_task_without_pending_work():
    cache = FrameCache(
        max_frames_per_task=10,
        max_task_mb=10,
        max_global_mb=10,
        ttl_seconds=1,
    )
    try:
        cache.touch("task", 0)
        task = cache._get_or_create("task")
        task.last_access = time.time() - 2

        cache.expire_idle()

        assert cache.stats()["tasks"] == 0
    finally:
        cache._pool.shutdown(wait=True)


def test_drop_task_does_not_store_stale_running_encode():
    cache = FrameCache(
        max_frames_per_task=10,
        max_task_mb=10,
        max_global_mb=10,
        encoder_workers=1,
    )
    started = threading.Event()
    release = threading.Event()

    def encode_batch(indices):
        started.set()
        assert release.wait(timeout=2)
        return {idx: FakeEmbedding(10) for idx in indices}

    try:
        cache.schedule_missing("task", [1], encode_batch)
        assert started.wait(timeout=1)

        cache.drop_task("task")
        cache.touch("task", 0)
    finally:
        release.set()
        cache._pool.shutdown(wait=True)

    assert not cache.has("task", 1)


def test_drop_task_cancels_queued_encodes():
    cache = FrameCache(
        max_frames_per_task=10,
        max_task_mb=10,
        max_global_mb=10,
        encoder_workers=1,
    )
    started = threading.Event()
    release = threading.Event()
    calls = []

    def encode_batch(indices):
        calls.append(tuple(indices))
        if indices == [1]:
            started.set()
            assert release.wait(timeout=2)
        return {idx: FakeEmbedding(10) for idx in indices}

    try:
        cache.schedule_missing("task", [1], encode_batch)
        assert started.wait(timeout=1)
        cache.schedule_missing("task", [2, 3], encode_batch)

        cache.drop_task("task")
    finally:
        release.set()
        cache._pool.shutdown(wait=True)

    assert calls == [(1,)]
    assert cache.stats()["tasks"] == 0
