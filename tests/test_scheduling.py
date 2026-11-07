"""Who a freed slot goes to: interactive first, then the caller that has used the least, then a
family already running; and the queue's depth per caller."""

import asyncio

import pytest

from conftest import AUTH
from ems.admission import Admission, Share, current_caller, current_op
from ems.app import caller_of
from ems.config import Caller, load_keys
from ems.errors import EmsError


class Held:
    """A request holding a slot until `release`, from `caller` for `op`."""

    def __init__(self, admission: Admission, op: str, caller: str, *, lane: str = "models", interactive: bool = False, order: list | None = None):
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.task = asyncio.create_task(self._run(admission, op, caller, lane, interactive, order))

    async def _run(self, admission, op, caller, lane, interactive, order):
        current_op.set(op)
        current_caller.set(caller)
        async with admission.slot(lane, interactive=interactive):
            if order is not None:
                order.append((caller, op))
            self.started.set()
            await self.release.wait()


async def settle():
    await asyncio.sleep(0.01)


async def finish(*held: Held):
    for one in held:
        one.release.set()
    await asyncio.gather(*(one.task for one in held))


async def test_the_caller_that_has_used_less_goes_first():
    admission = Admission(limit_models=1, limit_tools=1, queue_depth=8, interactive_reserve=0)
    order: list = []
    first = Held(admission, "faces.detect", "a")
    await first.started.wait()
    a_more = [Held(admission, "faces.detect", "a", order=order) for _ in range(2)]
    await settle()
    b = Held(admission, "faces.detect", "b", order=order)
    await settle()
    # a holds the slot now and has waiters older than b's; b has used nothing.
    first.release.set()
    await b.started.wait()
    assert order == [("b", "faces.detect")]
    await finish(first, b, *a_more)


async def test_a_weight_is_a_share():
    admission = Admission(limit_models=1, limit_tools=1, queue_depth=8, interactive_reserve=0, weights={"a": 3, "b": 1})
    admission.callers["a"].used = admission.callers["b"].used = 30.0
    order: list = []
    first = Held(admission, "faces.detect", "c")
    await first.started.wait()
    b = Held(admission, "faces.detect", "b", order=order)
    await settle()
    a = Held(admission, "faces.detect", "a", order=order)
    await settle()
    first.release.set()
    await a.started.wait()
    # The same seconds weigh a third for a: it goes before b, which waited longer.
    assert order == [("a", "faces.detect")]
    await finish(first, a, b)


async def test_idleness_saves_up_no_credit():
    admission = Admission(limit_models=1, limit_tools=1, queue_depth=8, interactive_reserve=0)
    admission.callers["a"] = Share(weight=1, used=500.0)
    first = Held(admission, "faces.detect", "a")
    await first.started.wait()
    b = Held(admission, "faces.detect", "b")
    await settle()
    # b arrived having used nothing: it starts level with a, not 500 s behind it.
    assert admission.callers["b"].used >= 500.0
    await finish(first, b)


async def test_a_running_family_goes_before_an_older_request_for_another():
    admission = Admission(limit_models=1, limit_tools=1, queue_depth=8, interactive_reserve=0)
    order: list = []
    first = Held(admission, "faces.detect", "a")
    await first.started.wait()
    whisper = Held(admission, "speech.transcribe", "a", order=order)
    await settle()
    faces = Held(admission, "faces.embed", "a", order=order)
    await settle()
    first.release.set()
    await faces.started.wait()
    assert order == [("a", "faces.embed")]
    await finish(first, faces, whisper)


async def test_the_oldest_goes_once_it_has_waited_long_enough():
    admission = Admission(limit_models=1, limit_tools=1, queue_depth=8, interactive_reserve=0, affinity_wait=0)
    order: list = []
    first = Held(admission, "faces.detect", "a")
    await first.started.wait()
    whisper = Held(admission, "speech.transcribe", "a", order=order)
    await settle()
    faces = Held(admission, "faces.embed", "a", order=order)
    await settle()
    first.release.set()
    await whisper.started.wait()
    assert order == [("a", "speech.transcribe")]
    await finish(first, faces, whisper)


async def test_the_lane_keeps_to_the_family_that_ran_last_once_idle():
    admission = Admission(limit_models=1, limit_tools=1, queue_depth=8, interactive_reserve=0)
    order: list = []
    first = Held(admission, "speech.transcribe", "a")
    await first.started.wait()
    faces = Held(admission, "faces.detect", "a", order=order)
    await settle()
    whisper = Held(admission, "speech.transcribe", "a", order=order)
    await settle()
    first.release.set()
    await whisper.started.wait()
    assert order == [("a", "speech.transcribe")]
    await finish(first, faces, whisper)


async def test_the_tool_lane_takes_the_oldest_of_a_caller():
    admission = Admission(limit_models=1, limit_tools=1, queue_depth=8, interactive_reserve=0)
    order: list = []
    first = Held(admission, "media.probe", "a", lane="tools")
    await first.started.wait()
    renditions = Held(admission, "image.renditions", "a", lane="tools", order=order)
    await settle()
    probe = Held(admission, "media.probe", "a", lane="tools", order=order)
    await settle()
    first.release.set()
    await renditions.started.wait()
    assert order == [("a", "image.renditions")]
    await finish(first, renditions, probe)


async def test_interactive_goes_before_batch_and_into_the_reserve():
    admission = Admission(limit_models=1, limit_tools=1, queue_depth=8, interactive_reserve=1)
    first = Held(admission, "faces.detect", "a")
    await first.started.wait()
    batch = Held(admission, "faces.detect", "b")
    await settle()
    search = Held(admission, "text.embed", "a", interactive=True)
    # The reserve's slot: it runs beside the batch request, while the other batch request waits.
    await search.started.wait()
    assert not batch.started.is_set()
    await finish(first, search, batch)


async def test_the_queue_is_full_for_one_caller_and_not_another():
    admission = Admission(limit_models=1, limit_tools=1, queue_depth=1, interactive_reserve=0)
    first = Held(admission, "faces.detect", "a")
    await first.started.wait()
    waiting = Held(admission, "faces.detect", "a")
    await settle()
    current_caller.set("a")
    with pytest.raises(EmsError) as refused:
        async with admission.slot("models"):
            pass
    assert refused.value.code.name == "busy"
    other = Held(admission, "faces.detect", "b")
    await settle()
    assert admission.caller_states()["b"]["waiting"] == 1
    await finish(first, waiting, other)


async def test_a_waiter_that_goes_away_leaves_the_slot_to_the_next():
    admission = Admission(limit_models=1, limit_tools=1, queue_depth=8, interactive_reserve=0)
    first = Held(admission, "faces.detect", "a")
    await first.started.wait()
    gone = Held(admission, "faces.detect", "b")
    await settle()
    last = Held(admission, "faces.detect", "c")
    await settle()
    gone.task.cancel()
    first.release.set()
    await last.started.wait()
    assert admission.snapshot()["models"] == {"running": 1, "waiting": 0, "limit": 1, "reserve": 0}
    await finish(first, last)
    with pytest.raises(asyncio.CancelledError):
        await gone.task
    assert admission.snapshot()["models"]["running"] == 0 and admission.requests() == []


async def test_a_waiter_granted_as_it_goes_away_gives_the_slot_on():
    admission = Admission(limit_models=1, limit_tools=1, queue_depth=8, interactive_reserve=0)
    first = Held(admission, "faces.detect", "a")
    await first.started.wait()
    gone = Held(admission, "faces.detect", "b")
    await settle()
    last = Held(admission, "faces.detect", "c")
    await settle()
    dispatch = admission._dispatch

    def granted_then_cancelled(lane):
        dispatch(lane)
        # The slot is gone's; it goes away before it has run again.
        if not gone.task.done():
            gone.task.cancel()

    admission._dispatch = granted_then_cancelled
    first.release.set()
    await last.started.wait()
    assert not gone.started.is_set() and admission.snapshot()["models"]["running"] == 1
    await finish(first, last)
    with pytest.raises(asyncio.CancelledError):
        await gone.task
    assert admission.snapshot()["models"]["running"] == 0 and admission.requests() == []


async def test_health_names_each_caller_and_who_a_request_is_from():
    admission = Admission(limit_models=1, limit_tools=1, queue_depth=8, interactive_reserve=0, weights={"core-a": 2})
    held = Held(admission, "faces.detect", "core-a")
    await held.started.wait()
    [request] = admission.requests()
    assert request["caller"] == "core-a" and request["family"] == "faces"
    state = admission.caller_states()["core-a"]
    assert (state["weight"], state["running"], state["waiting"]) == (2, 1, 0)
    await finish(held)
    assert admission.caller_states()["core-a"]["used_seconds"] >= 0


def test_a_key_names_its_caller_and_weight(tmp_path):
    keys = tmp_path / "keys"
    keys.write_text("# callers\nk1 core-home 2\nk2 core-cabin\nk3\n\n")
    assert load_keys(str(keys), "inline") == {
        "inline": Caller("default"),
        "k1": Caller("core-home", 2.0),
        "k2": Caller("core-cabin"),
        "k3": Caller("caller-4"),
    }


@pytest.mark.parametrize("weight", ["0", "-1", "lots"])
def test_a_weight_must_be_a_positive_number(tmp_path, weight):
    keys = tmp_path / "keys"
    keys.write_text(f"k1 core {weight}\n")
    with pytest.raises(ValueError, match="keys:1"):
        load_keys(str(keys), None)


def test_the_caller_is_the_name_of_the_key_the_request_carries():
    class State:
        settings = type("S", (), {"keys": {"k1": Caller("core-home")}})

    app = type("App", (), {"state": State})
    assert caller_of({"app": app, "headers": [(b"authorization", b"Bearer k1")]}) == "core-home"
    assert caller_of({"app": app, "headers": [(b"authorization", b"Bearer nope")]}) == "unknown"
    assert caller_of({"app": app, "headers": []}) == "unknown"
    assert caller_of({"headers": []}) == "anonymous"


async def test_a_request_through_the_app_is_from_its_key(client, fake_worker):
    body = (await client.get("/v0/health", headers=AUTH)).json()
    assert "callers" in body
    assert body["callers"]["default"]["weight"] == 1


async def test_another_family_waits_while_one_runs_though_slots_are_free():
    admission = Admission(limit_models=4, limit_tools=1, queue_depth=8, interactive_reserve=0)
    order: list = []
    faces = Held(admission, "faces.detect", "a")
    await faces.started.wait()
    whisper = Held(admission, "speech.transcribe", "a", order=order)
    more_faces = Held(admission, "faces.embed", "a", order=order)
    await more_faces.started.wait()
    await settle()
    # Two of four slots free, and the transcript still waits for the faces to be done.
    assert order == [("a", "faces.embed")] and admission.snapshot()["models"]["waiting"] == 1
    faces.release.set()
    await settle()
    assert not whisper.started.is_set()
    more_faces.release.set()
    await whisper.started.wait()
    await finish(faces, more_faces, whisper)


async def test_a_held_back_family_goes_once_it_has_waited_long_enough():
    admission = Admission(limit_models=4, limit_tools=1, queue_depth=8, interactive_reserve=0, affinity_wait=0.05)
    faces = Held(admission, "faces.detect", "a")
    await faces.started.wait()
    whisper = Held(admission, "speech.transcribe", "a")
    # No slot is given back meanwhile: its wait alone lets it in, beside the faces.
    await asyncio.wait_for(whisper.started.wait(), 2)
    assert admission.snapshot()["models"]["running"] == 2
    await finish(faces, whisper)


async def test_interactive_work_is_not_held_to_the_running_family():
    admission = Admission(limit_models=4, limit_tools=1, queue_depth=8, interactive_reserve=0)
    faces = Held(admission, "faces.detect", "a")
    await faces.started.wait()
    search = Held(admission, "vision.embed_text", "a", interactive=True)
    await asyncio.wait_for(search.started.wait(), 1)
    await finish(faces, search)


async def test_families_zero_runs_any_number_at_once():
    admission = Admission(limit_models=4, limit_tools=1, queue_depth=8, interactive_reserve=0, families=0)
    faces = Held(admission, "faces.detect", "a")
    whisper = Held(admission, "speech.transcribe", "a")
    await asyncio.wait_for(asyncio.gather(faces.started.wait(), whisper.started.wait()), 1)
    await finish(faces, whisper)


async def test_one_freed_slot_starts_one_family_though_more_slots_are_free():
    admission = Admission(limit_models=4, limit_tools=1, queue_depth=8, interactive_reserve=0)
    whisper = Held(admission, "speech.transcribe", "a")
    await whisper.started.wait()
    faces = Held(admission, "faces.detect", "a")
    ocr = Held(admission, "ocr.read", "a")
    await settle()
    whisper.release.set()
    await settle()
    # The dispatch that follows grants the faces; the ocr is another family, and waits for them.
    assert faces.started.is_set() and not ocr.started.is_set()
    await finish(whisper, faces, ocr)
