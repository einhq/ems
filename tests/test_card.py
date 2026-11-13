"""The card goes to batch work while there is some, and to search after `quiet_after` without any."""

import asyncio
import time

import pytest

from conftest import FakeModelWorker
from ems.card import CardBudget, room_mb
from ems.errors import MODEL_UNAVAILABLE, EmsError
from ems.worker import Worker
from test_query_worker import Up


class Switchable(Worker):
    """A card worker the test starts and stops without a process. `starts` is what each start
    pinned, `loads` what it was asked to load, at a start or by `pin`."""

    def __init__(self, *args, order: list[str], **kwargs):
        super().__init__(*args, **kwargs)
        self.up = False
        self.order = order
        self.starts: list[str] = []
        self.loads: list[str] = []
        self.holds: list[str] = []

    @property
    def running(self) -> bool:
        return self.up

    @property
    def pid(self) -> int | None:
        return 4242 if self.up else None

    async def start(self) -> None:
        self.up = True
        pinned = self.env.get("CFG_MODEL_PINNED", "")
        self.starts.append(pinned)
        self.holds = [family for family in pinned.split(",") if family]
        if pinned:
            self.loads.append(pinned)

    async def stop(self) -> None:
        self.up = False
        self.holds = []
        self.order.append("stop card")

    def pin(self, families: list[str]) -> dict:
        self.holds = [*self.holds, *families]
        self.loads.append(",".join(families))
        return {"pinning": families}


def card_stats(free_mb: float | None):
    async def stats():
        return None if free_mb is None else {"mem_total_mb": 16311.0, "mem_used_mb": 16311.0 - free_mb}
    return stats


@pytest.fixture
async def setup(client, fake_worker, tmp_path, dirs):
    """The client's EMS with a query worker and a card worker; `arm(free_mb)` turns the arbiter on."""
    query_fake = FakeModelWorker(str(tmp_path / "query.sock"))
    card_fake = FakeModelWorker(str(tmp_path / "card.sock"))
    await query_fake.start()
    await card_fake.start()
    ctx = client.app.state.ctx
    order: list[str] = []
    ctx.query_worker = Up("/nonexistent/model_server.py", query_fake.socket_path, log=lambda m: None)
    card = Switchable("/nonexistent/model_server.py", card_fake.socket_path, log=lambda m: None, order=order)

    card_fake.on("health", lambda req: {"loaded": [f"{family}:cuda" for family in card.holds]})
    card_fake.on("pin", lambda req: card.pin(req["families"]))
    for fake, vector in ((query_fake, [1.0]), (card_fake, [9.0])):
        fake.on("embed_texts", lambda req, v=vector: {"embeddings": [v], "model": "bge-m3"})
        fake.on("embed_image_texts", lambda req, v=vector: {"embeddings": [v], "model": "siglip2"})
    fake_worker.on("unload", lambda req: (order.append(f"unload {req.get('families') or 'all'}"), {"unloaded": []})[1])
    fake_worker.on("embed_images", lambda req: (order.append("embed"), {"embeddings": [[1.0]], "model": "siglip2"})[1])
    ro, _ = dirs
    (ro / "a.jpg").write_bytes(b"jpeg")
    fake_worker.on("embed_texts", lambda req: {"embeddings": [[2.0]], "model": "bge-m3"})

    arbiter = ctx.card

    async def arm(free_mb: float | None):
        arbiter.query, arbiter.card, arbiter.card_stats = ctx.query_worker, card, card_stats(free_mb)
        arbiter.budget = CardBudget()
        await arbiter.start()
        arbiter.last_batch = time.time() - arbiter.quiet_after - 1
        return arbiter

    yield arm, order, card
    await arbiter.stop()
    await query_fake.stop()
    await card_fake.stop()


async def search(client):
    text = (await client.post("/v0/text/embed", json={"text": "segelbåt", "role": "query"})).json()["result"]["embedding"]
    picture = (await client.post("/v0/vision/embed_text", json={"text": "segelbåt"})).json()["result"]["embedding"]
    return text, picture


async def test_quiet_on_a_big_card_puts_both_query_families_on_it(client, setup):
    arm, order, card = setup
    arbiter = await arm(10000)
    assert await search(client) == ([1.0], [1.0])
    # While busy the card worker waits with nothing loaded.
    assert card.running and card.starts == [""] and card.loads == []
    await arbiter.tick()
    assert arbiter.mode == "quiet" and order == ["unload all"]
    assert card.starts == [""] and card.loads == ["text-query,siglip-text"]
    assert await search(client) == ([9.0], [9.0])
    health = (await client.get("/v0/health")).json()
    assert health["card"]["mode"] == "quiet" and health["card"]["search_on_card"] == ["siglip-text", "text-query"]


async def test_a_small_card_takes_what_fits(client, setup):
    arm, _, card = setup
    # 8 GB beside a 5.6 GB chat model: room for text-query (1800 + 400 + 1024 MB), not siglip-text.
    arbiter = await arm(3500)
    await arbiter.tick()
    assert card.loads == ["text-query"]
    assert await search(client) == ([9.0], [1.0])
    assert "siglip-text" in arbiter.skipped


async def test_a_card_without_room_is_left_alone(client, setup):
    arm, _, card = setup
    arbiter = await arm(2000)
    await arbiter.tick()
    assert arbiter.mode == "quiet" and card.loads == []
    assert await search(client) == ([1.0], [1.0])


async def test_without_a_card_nothing_is_arbitrated(client, setup, fake_worker):
    arm, order, card = setup
    arbiter = await arm(None)
    await arbiter.tick()
    assert arbiter.enabled is False and order == [] and card.starts == []
    await client.post("/v0/text/embed", json={"texts": ["en segelbåt"]})
    assert "device" not in fake_worker.requests[-1]
    assert (await client.get("/v0/health")).json()["card"] == {"mode": "off", "share_mb": None}


async def test_documents_go_to_the_card_while_quiet(client, setup, fake_worker):
    arm, _, _ = setup
    arbiter = await arm(10000)
    await client.post("/v0/text/embed", json={"texts": ["en segelbåt"]})
    await arbiter.tick()
    await client.post("/v0/text/embed", json={"texts": ["en segelbåt"]})
    sent = [request for request in fake_worker.requests if request["op"] == "embed_texts"]
    assert "device" not in sent[0] and sent[1]["device"] == "card"
    # Documents are not batch work: they leave the card quiet.
    assert arbiter.mode == "quiet"


async def test_batch_work_stops_the_card_worker_before_it_runs(client, setup, dirs):
    arm, order, card = setup
    ro, _ = dirs
    arbiter = await arm(10000)
    await arbiter.tick()
    order.clear()
    response = await client.post("/v0/vision/embed", json={"file": {"path": str(ro / "a.jpg"), "mimetype": "image/jpeg"}})
    assert response.status_code == 200, response.text
    assert order == ["stop card", "unload ['text-embed']", "embed"]
    # A new one waits for the next quiet, holding nothing.
    assert arbiter.mode == "busy" and card.running and card.starts[-1] == "" and card.holds == []
    assert await search(client) == ([1.0], [1.0])
    # It stays busy while batch work keeps coming.
    await arbiter.tick()
    assert arbiter.mode == "busy"


async def test_batch_work_arriving_while_going_quiet_keeps_the_card(client, setup, fake_worker, dirs):
    arm, order, card = setup
    ro, _ = dirs
    arbiter = await arm(10000)

    async def slow_unload(req):
        order.append("unload all")
        await asyncio.sleep(0.2)
        return {"unloaded": []}

    fake_worker.on("unload", slow_unload)
    going_quiet = asyncio.create_task(arbiter.tick())
    await asyncio.sleep(0.05)
    response = await client.post("/v0/vision/embed", json={"file": {"path": str(ro / "a.jpg"), "mimetype": "image/jpeg"}})
    await going_quiet
    assert response.status_code == 200, response.text
    # The card worker loaded nothing, and the batch work ran after the unload.
    assert order == ["unload all", "embed"] and card.loads == []
    assert arbiter.mode == "busy"


async def test_a_card_worker_that_restarted_is_used_again_once_it_has_loaded(client, setup):
    arm, _, card = setup
    arbiter = await arm(10000)
    await arbiter.tick()
    card.restarts += 1
    assert await search(client) == ([1.0], [1.0])
    await arbiter.tick()
    assert await search(client) == ([9.0], [9.0])


async def test_a_card_worker_that_is_down_at_the_switch_starts_with_the_families(client, setup):
    arm, _, card = setup
    arbiter = await arm(10000)
    card.up = False
    await arbiter.tick()
    assert card.starts == ["", "text-query,siglip-text"]
    assert await search(client) == ([9.0], [9.0])


async def test_a_card_worker_that_does_not_take_the_families_is_replaced_by_one_that_starts_with_them(client, setup):
    arm, _, card = setup
    arbiter = await arm(10000)

    async def refuse(op, payload, **kwargs):
        if op == "pin":
            raise EmsError(MODEL_UNAVAILABLE, "model worker is not answering")
        return await Worker.call(card, op, payload, **kwargs)

    card.call = refuse
    await arbiter.tick()
    assert card.order[-1] == "stop card" and card.starts == ["", "text-query,siglip-text"]
    assert await search(client) == ([9.0], [9.0])


async def test_health_says_where_each_family_runs_in_each_mode(client, setup, fake_worker):
    arm, _, _ = setup
    fake_worker.on("health", lambda req: {"loaded": [], "families": {
        "whisper": {"device": "auto", "pinned": False, "copies": []},
        "text-embed": {"device": "cpu", "pinned": False, "copies": []},
        "text-query": {"device": "cpu", "pinned": True, "copies": []},
        "siglip-text": {"device": "cpu", "pinned": True, "copies": []},
    }})
    arbiter = await arm(3500)

    from ems.ops import system
    system._smi_cache.update(at=time.monotonic(), stats={"mem_total_mb": 16311.0, "mem_used_mb": 16311.0 - 3500})
    # Busy: a prediction from the card now.
    plan = (await client.get("/v0/health")).json()["card"]["plan"]
    assert plan["whisper"] == {"busy": "card", "quiet": "unloaded"}
    assert plan["text-embed"] == {"busy": "cpu", "quiet": "card"}
    assert plan["text-query"] == {"busy": "cpu", "quiet": "card"}
    assert plan["siglip-text"]["busy"] == "cpu" and plan["siglip-text"]["quiet"] == "cpu" and "needs" in plan["siglip-text"]["why"]
    # Quiet: what happened.
    await arbiter.tick()
    plan = (await client.get("/v0/health")).json()["card"]["plan"]
    assert plan["text-query"]["quiet"] == "card" and plan["siglip-text"]["quiet"] == "cpu"


async def test_a_share_set_while_busy_reaches_the_workers_and_health(client, setup, fake_worker):
    arm, _, card = setup
    shares = []
    fake_worker.on("card_share", lambda req: (shares.append(req["mb"]), {"unloaded": ["whisper:cuda:float16"]})[1])
    arbiter = await arm(10000)
    response = await client.put("/v0/card", json={"share_mb": 6000})
    assert response.status_code == 200, response.text
    assert response.json()["result"] == {"share_mb": 6000, "unloaded": ["whisper:cuda:float16"]}
    # The batch worker unloads what no longer fits; each worker starts with the share from now on.
    assert shares == [6000]
    assert arbiter.batch.env["CFG_CARD_SHARE_MB"] == "6000" and card.env["CFG_CARD_SHARE_MB"] == "6000"
    assert (await client.get("/v0/health")).json()["card"]["share_mb"] == 6000
    await client.put("/v0/card", json={"share_mb": None})
    assert shares == [6000, None] and arbiter.batch.env["CFG_CARD_SHARE_MB"] == ""
    assert arbiter.mode == "busy" and card.loads == []
    for body in ({"share_mb": -1}, {}, {"share_mb": "all"}):
        assert (await client.put("/v0/card", json=body)).status_code == 400, body


async def test_a_share_set_while_quiet_places_search_on_the_card_again(client, setup, fake_worker):
    arm, order, card = setup
    fake_worker.on("card_share", lambda req: {"unloaded": []})
    arbiter = await arm(10000)
    await arbiter.tick()
    assert card.holds == ["text-query", "siglip-text"]

    async def stats():
        # The card worker holds 5.3 GB of the card, the batch worker 0.2 GB, another service 5 GB.
        return {"mem_total_mb": 16311.0, "mem_used_mb": 10500.0, "processes": [{"pid": 7, "used_mb": 200}, {"pid": card.pid, "used_mb": 5300}]}

    arbiter.card_stats = stats
    # 6 GB leaves room for text-query alone: a card worker that loads only that replaces the one.
    order.clear()
    await client.put("/v0/card", json={"share_mb": 6000})
    await arbiter.placing
    assert order == ["stop card"] and card.holds == ["text-query"] and arbiter.mode == "quiet"
    assert await search(client) == ([9.0], [1.0]) and "siglip-text" in arbiter.skipped
    # The whole card again: siglip-text joins it on the same card worker.
    await client.put("/v0/card", json={"share_mb": None})
    await arbiter.placing
    assert order == ["stop card"] and card.loads[-1] == "text-query,siglip-text"
    assert await search(client) == ([9.0], [9.0]) and arbiter.skipped == {}
    # None of it: search and documents leave the card.
    await client.put("/v0/card", json={"share_mb": 0})
    await arbiter.placing
    assert order == ["stop card", "stop card"] and card.holds == [] and arbiter.on_card == frozenset()
    assert await search(client) == ([1.0], [1.0])
    await client.post("/v0/text/embed", json={"texts": ["en segelbåt"]})
    assert "device" not in [request for request in fake_worker.requests if request["op"] == "embed_texts"][-1]


async def test_the_share_is_answered_before_search_is_placed_again(client, setup, fake_worker):
    arm, _, card = setup
    fake_worker.on("card_share", lambda req: {"unloaded": []})
    arbiter = await arm(10000)
    await arbiter.tick()
    loaded = asyncio.Event()
    card_health = card.health

    async def slow_health(wait=None):
        await loaded.wait()
        return await card_health(wait)

    card.health = slow_health
    card.holds = []
    arbiter.on_card = frozenset()
    response = await asyncio.wait_for(client.put("/v0/card", json={"share_mb": 12000}), 5)
    assert response.status_code == 200 and not arbiter.placing.done()
    loaded.set()
    await arbiter.placing
    assert arbiter.on_card == {"text-query", "siglip-text"} and arbiter.mode == "quiet"


async def test_the_plan_keeps_a_family_the_share_cannot_hold_on_the_cpu(client, setup, fake_worker):
    arm, _, _ = setup
    fake_worker.on("card_share", lambda req: {"unloaded": []})
    fake_worker.on("health", lambda req: {"loaded": [], "families": {
        "whisper": {"device": "auto", "card_need_mb": 4000, "pinned": False, "copies": []},
        "faces": {"device": "auto", "card_need_mb": 2500, "pinned": False, "copies": []},
        "text-embed": {"device": "cpu", "card_need_mb": 1500, "pinned": False, "copies": []},
        "text-query": {"device": "cpu", "pinned": True, "copies": []},
        "siglip-text": {"device": "cpu", "pinned": True, "copies": []},
    }})
    await arm(10000)

    async def plan(share_mb):
        await client.put("/v0/card", json={"share_mb": share_mb})
        return (await client.get("/v0/health")).json()["card"]["plan"]

    assert (await plan(4000))["whisper"]["busy"] == "cpu" and (await plan(4000))["faces"]["busy"] == "card"
    assert (await plan(4400))["whisper"]["busy"] == "card"
    nothing = await plan(0)
    assert nothing["whisper"]["busy"] == nothing["faces"]["busy"] == "cpu" and nothing["text-embed"] == {"busy": "cpu", "quiet": "cpu"}


def test_room_keeps_to_the_share_whatever_the_card_has_free():
    # 16 GB, 6 GB in use, all of it EMS's own: the card has 10 GB free, the share 10.2 GB.
    stats = {"mem_total_mb": 16000.0, "mem_used_mb": 6000.0, "processes": [{"pid": 57, "used_mb": 6000}]}
    assert room_mb(stats) == 10000
    assert room_mb(stats, share_mb=10200) == 4200
    # What is about to be given back counts as room on both bounds.
    assert room_mb(stats, share_mb=10200, held_mb=6000) == 10200
    # The other tenant back: the card's free memory is the tighter bound.
    stats = {**stats, "mem_used_mb": 12600.0}
    assert room_mb(stats, share_mb=10200) == 3400
    assert room_mb(stats, share_mb=0, held_mb=6000) == 0
    assert room_mb(None) == 0
