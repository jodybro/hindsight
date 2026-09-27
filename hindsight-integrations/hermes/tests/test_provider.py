"""The provider driven through the Hermes MemoryProvider interface, asserting what it
sends to Hindsight (a recording fake client stands in for the real SDK)."""

import json
import types

import hindsight_hermes as plugin
from conftest import FakeClient


def _retain_item(fake: FakeClient, index: int = 0) -> dict:
    return fake.retains[index]["items"][0]


def _turns_of(fake: FakeClient, index: int = 0) -> list[list[str]]:
    """Message texts per turn in one retain. Content is ``"[" + ",".join(turns) + "]"``
    where each turn is itself a JSON array, so the whole payload is a list of turns."""
    return [[m["content"] for m in turn] for turn in json.loads(_retain_item(fake, index)["content"])]


def test_sync_turn_retains_the_turn(provider):
    instance, fake = provider({"bank_id": "team", "retain_tags": "hermes"})
    instance.sync_turn("what is my name?", "Ada.")
    instance.shutdown()

    assert len(fake.retains) == 1
    call = fake.retains[0]
    assert call["bank_id"] == "team"
    assert call["document_id"] == "session-1"  # stable id + append on a capable API
    item = _retain_item(fake)
    assert item["update_mode"] == "append"
    assert "hermes" in item["tags"] and "session:session-1" in item["tags"]
    messages = json.loads(item["content"][1:-1])
    assert [m["content"] for m in messages] == ["User: what is my name?", "Assistant: Ada."]


def test_retain_every_n_turns_buffers_then_ships_the_batch(provider):
    instance, fake = provider({"retain_every_n_turns": 2})
    instance.sync_turn("one", "1")
    assert fake.retains == []
    instance.sync_turn("two", "2")
    instance.shutdown()

    assert len(fake.retains) == 1
    assert _retain_item(fake)["metadata"]["message_count"] == "4"


def test_auto_retain_off_stores_nothing(provider):
    instance, fake = provider({"auto_retain": False})
    instance.sync_turn("hello", "hi")
    instance.shutdown()
    assert fake.retains == []


def test_recall_tool_queries_the_bank_and_formats_results(provider):
    instance, fake = provider(
        {"bank_id": "team", "recall_budget": "high"}, client=FakeClient(recall_texts=["fact one", "fact two"])
    )
    result = json.loads(instance.handle_tool_call("hindsight_recall", {"query": "who am I?"}))

    assert fake.recalls[0]["bank_id"] == "team"
    assert fake.recalls[0]["budget"] == "high"
    assert fake.recalls[0]["types"] == ["observation"]  # observation-only default
    assert result["result"] == "1. fact one\n2. fact two"
    instance.shutdown()


def test_reflect_tool_uses_reflect(provider):
    instance, fake = provider({}, client=FakeClient(reflect_text="You are Ada."))
    result = json.loads(instance.handle_tool_call("hindsight_reflect", {"query": "who am I?"}))
    assert fake.reflects[0]["query"] == "who am I?"
    assert result["result"] == "You are Ada."
    instance.shutdown()


def test_retain_tool_stores_content_with_per_call_tags(provider):
    instance, fake = provider({"retain_tags": "base"})
    instance.handle_tool_call("hindsight_retain", {"content": "Ada likes tea", "tags": ["drink"]})
    item = _retain_item(fake)
    assert item["content"] == "Ada likes tea"
    assert item["tags"] == ["base", "drink"]
    instance.shutdown()


def test_tool_call_errors_are_reported_not_raised(provider):
    instance, _ = provider({})
    assert instance.handle_tool_call("hindsight_recall", {}).startswith("ERROR:")
    assert instance.handle_tool_call("nope", {"query": "x"}).startswith("ERROR:")
    instance.shutdown()


def test_prefetch_injects_recalled_memories(provider):
    instance, fake = provider({"recall_sync": True}, client=FakeClient(recall_texts=["fact one"]))
    block = instance.prefetch("what do you know?")
    assert "- fact one" in block
    status = instance.recall_status()
    assert status.count == 1 and status.provider_label == "Hindsight"
    instance.shutdown()


def test_context_mode_hides_tools_tools_mode_skips_recall(provider):
    context_only, _ = provider({"memory_mode": "context"})
    assert context_only.get_tool_schemas() == []
    context_only.shutdown()

    tools_only, fake = provider({"memory_mode": "tools", "recall_sync": True})
    assert [t["name"] for t in tools_only.get_tool_schemas()] == [
        "hindsight_retain",
        "hindsight_recall",
        "hindsight_reflect",
    ]
    assert tools_only.prefetch("anything") == ""
    assert fake.recalls == []
    tools_only.shutdown()


def test_session_switch_starts_a_new_document(provider):
    instance, fake = provider({})
    instance.sync_turn("one", "1")
    instance.on_session_switch("session-2", reset=True)
    instance.sync_turn("two", "2")
    instance.shutdown()

    # The switch flushes the old session's buffer under the old document id first,
    # so the new session's turn can never land in the previous document. In append
    # mode the buffer is already empty here (sync_turn shipped and dropped the turn),
    # so there is nothing left to flush — previously this re-shipped the retained
    # turn under session-1 a second time, duplicating it in the document.
    assert [call["document_id"] for call in fake.retains] == ["session-1", "session-2"]


def test_register_exposes_the_provider_to_hermes():
    registered = []
    plugin.register(type("Ctx", (), {"register_memory_provider": lambda _self, p: registered.append(p)})())
    assert registered and registered[0].name == "hindsight"


def test_append_mode_drops_retained_turns_from_the_buffer(provider):
    """Append retains ship a delta, so keeping every turn would pin the whole session
    in memory on a long-running gateway (hermes-agent #62950).

    Append mode comes from the API capability probe, which the fixture pins on — it is
    not a config key.
    """
    instance, fake = provider({})
    instance.sync_turn("one", "1")
    instance.sync_turn("two", "2")

    # Buffer state is read before shutdown(); retains only land once the writer drains.
    assert instance._session_turns == []
    assert instance._last_retained_turn_count == 0
    instance.shutdown()

    # Each retain still carries only its own un-retained tail, never a replay.
    assert _turns_of(fake, 0) == [["User: one", "Assistant: 1"]]
    assert _turns_of(fake, 1) == [["User: two", "Assistant: 2"]]


def test_overwrite_mode_keeps_every_turn(provider, monkeypatch):
    """Overwrite resends the full session on each retain, so its buffer must NOT be
    cleared — only the append path drops shipped turns."""
    instance, fake = provider({})
    # An API without update_mode='append' support: the fixture pins the probe on, so
    # turn it back off to exercise the overwrite path.
    monkeypatch.setattr(plugin, "_check_api_supports_update_mode_append", lambda *a, **k: False)
    instance.sync_turn("one", "1")
    instance.sync_turn("two", "2")

    assert len(instance._session_turns) == 2  # one buffered entry per turn
    instance.shutdown()

    # The second retain resends the whole session, which is what overwrite means.
    assert _turns_of(fake, 1) == [["User: one", "Assistant: 1"], ["User: two", "Assistant: 2"]]


def test_root_warning_goes_through_the_hosts_warning_callback(provider, monkeypatch):
    """The 'cannot run as root' notice is an automatic startup diagnostic: hosts that
    wire a gated sink must receive it there, not on stderr (hermes-agent cd3de040ab9)."""
    seen = []
    instance, _ = provider({}, warning_callback=seen.append, platform="telegram")
    assert instance._platform == "telegram"

    monkeypatch.setattr(plugin.os, "geteuid", lambda: 0, raising=False)
    instance._mode = "local_embedded"
    instance._start_embedded_daemon()

    assert len(seen) == 1 and "cannot run as root" in seen[0]
    assert instance._mode == "disabled"
    instance.shutdown()


def test_warning_sink_defaults_exist_without_initialize():
    """_start_embedded_daemon reads these directly, and availability probes construct a
    provider without ever calling initialize() — so __init__ must supply both."""
    bare = plugin.HindsightMemoryProvider()
    assert bare._warning_callback is None
    assert bare._platform == "cli"


def test_parent_tag_is_stamped_by_default_on_a_branch(provider):
    instance, fake = provider({})
    instance.on_session_switch("session-2", parent_session_id="session-1")
    instance.sync_turn("two", "2")
    instance.shutdown()
    assert {"session:session-2", "parent:session-1"} <= set(_retain_item(fake)["tags"])


def test_retain_session_tags_false_drops_session_and_parent_tags(provider):
    instance, fake = provider({"retain_tags": "hermes", "retain_session_tags": False})
    instance.sync_turn("one", "1")
    instance.on_session_switch("session-2", parent_session_id="session-1")
    instance.sync_turn("two", "2")
    instance.shutdown()

    assert len(fake.retains) == 2
    for index in range(2):
        assert _retain_item(fake, index)["tags"] == ["hermes"]  # configured tags survive
    # Session provenance is still recorded, just not as a scope-splitting tag.
    assert _retain_item(fake, 1)["metadata"]["session_id"] == "session-2"


def test_retain_session_tags_string_false_with_no_other_tags_sends_no_tags(provider):
    instance, fake = provider({"retain_session_tags": "false"})
    instance.sync_turn("one", "1")
    instance.shutdown()
    assert "tags" not in _retain_item(fake)


def test_retain_session_tags_is_in_the_config_schema(provider):
    instance, _ = provider({})
    schema = {entry["key"]: entry for entry in instance.get_config_schema()}
    assert schema["retain_session_tags"]["default"] is True
    instance.shutdown()


def test_recall_omits_prefer_observations_by_default(provider):
    """Default off: the arecall kwargs stay exactly what they were before the gate."""
    instance, fake = provider({"recall_sync": True}, client=FakeClient(recall_texts=["fact one"]))
    instance.prefetch("what do you know?")
    assert "prefer_observations" not in fake.recalls[0]
    instance.shutdown()


def test_prefer_observations_true_is_sent_on_sync_auto_recall(provider):
    instance, fake = provider(
        {"recall_sync": True, "prefer_observations": True}, client=FakeClient(recall_texts=["fact one"])
    )
    block = instance.prefetch("what do you know?")
    assert "- fact one" in block
    assert fake.recalls[0]["prefer_observations"] is True
    assert fake.recalls[0]["types"] == ["observation"]  # recall_types default untouched
    instance.shutdown()


def test_prefer_observations_true_is_sent_on_background_prefetch(provider):
    """The default auto-recall path (recall_sync off): queue_prefetch's worker thread."""
    instance, fake = provider(
        {"prefer_observations": True, "prefetch_waits_for_retain": False},
        client=FakeClient(recall_texts=["fact one"]),
    )
    instance.queue_prefetch("what do you know?")
    instance._prefetch_thread.join(timeout=5)
    assert fake.recalls and fake.recalls[0]["prefer_observations"] is True
    assert "- fact one" in instance.prefetch("next turn")
    instance.shutdown()


def test_prefer_observations_string_values_are_parsed(provider):
    on, fake_on = provider({"recall_sync": True, "prefer_observations": "true"})
    on.prefetch("q")
    assert fake_on.recalls[0]["prefer_observations"] is True
    on.shutdown()

    off, fake_off = provider({"recall_sync": True, "prefer_observations": "false"})
    off.prefetch("q")
    assert "prefer_observations" not in fake_off.recalls[0]
    off.shutdown()


def test_prefer_observations_also_applies_to_the_recall_tool(provider):
    instance, fake = provider({"prefer_observations": True}, client=FakeClient(recall_texts=["fact one"]))
    instance.handle_tool_call("hindsight_recall", {"query": "who am I?"})
    assert fake.recalls[0]["prefer_observations"] is True
    instance.shutdown()


def test_prefer_observations_is_in_the_config_schema(provider):
    instance, _ = provider({})
    schema = {entry["key"]: entry for entry in instance.get_config_schema()}
    assert schema["prefer_observations"]["default"] is False
    instance.shutdown()


def test_recall_omits_trace_by_default(provider):
    """Default off: the arecall kwargs stay exactly what they were before the gate."""
    instance, fake = provider({"recall_sync": True}, client=FakeClient(recall_texts=["fact one"]))
    instance.prefetch("what do you know?")
    assert "trace" not in fake.recalls[0]
    instance.shutdown()


def test_recall_trace_true_is_sent_on_sync_auto_recall(provider):
    instance, fake = provider(
        {"recall_sync": True, "recall_trace": True}, client=FakeClient(recall_texts=["fact one"])
    )
    block = instance.prefetch("what do you know?")
    assert "- fact one" in block
    assert fake.recalls[0]["trace"] is True
    instance.shutdown()


def test_recall_trace_true_is_sent_on_background_prefetch(provider):
    """The default auto-recall path (recall_sync off): queue_prefetch's worker thread."""
    instance, fake = provider(
        {"recall_trace": True, "prefetch_waits_for_retain": False},
        client=FakeClient(recall_texts=["fact one"]),
    )
    instance.queue_prefetch("what do you know?")
    instance._prefetch_thread.join(timeout=5)
    assert fake.recalls and fake.recalls[0]["trace"] is True
    assert "- fact one" in instance.prefetch("next turn")
    instance.shutdown()


def test_recall_trace_string_values_are_parsed(provider):
    on, fake_on = provider({"recall_sync": True, "recall_trace": "true"})
    on.prefetch("q")
    assert fake_on.recalls[0]["trace"] is True
    on.shutdown()

    off, fake_off = provider({"recall_sync": True, "recall_trace": "false"})
    off.prefetch("q")
    assert "trace" not in fake_off.recalls[0]
    off.shutdown()


def test_recall_trace_also_applies_to_the_recall_tool(provider):
    instance, fake = provider({"recall_trace": True}, client=FakeClient(recall_texts=["fact one"]))
    instance.handle_tool_call("hindsight_recall", {"query": "who am I?"})
    assert fake.recalls[0]["trace"] is True
    instance.shutdown()


def test_recall_trace_is_in_the_config_schema(provider):
    instance, _ = provider({})
    schema = {entry["key"]: entry for entry in instance.get_config_schema()}
    assert schema["recall_trace"]["default"] is False
    instance.shutdown()


def _typed_client() -> FakeClient:
    client = FakeClient(recall_texts=["fact one", "fact two"])
    base = client.arecall

    async def arecall(**kwargs):
        resp = await base(**kwargs)
        for r, (rid, rtype) in zip(resp.results, [("a1b2c3d4-0000-1111", "observation"), ("9f8e7d6c-2222-3333", "world")]):
            r.id, r.type = rid, rtype
        return resp

    client.arecall = arecall
    return client


def test_recall_indicator_detail_off_by_default_keeps_the_plain_status(provider):
    seen = []
    instance, _ = provider({"recall_sync": True}, client=_typed_client(), status_callback=seen.append)
    instance.prefetch("what do you know?")
    status = instance.recall_status()
    assert status is not None and status.count == 2
    assert seen == []
    instance.shutdown()


def test_recall_indicator_detail_on_sync_auto_recall_emits_type_and_short_id(provider):
    seen = []
    instance, _ = provider(
        {"recall_sync": True, "recall_indicator_detail": True}, client=_typed_client(), status_callback=seen.append
    )
    instance.prefetch("what do you know?")
    assert instance.recall_status() is None  # emitted directly, so the generic line is suppressed
    assert seen == [f"{plugin._HINDSIGHT_GLYPH} Hindsight — recalled 2 memories (obs a1b2c3d4, world 9f8e7d6c)"]
    instance.shutdown()


def test_recall_indicator_detail_on_background_prefetch(provider):
    """The default auto-recall path (recall_sync off): refs must survive the thread handoff."""
    seen = []
    instance, _ = provider(
        {"recall_indicator_detail": True, "prefetch_waits_for_retain": False},
        client=_typed_client(),
        status_callback=seen.append,
    )
    instance.queue_prefetch("what do you know?")
    instance._prefetch_thread.join(timeout=5)
    assert "- fact one" in instance.prefetch("next turn")
    assert instance.recall_status() is None
    assert seen == [f"{plugin._HINDSIGHT_GLYPH} Hindsight — recalled 2 memories (obs a1b2c3d4, world 9f8e7d6c)"]
    instance.shutdown()


def test_recall_indicator_detail_string_values_are_parsed(provider):
    seen = []
    on, _ = provider(
        {"recall_sync": True, "recall_indicator_detail": "true"}, client=_typed_client(), status_callback=seen.append
    )
    on.prefetch("q")
    assert on.recall_status() is None and len(seen) == 1
    on.shutdown()

    seen.clear()
    off, _ = provider(
        {"recall_sync": True, "recall_indicator_detail": "false"}, client=_typed_client(), status_callback=seen.append
    )
    off.prefetch("q")
    assert off.recall_status() is not None and seen == []
    off.shutdown()


def test_recall_indicator_detail_falls_back_without_a_status_callback(provider):
    """Non-CLI platforms get no status_callback: keep the core's plain line."""
    instance, _ = provider({"recall_sync": True, "recall_indicator_detail": True}, client=_typed_client())
    instance.prefetch("q")
    status = instance.recall_status()
    assert status is not None and status.count == 2
    instance.shutdown()


def test_recall_indicator_detail_respects_recall_indicator_off(provider):
    seen = []
    instance, _ = provider(
        {"recall_sync": True, "recall_indicator_detail": True, "recall_indicator": False},
        client=_typed_client(),
        status_callback=seen.append,
    )
    instance.prefetch("q")
    assert instance.recall_status() is None and seen == []
    instance.shutdown()


def test_recall_indicator_detail_is_in_the_config_schema(provider):
    instance, _ = provider({})
    schema = {entry["key"]: entry for entry in instance.get_config_schema()}
    assert schema["recall_indicator_detail"]["default"] == "off"  # was False; true/false still parse
    instance.shutdown()


def _rewrite_config(hermes_env_path, **changes) -> None:
    """Edit the live config.json mid-process, the way a user hand-edits it."""
    path = hermes_env_path / "hindsight" / "config.json"
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data))


def test_session_switch_reloads_retain_session_tags_from_config(provider, hermes_env):
    """A config.json edit AFTER initialize() takes effect on the next /new, /resume, /branch."""
    instance, fake = provider({"retain_tags": "hermes"})  # retain_session_tags defaults to True
    instance.sync_turn("one", "1")
    _rewrite_config(hermes_env, retain_session_tags=False)
    instance.on_session_switch("session-2", parent_session_id="session-1")
    instance.sync_turn("two", "2")
    instance.shutdown()

    assert "session:session-1" in _retain_item(fake, 0)["tags"]  # before the edit: tagged
    assert _retain_item(fake, 1)["tags"] == ["hermes"]  # after the switch: lineage tags gone


def test_session_switch_reload_parses_string_values_and_can_turn_tags_back_on(provider, hermes_env):
    instance, fake = provider({"retain_session_tags": "false"})
    instance.sync_turn("one", "1")
    _rewrite_config(hermes_env, retain_session_tags="true")
    instance.on_session_switch("session-2", parent_session_id="session-1")
    instance.sync_turn("two", "2")
    instance.shutdown()

    assert "tags" not in _retain_item(fake, 0)
    assert {"session:session-2", "parent:session-1"} <= set(_retain_item(fake, 1)["tags"])


def test_session_switch_flushes_buffered_turns_under_the_old_policy(provider, hermes_env):
    """The old session's buffered turns ship as the old session was configured; only the
    new session picks up the edited policy."""
    instance, fake = provider({"retain_every_n_turns": 2})
    instance.sync_turn("one", "1")  # buffered, not yet shipped
    _rewrite_config(hermes_env, retain_session_tags=False, retain_every_n_turns=1)
    instance.on_session_switch("session-2")
    instance.sync_turn("two", "2")  # every-n now 1 -> ships immediately
    instance.shutdown()

    assert [call["document_id"] for call in fake.retains] == ["session-1", "session-2"]
    assert "session:session-1" in _retain_item(fake, 0)["tags"]
    assert "tags" not in _retain_item(fake, 1)


def test_session_switch_reloads_other_retain_policy_knobs(provider, hermes_env):
    instance, fake = provider({})
    _rewrite_config(hermes_env, auto_retain=False, retain_async=False, retain_every_n_turns=3)
    instance.on_session_switch("session-2")
    assert instance._auto_retain is False
    assert instance._retain_async is False
    assert instance._retain_every_n_turns == 3
    instance.sync_turn("two", "2")
    instance.shutdown()
    assert fake.retains == []  # auto_retain off after the reload


def test_session_switch_reloads_recall_settings(provider, hermes_env):
    """Recall knobs had the same staleness: they are pure config and reload on switch too."""
    instance, fake = provider({}, client=FakeClient(recall_texts=["fact one"]))
    instance.handle_tool_call("hindsight_recall", {"query": "q"})
    assert "trace" not in fake.recalls[0] and "prefer_observations" not in fake.recalls[0]
    _rewrite_config(hermes_env, recall_trace=True, prefer_observations="true", recall_types=["observation", "world"])
    instance.on_session_switch("session-2")
    instance.handle_tool_call("hindsight_recall", {"query": "q"})
    assert fake.recalls[1]["trace"] is True
    assert fake.recalls[1]["prefer_observations"] is True
    assert fake.recalls[1]["types"] == ["observation", "world"]
    instance.shutdown()


def test_session_switch_does_not_reload_connection_settings(provider, hermes_env):
    """Endpoint/bank/mode and the live client stay as initialize() set them: swapping them
    under an in-flight session could split its writes across banks or servers."""
    instance, fake = provider({"bank_id": "team"})
    client_before, config_before = instance._get_client(), instance._config
    _rewrite_config(hermes_env, bank_id="other", api_url="http://elsewhere:1", mode="local_external")
    instance.on_session_switch("session-2")
    assert instance._bank_id == "team"
    assert instance._mode == "cloud"
    assert instance._api_url != "http://elsewhere:1"
    assert instance._config is config_before
    assert instance._get_client() is client_before
    instance.sync_turn("two", "2")
    instance.shutdown()
    assert fake.retains[0]["bank_id"] == "team"


def test_session_switch_keeps_cached_policy_when_config_reload_fails(provider, monkeypatch):
    instance, fake = provider({"retain_session_tags": False})

    def _boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(plugin, "_load_config", _boom)
    instance.on_session_switch("session-2")  # must not raise
    assert instance._session_id == "session-2"
    assert instance._retain_session_tags is False
    instance.sync_turn("two", "2")
    instance.shutdown()
    assert _retain_item(fake)["metadata"]["session_id"] == "session-2"
    assert "tags" not in _retain_item(fake)


def test_session_switch_malformed_config_value_keeps_the_whole_cached_policy(provider, hermes_env):
    """A hand-typo (non-int retain_every_n_turns) must not leave the policy half-applied."""
    instance, fake = provider({"retain_session_tags": True, "recall_trace": False})
    _rewrite_config(hermes_env, retain_session_tags=False, recall_trace=True, retain_every_n_turns="two")
    instance.on_session_switch("session-2")
    assert instance._session_id == "session-2"
    assert instance._retain_session_tags is True  # not half-applied
    assert instance._retain_every_n_turns == 1
    assert instance._recall_trace is False
    instance.shutdown()



# --- recall_indicator_detail: off | short | debug (+ /hs-rate) -------------------


def _scored_client() -> FakeClient:
    """Like _typed_client, plus rerank scores so the debug table has a score column."""
    client = _typed_client()
    base = client.arecall

    async def arecall(**kwargs):
        resp = await base(**kwargs)
        for r, score in zip(resp.results, (0.98, 0.41)):
            r.scores = types.SimpleNamespace(reranker=score, final=score)
        return resp

    client.arecall = arecall
    return client


_DEBUG_CFG = {"recall_sync": True, "recall_indicator_detail": "debug"}


def test_recall_indicator_detail_short_is_the_true_alias(provider):
    for value in ("short", True, "true", "SHORT"):
        seen = []
        instance, _ = provider({"recall_sync": True, "recall_indicator_detail": value}, client=_typed_client(),
                               status_callback=seen.append)
        instance.prefetch("q")
        assert seen == [f"{plugin._HINDSIGHT_GLYPH} Hindsight — recalled 2 memories (obs a1b2c3d4, world 9f8e7d6c)"]
        instance.shutdown()


def test_recall_indicator_detail_off_values(provider):
    for value in ("off", False, "false", "", None):
        seen = []
        instance, _ = provider({"recall_sync": True, "recall_indicator_detail": value}, client=_typed_client(),
                               status_callback=seen.append)
        instance.prefetch("q")
        assert seen == [] and instance.recall_status().count == 2
        instance.shutdown()


def test_recall_indicator_detail_unknown_value_falls_back_to_off(provider):
    seen = []
    instance, _ = provider({"recall_sync": True, "recall_indicator_detail": "verbose"}, client=_typed_client(),
                           status_callback=seen.append)
    instance.prefetch("q")
    assert seen == [] and instance._recall_indicator_detail == "off"
    instance.shutdown()


def test_recall_indicator_detail_env_var_used_when_config_unset(provider):
    plugin_secrets = __import__("conftest").SECRETS
    plugin_secrets["HINDSIGHT_RECALL_INDICATOR_DETAIL"] = "debug"
    instance, _ = provider({"recall_sync": True}, client=_scored_client(), status_callback=[].append)
    assert instance._recall_indicator_detail == "debug"
    instance.shutdown()


def test_recall_indicator_detail_config_wins_over_env_var(provider):
    __import__("conftest").SECRETS["HINDSIGHT_RECALL_INDICATOR_DETAIL"] = "debug"
    instance, _ = provider({"recall_sync": True, "recall_indicator_detail": "short"}, client=_typed_client(),
                           status_callback=[].append)
    assert instance._recall_indicator_detail == "short"
    instance.shutdown()


def test_recall_indicator_detail_debug_emits_header_and_table_with_full_ids(provider):
    seen = []
    instance, _ = provider(_DEBUG_CFG, client=_scored_client(), status_callback=seen.append)
    instance.prefetch("what do you know?")
    assert instance.recall_status() is None
    assert len(seen) == 1
    lines = seen[0].splitlines()
    assert lines[0] == f"{plugin._HINDSIGHT_GLYPH} Hindsight — recalled 2 memories (obs a1b2c3d4, world 9f8e7d6c)"
    body = "\n".join(lines[1:])
    assert "a1b2c3d4-0000-1111" in body and "9f8e7d6c-2222-3333" in body  # full ids, not truncated
    assert "0.98" in body and "0.41" in body  # rerank score column
    assert "rate" in body.lower() and "/hs-rate" in body
    row1 = next(line for line in lines if "a1b2c3d4-0000-1111" in line)
    assert row1.lstrip().startswith("1")
    instance.shutdown()


def test_recall_indicator_detail_debug_on_background_prefetch(provider):
    seen = []
    instance, _ = provider({"recall_indicator_detail": "debug", "prefetch_waits_for_retain": False},
                           client=_scored_client(), status_callback=seen.append)
    instance.queue_prefetch("what do you know?")
    instance._prefetch_thread.join(timeout=5)
    instance.prefetch("next turn")
    assert len(seen) == 1 and "9f8e7d6c-2222-3333" in seen[0]
    assert instance._last_recall_query == "what do you know?"  # the query that actually produced the rows
    instance.shutdown()


def test_recall_indicator_detail_debug_without_scores_shows_dash(provider):
    seen = []
    instance, _ = provider(_DEBUG_CFG, client=_typed_client(), status_callback=seen.append)
    instance.prefetch("q")
    row = next(line for line in seen[0].splitlines() if "a1b2c3d4-0000-1111" in line)
    assert "-" in row.split("a1b2c3d4-0000-1111")[1]
    instance.shutdown()


def test_hs_rate_writes_jsonl_ratings(provider, hermes_env):
    instance, _ = provider(_DEBUG_CFG, client=_scored_client(), status_callback=[].append)
    instance.prefetch("what do you know?")
    out = instance._handle_rate_command("1y 2n")
    assert "2" in out
    path = hermes_env / "hindsight" / "recall-ratings.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(r["memory_id"], r["relevant"]) for r in rows] == [("a1b2c3d4-0000-1111", True), ("9f8e7d6c-2222-3333", False)]
    first = rows[0]
    assert first["query"] == "what do you know?"
    assert first["type"] == "observation" and first["rank"] == 1
    assert first["rerank_score"] == 0.98
    assert first["session_id"] == "session-1" and first["bank_id"]
    assert first["ts"].endswith("Z")
    instance.shutdown()


def test_hs_rate_accepts_separators_and_words(provider, hermes_env):
    instance, _ = provider(_DEBUG_CFG, client=_scored_client(), status_callback=[].append)
    instance.prefetch("q")
    instance._handle_rate_command("1=yes, 2:no")
    rows = [json.loads(line) for line in (hermes_env / "hindsight" / "recall-ratings.jsonl").read_text().splitlines()]
    assert [r["relevant"] for r in rows] == [True, False]
    instance.shutdown()


def test_hs_rate_rejects_bad_input_without_writing(provider, hermes_env):
    instance, _ = provider(_DEBUG_CFG, client=_scored_client(), status_callback=[].append)
    instance.prefetch("q")
    path = hermes_env / "hindsight" / "recall-ratings.jsonl"
    for bad in ("", "3y", "1maybe", "1y 9n", "banana"):
        out = instance._handle_rate_command(bad)
        assert out and not path.exists(), bad
    instance.shutdown()


def test_hs_rate_without_a_recall_says_so(provider, hermes_env):
    instance, _ = provider(_DEBUG_CFG, client=_scored_client(), status_callback=[].append)
    out = instance._handle_rate_command("1y")
    assert "no recall" in out.lower()
    assert not (hermes_env / "hindsight" / "recall-ratings.jsonl").exists()
    instance.shutdown()


def test_hs_rate_is_registered_as_a_slash_command():
    registered = {}

    class Ctx:
        def register_memory_provider(self, provider):
            registered["provider"] = provider

        def register_command(self, name, handler, description="", args_hint="", **_):
            registered[name] = (handler, args_hint)

    plugin.register(Ctx())
    assert "hs-rate" in registered
    handler, hint = registered["hs-rate"]
    assert hint and callable(handler)


def test_recall_indicator_detail_schema_lists_the_modes(provider):
    instance, _ = provider({})
    schema = {entry["key"]: entry for entry in instance.get_config_schema()}
    entry = schema["recall_indicator_detail"]
    assert entry["default"] == "off"
    assert entry["choices"] == ["off", "short", "debug"]
    assert entry.get("env_var") == "HINDSIGHT_RECALL_INDICATOR_DETAIL"
    instance.shutdown()


# -- retain_flush_on_shutdown: ship the buffered tail when the session ends ---------


def test_shutdown_drops_buffered_turns_by_default(provider):
    """Pins upstream behavior: with the flag off, a tail under retain_every_n_turns is lost."""
    instance, fake = provider({"retain_every_n_turns": 10})
    for i in range(3):
        instance.sync_turn(f"q{i}", f"a{i}")
    instance.shutdown()
    assert fake.retains == []


def test_shutdown_flushes_buffered_turns_when_enabled(provider):
    instance, fake = provider({"retain_every_n_turns": 10, "retain_flush_on_shutdown": True})
    for i in range(3):
        instance.sync_turn(f"q{i}", f"a{i}")
    assert fake.retains == []  # still buffered before shutdown
    instance.shutdown()

    assert len(fake.retains) == 1
    assert fake.retains[0]["document_id"] == "session-1"
    assert _retain_item(fake)["update_mode"] == "append"
    assert _turns_of(fake) == [["User: q0", "Assistant: a0"], ["User: q1", "Assistant: a1"], ["User: q2", "Assistant: a2"]]


def test_shutdown_flush_ships_only_the_unshipped_tail(provider):
    instance, fake = provider({"retain_every_n_turns": 2, "retain_flush_on_shutdown": True})
    for i in range(3):
        instance.sync_turn(f"q{i}", f"a{i}")
    instance.shutdown()

    assert len(fake.retains) == 2
    assert _turns_of(fake, 1) == [["User: q2", "Assistant: a2"]]


def test_shutdown_flush_is_a_noop_when_nothing_is_buffered(provider):
    instance, fake = provider({"retain_every_n_turns": 2, "retain_flush_on_shutdown": True})
    instance.sync_turn("q0", "a0")
    instance.sync_turn("q1", "a1")
    instance.shutdown()
    assert len(fake.retains) == 1


def test_shutdown_flush_runs_once_across_repeated_shutdowns(provider):
    """MemoryManager.shutdown_all() and the atexit drain can both reach shutdown()."""
    instance, fake = provider({"retain_every_n_turns": 10, "retain_flush_on_shutdown": True})
    instance.sync_turn("q0", "a0")
    instance.shutdown()
    instance.shutdown()
    instance._atexit_shutdown()
    assert len(fake.retains) == 1


def test_shutdown_flush_respects_auto_retain_off(provider):
    instance, fake = provider({"auto_retain": False, "retain_flush_on_shutdown": True})
    instance.sync_turn("q0", "a0")
    instance.shutdown()
    assert fake.retains == []


def test_shutdown_flush_string_values_are_parsed(provider):
    instance, fake = provider({"retain_every_n_turns": 10, "retain_flush_on_shutdown": "true"})
    instance.sync_turn("q0", "a0")
    instance.shutdown()
    assert len(fake.retains) == 1

    instance, fake = provider({"retain_every_n_turns": 10, "retain_flush_on_shutdown": "false"})
    instance.sync_turn("q0", "a0")
    instance.shutdown()
    assert fake.retains == []


def test_shutdown_flush_after_session_switch_targets_the_new_session(provider):
    instance, fake = provider({"retain_every_n_turns": 10, "retain_flush_on_shutdown": True})
    instance.sync_turn("old", "o")
    instance.on_session_switch("session-2")  # flushes the old tail itself
    instance.sync_turn("new", "n")
    instance.shutdown()
    assert [call["document_id"] for call in fake.retains] == ["session-1", "session-2"]
    assert _turns_of(fake, 1) == [["User: new", "Assistant: n"]]


def test_retain_flush_on_shutdown_is_in_the_config_schema(provider):
    instance, _ = provider({})
    schema = {entry["key"]: entry for entry in instance.get_config_schema()}
    assert schema["retain_flush_on_shutdown"]["default"] is False
    instance.shutdown()


def test_shutdown_without_initialize_is_safe(hermes_env):
    """Hosts construct providers for availability probes and may shut them down unused."""
    plugin.HindsightMemoryProvider().shutdown()
