"""Agent loop tests with a scripted fake Claude (no network)."""

from sift import store, topics
from sift.agent import Agent
from tests.test_topics import corpus


class ScriptedClaude:
    def __init__(self, turns):
        self.turns = list(turns)
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = []

    def messages(self, system, messages, tools=None, max_tokens=None):
        self.calls.append(messages[-1])
        self.input_tokens += 100
        self.output_tokens += 10
        return self.turns.pop(0)


def db():
    con = store.connect(":memory:")
    recs = corpus()
    for r in recs:
        r.metadata.update({"game": "Helldivers 2", "sample_window": "P"})
    store.upsert_records(con, recs)
    con.execute("UPDATE records SET metadata = json_merge_patch(metadata, '{\"appid\": 553850}')")
    topics.build_topics(con, 553850, n_topics=3)
    return con


def test_citations_are_verified_against_what_tools_returned():
    con = db()
    tool_turn = {"content": [{"type": "tool_use", "id": "t1", "name": "search_reviews",
                              "input": {"query": "account psn refund", "game": "helldivers 2", "limit": 3}}]}
    client = ScriptedClaude([tool_turn, {"content": [{"type": "text", "text": "placeholder"}]}])
    agent = Agent(con, client)
    shown = agent.index.search("account psn refund", game="helldivers 2", limit=3)
    real = shown[0][0]
    client.turns[-1] = {"content": [{"type": "text",
                         "text": f"Players objected to account linking [{real}] [stats]. Also [steam:553850:999999, {real}]."}]}
    ans = agent.ask("Why did players get angry?")
    assert ans.verified == [real]
    assert ans.unverified == ["steam:553850:999999"]
    assert ans.tool_calls[0]["tool"] == "search_reviews"
    # tool results reach the model wrapped as untrusted text
    sent = client.calls[-1]["content"][0]["content"]
    assert "<untrusted_review>" in sent


def test_round_cap_forces_an_answer():
    con = db()
    loop = {"content": [{"type": "tool_use", "id": "x", "name": "topic_trends", "input": {"game": "helldivers 2"}}]}
    client = ScriptedClaude([loop] * 3 + [{"content": [{"type": "text", "text": "done"}]}])
    ans = Agent(con, client, max_rounds=3).ask("q")
    assert ans.text == "done" and len(ans.tool_calls) == 3


def test_compare_unknown_patch_lists_available():
    con = db()
    out = Agent(con, ScriptedClaude([]))._compare("helldivers 2", "Nope")
    assert "error" in out
