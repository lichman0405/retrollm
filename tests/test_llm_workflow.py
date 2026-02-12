from __future__ import annotations

from retrollm.llm.controller import LLMMetaController
from retrollm.llm.provider_loader import LLMSettings


class _FakeProvider:
    def __init__(self, responses: list[object]):
        self._responses = responses
        self._idx = 0

    def chat(self, **kwargs) -> str:  # type: ignore[no-untyped-def]
        del kwargs
        if self._idx >= len(self._responses):
            return "{}"
        response = self._responses[self._idx]
        self._idx += 1
        if isinstance(response, Exception):
            raise response
        return str(response)


def _make_controller(responses: list[object]) -> LLMMetaController:
    controller = object.__new__(LLMMetaController)
    controller.settings = LLMSettings(
        provider="fake",
        base_url="",
        api_key="",
        model="fake-model",
        temperature=0.2,
        timeout_s=60,
    )
    controller.provider = _FakeProvider(responses)
    controller.events = []
    return controller


def test_translate_constraints_llm_path() -> None:
    controller = _make_controller(
        [
            '{"avoid_reactants":["Pd","Cr"],"avoid_template_indices":[1,2],'
            '"max_steps":4,"prefer_in_stock_subgoals":true,"notes":"test"}'
        ]
    )
    out = controller.translate_constraints("avoid Pd and Cr, <=4 steps")
    assert out["source"] == "llm"
    assert out["max_steps"] == 4
    assert out["avoid_template_indices"] == [1, 2]


def test_rerank_routes_llm_path() -> None:
    controller = _make_controller(
        [
            '{"ranking":[{"route_index":2,"score":0.91,"reason":"better"},'
            '{"route_index":1,"score":0.35,"reason":"worse"}],'
            '"global_reason":"route 2 is safer"}'
        ]
    )
    routes = [
        {"score": 0.5, "depth": 4, "solved": False, "steps": [], "molecules": []},
        {"score": 0.4, "depth": 3, "solved": True, "steps": [], "molecules": []},
    ]
    ranked, meta = controller.rerank_routes(routes, objective={})
    assert meta["mode"] == "llm"
    assert ranked[0]["llm_rank"] == 1
    assert ranked[0]["llm_rank_score"] == 0.91


def test_diagnose_failure_fallback() -> None:
    controller = _make_controller([RuntimeError("network down")])
    out = controller.diagnose_failure(
        diagnostics={"root_children": 0},
        constraints={},
        config_snapshot={"topk_templates": 10, "max_depth": 4, "use_filter": True},
    )
    assert out["source"] == "heuristic_fallback"
    assert "retry_plan" in out


def test_generate_handoff_fallback() -> None:
    controller = _make_controller([RuntimeError("timeout")])
    text = controller.generate_handoff(
        target_smiles="CCO",
        routes=[{"solved": False, "score": 0.2, "depth": 2, "steps": []}],
        constraints={},
        diagnosis=None,
    )
    assert "RetroLLM Handoff" in text


def test_generate_handoff_with_no_route_reason() -> None:
    controller = _make_controller([])
    text = controller.generate_handoff(
        target_smiles="O=C(O)c1ccc(C(=O)O)cc1",
        routes=[],
        constraints={},
        diagnosis=None,
        no_route_reason="target_in_stock",
    )
    assert "No candidate routes available." in text
    assert "Reason: Target molecule is already in stock." in text
