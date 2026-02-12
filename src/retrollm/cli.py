"""Command line interface."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def _write_text(path: str | Path, text: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def _cmd_download_data(args: argparse.Namespace) -> int:
    from retrollm.data.public_data import download_public_data

    out = download_public_data(args.path)
    Console().print(f"Wrote config to: {out}")
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    from retrollm.chem.molecule import Molecule
    from retrollm.config import load_artifacts_config
    from retrollm.policy.onnx_policy import OnnxTemplatePolicy

    cfg = load_artifacts_config(args.config)
    mol = Molecule(args.smiles)
    fp = mol.morgan_fingerprint(radius=2, nbits=2048)

    policy = OnnxTemplatePolicy(cfg.policy_model_onnx)
    preds = policy.predict_topk(fp, topk=args.topk)
    Console().print(f"Top-{len(preds)} template indices (first 5 shown):")
    for pred in preds[:5]:
        Console().print(f"  idx={pred.index}  p={pred.probability:.6f}")
    return 0


def _format_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _render_search_result(
    console: Console, payload: dict[str, Any], output_path: str | None, verbose: bool
) -> None:
    summary = Table(show_header=False, box=None)
    summary.add_row("Solved", "yes" if payload.get("solved") else "no")
    summary.add_row(
        "First solution iteration", str(payload.get("first_solution_iteration") or "-")
    )
    summary.add_row("Iterations", str(payload.get("iterations")))
    summary.add_row("Search time (s)", _format_float(payload.get("search_time_s")))
    summary.add_row("Root children", str(payload.get("root_children")))
    summary.add_row("Retry attempts", str(payload.get("retry_attempts", 0)))
    if output_path:
        summary.add_row("Saved JSON", str(Path(output_path).resolve()))

    console.print(Panel(summary, title="Search Summary", expand=False))

    llm_payload = payload.get("llm", {})
    constraints = llm_payload.get("constraints", {})
    if constraints:
        ctable = Table(show_header=False, box=None)
        ctable.add_row("source", str(constraints.get("source", "unknown")))
        ctable.add_row("avoid_reactants", ", ".join(constraints.get("avoid_reactants", [])) or "-")
        ctable.add_row(
            "avoid_template_indices",
            ", ".join(str(v) for v in constraints.get("avoid_template_indices", [])) or "-",
        )
        ctable.add_row("max_steps", str(constraints.get("max_steps", "-")))
        ctable.add_row(
            "prefer_in_stock_subgoals",
            str(constraints.get("prefer_in_stock_subgoals", False)),
        )
        notes = str(constraints.get("notes", "")).strip()
        if notes:
            ctable.add_row("notes", notes)
        console.print(Panel(ctable, title="Constraint Translation", expand=False))

    rerank = llm_payload.get("rerank", {})
    if rerank:
        rtable = Table(show_header=False, box=None)
        rtable.add_row("applied", str(rerank.get("applied", False)))
        rtable.add_row("mode", str(rerank.get("mode", "-")))
        rtable.add_row("reason", str(rerank.get("global_reason", rerank.get("reason", "-"))))
        if rerank.get("error"):
            rtable.add_row("error", str(rerank.get("error")))
        console.print(Panel(rtable, title="Route Rerank", expand=False))

    diagnosis = llm_payload.get("diagnosis")
    if isinstance(diagnosis, dict) and diagnosis:
        dtable = Table(show_header=False, box=None)
        dtable.add_row("source", str(diagnosis.get("source", "-")))
        dtable.add_row("diagnosis", str(diagnosis.get("diagnosis", "-")))
        dtable.add_row("rationale", str(diagnosis.get("rationale", "-")))
        dtable.add_row("retry_plan", json.dumps(diagnosis.get("retry_plan", {}), ensure_ascii=False))
        console.print(Panel(dtable, title="Failure Diagnosis", expand=False))

    routes = payload.get("routes", [])
    if not routes:
        console.print("No route candidates found.")
    for idx, route in enumerate(routes, start=1):
        header = (
            f"score={_format_float(route.get('score'))}  "
            f"visits={route.get('visits')}  depth={route.get('depth')}  "
            f"solved={'yes' if route.get('solved') else 'no'}  "
            f"llm_rank={route.get('llm_rank', '-')}"
        )
        console.print(Panel(header, title=f"Route {idx}", expand=False))
        steps = Table()
        steps.add_column("Step", no_wrap=True)
        steps.add_column("Expanded", overflow="fold")
        steps.add_column("Reactants", overflow="fold")
        steps.add_column("Template", no_wrap=True)
        steps.add_column("Source", no_wrap=True)
        steps.add_column("p_tmpl", no_wrap=True, justify="right")
        steps.add_column("p_filt", no_wrap=True, justify="right")
        for step in route.get("steps", []):
            steps.add_row(
                str(step.get("step", "")),
                str(step.get("expanded_molecule", "")),
                " . ".join(step.get("reactants", [])),
                str(step.get("template_index", "")),
                str(step.get("policy_source", "")),
                _format_float(step.get("policy_probability")),
                _format_float(step.get("filter_score")),
            )
        console.print(steps)
        if route.get("llm_rank_reason"):
            console.print(f"LLM reason: {route.get('llm_rank_reason')}")

    handoff = llm_payload.get("handoff_markdown")
    if isinstance(handoff, str) and handoff.strip():
        preview_lines = handoff.strip().splitlines()[:10]
        console.print(Panel("\n".join(preview_lines), title="Handoff Preview", expand=False))

    if verbose:
        cfg = payload.get("config_snapshot", {})
        console.print(Panel(json.dumps(cfg, indent=2), title="Config Snapshot"))
        events = llm_payload.get("events", [])
        if isinstance(events, list) and events:
            event_table = Table("Stage", "Message", "Payload")
            for event in events[-20:]:
                event_table.add_row(
                    str(event.get("stage", "")),
                    str(event.get("message", "")),
                    json.dumps(event.get("payload", {}), ensure_ascii=False),
                )
            console.print(Panel(event_table, title="LLM Events"))


def _cmd_search(args: argparse.Namespace) -> int:
    from retrollm.finder import Planner
    from retrollm.search import SearchConfig

    config = SearchConfig(
        iteration_limit=args.iteration_limit,
        time_limit_s=args.time_limit,
        c_puct=args.c_puct,
        topk_templates=args.topk,
        max_depth=args.max_depth,
        llm_intervention_interval=args.llm_interval,
        use_filter=not args.no_filter,
        filter_cutoff=args.filter_cutoff,
        use_ringbreaker=not args.no_ringbreaker,
        ringbreaker_topk=args.ringbreaker_topk,
        max_routes=args.max_routes,
        constraints_text=args.constraints,
        llm_use_subgoal_advisor=not args.no_llm_advisor,
        llm_use_route_reranker=not args.no_llm_rerank,
        llm_use_failure_diagnosis=not args.no_llm_diagnosis,
        llm_use_handoff=not args.no_llm_handoff,
        llm_retry_on_failure=not args.no_llm_retry,
        llm_max_retry_attempts=args.llm_max_retries,
    )
    planner = Planner.from_config_file(args.config, search_config=config)
    result = planner.search(args.smiles, use_llm=args.use_llm)
    payload = asdict(result)

    if args.output:
        _write_json(args.output, payload)
    report_path: Path | None = None
    if args.report:
        handoff = payload.get("llm", {}).get("handoff_markdown", "")
        if isinstance(handoff, str) and handoff.strip():
            report_path = _write_text(args.report, handoff)

    console = Console()
    if args.json:
        console.print(json.dumps(payload, indent=2))
    else:
        _render_search_result(console, payload, output_path=args.output, verbose=args.verbose)
    if report_path is not None:
        console.print(f"Saved handoff report: {report_path.resolve()}")
    elif args.report:
        console.print("Handoff report was not generated (try enabling --use-llm).")
    return 0


def _render_doctor_report(
    console: Console, payload: dict[str, Any], output_path: str | None
) -> None:
    table = Table()
    table.add_column("Check", overflow="fold")
    table.add_column("Status", no_wrap=True)
    table.add_column("Detail", overflow="fold")
    table.add_column("Hint", overflow="fold")
    status_style = {"ok": "green", "warn": "yellow", "fail": "red"}

    for check in payload.get("checks", []):
        status = str(check.get("status", "")).lower()
        style = status_style.get(status, "white")
        table.add_row(
            str(check.get("name", "")),
            f"[{style}]{status.upper()}[/{style}]",
            str(check.get("detail", "")),
            str(check.get("hint", "")),
        )

    console.print(table)
    if output_path:
        console.print(f"Saved JSON: {Path(output_path).resolve()}")
    console.print(f"Overall: {'OK' if payload.get('ok') else 'FAIL'}")


def _cmd_doctor(args: argparse.Namespace) -> int:
    from retrollm.doctor import run_doctor

    report = run_doctor(args.config)
    payload = report.as_dict()

    if args.output:
        _write_json(args.output, payload)

    console = Console()
    if args.json:
        console.print(json.dumps(payload, indent=2))
    else:
        _render_doctor_report(console, payload, output_path=args.output)
    return 0 if report.ok else 1


def _requires_virtual_env(args: argparse.Namespace) -> bool:
    return args.cmd in {"smoke", "search"}


def _load_dotenv_if_available() -> None:
    try:
        from retrollm.utils.env import load_project_dotenv

        load_project_dotenv(_project_root())
    except Exception:
        return


def _enforce_virtual_env_if_needed(args: argparse.Namespace) -> None:
    if not _requires_virtual_env(args):
        return
    from retrollm.utils.env import ensure_virtual_env

    ensure_virtual_env()


def _set_common_search_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to config.yml")
    parser.add_argument("--smiles", required=True, help="Target SMILES")
    parser.add_argument("--iteration-limit", type=int, default=150)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--c-puct", type=float, default=1.4)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--llm-interval", type=int, default=50)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--constraints", default="", help="Natural-language search constraints")
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--filter-cutoff", type=float, default=0.0)
    parser.add_argument("--no-ringbreaker", action="store_true")
    parser.add_argument("--ringbreaker-topk", type=int, default=10)
    parser.add_argument("--max-routes", type=int, default=3)
    parser.add_argument("--no-llm-advisor", action="store_true")
    parser.add_argument("--no-llm-rerank", action="store_true")
    parser.add_argument("--no-llm-diagnosis", action="store_true")
    parser.add_argument("--no-llm-handoff", action="store_true")
    parser.add_argument("--no-llm-retry", action="store_true")
    parser.add_argument("--llm-max-retries", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="Print raw JSON result")
    parser.add_argument("--output", help="Write JSON result to a file")
    parser.add_argument("--report", help="Write markdown handoff report to a file")
    parser.add_argument("--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="retrollm")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser(
        "download-data", help="Download public artifacts and write config.yml"
    )
    p_dl.add_argument("path", help="Target directory")
    p_dl.set_defaults(func=_cmd_download_data)

    p_smoke = sub.add_parser(
        "smoke", help="Smoke test: load ONNX and run one forward pass"
    )
    p_smoke.add_argument("--config", required=True, help="Path to config.yml")
    p_smoke.add_argument("--smiles", required=True, help="Target SMILES")
    p_smoke.add_argument("--topk", type=int, default=50)
    p_smoke.set_defaults(func=_cmd_smoke)

    p_search = sub.add_parser(
        "search", help="Run standalone MCTS search with optional LLM meta-control"
    )
    _set_common_search_args(p_search)
    p_search.set_defaults(func=_cmd_search)

    p_doc = sub.add_parser(
        "doctor", help="Check local environment, dependencies, and data artifacts"
    )
    p_doc.add_argument("--config", default="./data/config.yml", help="Path to config.yml")
    p_doc.add_argument("--json", action="store_true", help="Print raw JSON report")
    p_doc.add_argument("--output", help="Write JSON report to a file")
    p_doc.set_defaults(func=_cmd_doctor)

    return p


def main() -> None:
    _load_dotenv_if_available()
    parser = build_parser()
    args = parser.parse_args()
    _enforce_virtual_env_if_needed(args)
    rc = args.func(args)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
