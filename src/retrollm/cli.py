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
from rich.text import Text


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


def _build_console(no_color: bool = False) -> Console:
    return Console(no_color=no_color)


def _yes_no_text(value: bool) -> Text:
    return Text("yes" if value else "no", style="bold green" if value else "bold red")


def _bool_text(value: bool) -> Text:
    return Text(str(value), style="bold green" if value else "bold red")


def _source_text(value: str) -> Text:
    style = {
        "llm": "bold cyan",
        "heuristic_fallback": "yellow",
        "heuristic_without_llm": "yellow",
        "empty": "dim",
    }.get(value, "white")
    return Text(value, style=style)


def _no_route_reason_message(reason: str | None) -> tuple[str, str]:
    return {
        "target_in_stock": (
            "No route candidates: target molecule is already in stock.",
            "green",
        ),
        "all_expansions_filtered_by_constraints": (
            "No route candidates: all expansions were filtered by constraints.",
            "yellow",
        ),
        "filter_runtime_error": (
            "No route candidates: filter policy failed at runtime and no valid expansions remained.",
            "red",
        ),
        "no_valid_expansions": (
            "No route candidates: no valid expansion from templates.",
            "yellow",
        ),
        "no_routes_after_ranking": ("No route candidates after ranking.", "yellow"),
    }.get(reason, ("No route candidates found.", "yellow"))


def _render_search_result(
    console: Console, payload: dict[str, Any], output_path: str | None, verbose: bool
) -> None:
    solved = bool(payload.get("solved"))
    reason = str(payload.get("no_route_reason")) if payload.get("no_route_reason") else None

    summary = Table(show_header=False, box=None, pad_edge=False)
    summary.add_column("Field", style="bold cyan", no_wrap=True)
    summary.add_column("Value", style="white")
    summary.add_row("Solved", _yes_no_text(solved))
    summary.add_row(
        "First solution iteration",
        Text(str(payload.get("first_solution_iteration") or "-"), style="white"),
    )
    summary.add_row("Iterations", Text(str(payload.get("iterations")), style="white"))
    summary.add_row(
        "Search time (s)",
        Text(_format_float(payload.get("search_time_s")), style="white"),
    )
    summary.add_row("Root children", Text(str(payload.get("root_children")), style="white"))
    retry_attempts = int(payload.get("retry_attempts", 0) or 0)
    summary.add_row(
        "Retry attempts",
        Text(str(retry_attempts), style="bold yellow" if retry_attempts > 0 else "green"),
    )
    if reason:
        summary.add_row("No-route reason", Text(reason, style="bold yellow"))
    if output_path:
        summary.add_row(
            "Saved JSON",
            Text(str(Path(output_path).resolve()), style="cyan"),
        )

    console.print(
        Panel(
            summary,
            title="Search Summary",
            border_style="green" if solved else "red",
            expand=False,
        )
    )

    llm_payload = payload.get("llm", {})
    constraints = llm_payload.get("constraints", {})
    if constraints:
        ctable = Table(show_header=False, box=None, pad_edge=False)
        ctable.add_column("Field", style="bold cyan", no_wrap=True)
        ctable.add_column("Value", style="white")
        source = str(constraints.get("source", "unknown"))
        ctable.add_row("source", _source_text(source))
        ctable.add_row(
            "avoid_reactants",
            Text(", ".join(constraints.get("avoid_reactants", [])) or "-", style="white"),
        )
        ctable.add_row(
            "avoid_template_indices",
            Text(
                ", ".join(str(v) for v in constraints.get("avoid_template_indices", []))
                or "-",
                style="white",
            ),
        )
        ctable.add_row("max_steps", Text(str(constraints.get("max_steps", "-")), style="white"))
        ctable.add_row(
            "prefer_in_stock_subgoals",
            _bool_text(bool(constraints.get("prefer_in_stock_subgoals", False))),
        )
        notes = str(constraints.get("notes", "")).strip()
        if notes:
            ctable.add_row("notes", Text(notes, style="dim"))
        console.print(
            Panel(
                ctable,
                title="Constraint Translation",
                border_style="cyan",
                expand=False,
            )
        )

    rerank = llm_payload.get("rerank", {})
    if rerank:
        rtable = Table(show_header=False, box=None, pad_edge=False)
        rtable.add_column("Field", style="bold cyan", no_wrap=True)
        rtable.add_column("Value", style="white")
        applied = bool(rerank.get("applied", False))
        mode = str(rerank.get("mode", "-"))
        rtable.add_row("applied", _bool_text(applied))
        rtable.add_row(
            "mode",
            Text(mode, style="bold cyan" if mode == "llm" else "yellow"),
        )
        rtable.add_row(
            "reason",
            Text(str(rerank.get("global_reason", rerank.get("reason", "-"))), style="white"),
        )
        if rerank.get("error"):
            rtable.add_row("error", Text(str(rerank.get("error")), style="bold red"))
        console.print(Panel(rtable, title="Route Rerank", border_style="blue", expand=False))

    diagnosis = llm_payload.get("diagnosis")
    if isinstance(diagnosis, dict) and diagnosis:
        dtable = Table(show_header=False, box=None, pad_edge=False)
        dtable.add_column("Field", style="bold yellow", no_wrap=True)
        dtable.add_column("Value", style="white")
        dtable.add_row("source", _source_text(str(diagnosis.get("source", "-"))))
        dtable.add_row("diagnosis", Text(str(diagnosis.get("diagnosis", "-")), style="white"))
        dtable.add_row("rationale", Text(str(diagnosis.get("rationale", "-")), style="dim"))
        dtable.add_row(
            "retry_plan",
            Text(json.dumps(diagnosis.get("retry_plan", {}), ensure_ascii=False), style="white"),
        )
        console.print(
            Panel(dtable, title="Failure Diagnosis", border_style="yellow", expand=False)
        )

    routes = payload.get("routes", [])
    if not routes:
        reason_text, reason_style = _no_route_reason_message(reason)
        console.print(
            Panel(
                Text(reason_text, style=f"bold {reason_style}"),
                title="Route Search",
                border_style=reason_style,
                expand=False,
            )
        )
    for idx, route in enumerate(routes, start=1):
        route_solved = bool(route.get("solved"))
        header = Text()
        header.append("score=", style="bold cyan")
        header.append(_format_float(route.get("score")), style="white")
        header.append("  visits=", style="bold cyan")
        header.append(str(route.get("visits")), style="white")
        header.append("  depth=", style="bold cyan")
        header.append(str(route.get("depth")), style="white")
        header.append("  solved=", style="bold cyan")
        header.append("yes" if route_solved else "no", style="bold green" if route_solved else "bold red")
        header.append("  llm_rank=", style="bold cyan")
        header.append(str(route.get("llm_rank", "-")), style="white")
        console.print(
            Panel(
                header,
                title=f"Route {idx}",
                border_style="green" if route_solved else "cyan",
                expand=False,
            )
        )
        steps = Table(header_style="bold cyan", row_styles=["none", "dim"])
        steps.add_column("Step", no_wrap=True)
        steps.add_column("Expanded", overflow="fold")
        steps.add_column("Reactants", overflow="fold")
        steps.add_column("Template", no_wrap=True)
        steps.add_column("Source", no_wrap=True)
        steps.add_column("p_tmpl", no_wrap=True, justify="right", style="green")
        steps.add_column("p_filt", no_wrap=True, justify="right", style="yellow")
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
            console.print(
                Text(
                    f"LLM reason: {route.get('llm_rank_reason')}",
                    style="bold cyan",
                )
            )

    handoff = llm_payload.get("handoff_markdown")
    if isinstance(handoff, str) and handoff.strip():
        preview_lines = handoff.strip().splitlines()[:10]
        console.print(
            Panel(
                "\n".join(preview_lines),
                title="Handoff Preview",
                border_style="white",
                expand=False,
            )
        )

    if verbose:
        cfg = payload.get("config_snapshot", {})
        console.print(
            Panel(
                json.dumps(cfg, indent=2),
                title="Config Snapshot",
                border_style="bright_black",
            )
        )
        events = llm_payload.get("events", [])
        if isinstance(events, list) and events:
            event_table = Table(
                "Stage",
                "Message",
                "Payload",
                header_style="bold cyan",
                row_styles=["none", "dim"],
            )
            for event in events[-20:]:
                stage = str(event.get("stage", ""))
                stage_style = {
                    "constraint_translation": "cyan",
                    "meta_control": "blue",
                    "subgoal_advisor": "green",
                    "route_rerank": "cyan",
                    "failure_diagnosis": "yellow",
                    "handoff": "white",
                }.get(stage, "white")
                event_table.add_row(
                    Text(stage, style=stage_style),
                    Text(str(event.get("message", "")), style="white"),
                    Text(json.dumps(event.get("payload", {}), ensure_ascii=False), style="dim"),
                )
            console.print(Panel(event_table, title="LLM Events", border_style="bright_black"))


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

    console = _build_console(no_color=args.no_color)
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
    table = Table(header_style="bold cyan", row_styles=["none", "dim"])
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

    console.print(Panel(table, title="Doctor Report", border_style="cyan", expand=False))
    if output_path:
        console.print(Text(f"Saved JSON: {Path(output_path).resolve()}", style="cyan"))
    overall_ok = bool(payload.get("ok"))
    console.print(Text(f"Overall: {'OK' if overall_ok else 'FAIL'}", style="bold green" if overall_ok else "bold red"))


def _cmd_doctor(args: argparse.Namespace) -> int:
    from retrollm.doctor import run_doctor

    report = run_doctor(args.config)
    payload = report.as_dict()

    if args.output:
        _write_json(args.output, payload)

    console = _build_console(no_color=args.no_color)
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
    parser.add_argument("--no-color", action="store_true", help="Disable colored terminal output")
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
    p_doc.add_argument("--no-color", action="store_true", help="Disable colored terminal output")
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
