from __future__ import annotations

from retrollm.cli import _no_route_reason_message, build_parser


def test_search_parser_supports_no_color() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["search", "--config", "./data/config.yml", "--smiles", "CCO", "--no-color"]
    )
    assert args.no_color is True


def test_doctor_parser_supports_no_color() -> None:
    parser = build_parser()
    args = parser.parse_args(["doctor", "--no-color"])
    assert args.no_color is True


def test_no_route_reason_message_target_in_stock() -> None:
    message, style = _no_route_reason_message("target_in_stock")
    assert "already in stock" in message
    assert style == "green"
