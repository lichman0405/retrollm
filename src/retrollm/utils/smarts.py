"""SMARTS helpers.

This module is intentionally lightweight: it does not attempt to *interpret* or
validate reaction SMARTS beyond simple parsing and readability formatting.
"""

from __future__ import annotations

import re


_REACTION_ARROW = ">>"
_ATOM_MAP_RE = re.compile(r":\d+(?=\])")


def split_reaction_smarts(smarts: str) -> tuple[str, list[str]]:
    """Split a reaction SMARTS into (product_pattern, reactant_patterns).

    RetroLLM templates are stored in retrosynthesis direction:

        product_pattern >> reactant1.reactant2...

    Returns:
        (lhs, reactants)

    Raises:
        ValueError: if the input does not contain exactly one ">>" arrow.
    """

    if not isinstance(smarts, str) or not smarts.strip():
        raise ValueError("smarts must be a non-empty string")

    parts = smarts.split(_REACTION_ARROW)
    if len(parts) != 2:
        raise ValueError("reaction SMARTS must contain exactly one '>>'")

    lhs = parts[0].strip()
    rhs = parts[1].strip()

    reactants = [chunk.strip() for chunk in rhs.split(".") if chunk.strip()]
    return lhs, reactants


def strip_atom_mapping(smarts: str) -> str:
    """Remove atom-mapping indices like ':12' in bracket atoms.

    This improves readability for humans but should NOT be fed back into RDChiral/RDKit
    as a substitute for the original template.
    """

    if not isinstance(smarts, str):
        return str(smarts)
    return _ATOM_MAP_RE.sub("", smarts)


def format_reaction_smarts(smarts: str, *, simplify: bool = True) -> str:
    """Format reaction SMARTS into a more readable, multi-line representation."""

    lhs, reactants = split_reaction_smarts(smarts)
    if simplify:
        lhs = strip_atom_mapping(lhs)
        reactants = [strip_atom_mapping(r) for r in reactants]

    lines: list[str] = []
    if simplify:
        lines.append("Product pattern (atom maps removed):")
    else:
        lines.append("Product pattern:")
    lines.append(f"  {lhs}")
    lines.append("Reactant patterns:")
    if reactants:
        for r in reactants:
            lines.append(f"  - {r}")
    else:
        lines.append("  - (none)")
    return "\n".join(lines)
