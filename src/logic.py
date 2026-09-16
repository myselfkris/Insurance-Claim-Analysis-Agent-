"""Three-valued boolean evaluation for coverage criteria expressions.

Used by the decision gate so AND / OR / NOT / parentheses are handled correctly.

Values:
  True  = criterion MET
  False = criterion NOT MET
  None  = criterion UNVERIFIED (unknown)

Three-valued truth tables:
  AND: False if any False; True if all True; otherwise None
  OR : True if any True; False if all False; otherwise None
  NOT: True <-> False; None stays None
"""
from __future__ import annotations

import ast
import re


def _and(a, b):
    if a is False or b is False:
        return False
    if a is True and b is True:
        return True
    return None


def _or(a, b):
    if a is True or b is True:
        return True
    if a is False and b is False:
        return False
    return None


def _not(a):
    if a is True:
        return False
    if a is False:
        return True
    return None


def _eval(node, statuses):
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result = True
            for value in node.values:
                result = _and(result, _eval(value, statuses))
            return result
        if isinstance(node.op, ast.Or):
            result = False
            for value in node.values:
                result = _or(result, _eval(value, statuses))
            return result
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _not(_eval(node.operand, statuses))
    if isinstance(node, ast.Name):
        return statuses.get(node.id)  # None if the id is unknown
    if isinstance(node, ast.Constant):
        return bool(node.value)
    raise ValueError(f"Unsupported expression node: {ast.dump(node)}")


def evaluate(expression: str, statuses: dict[str, bool | None]) -> bool | None:
    """Evaluate a boolean expression with three-valued logic."""
    expr = re.sub(r"\bAND\b", "and", expression)
    expr = re.sub(r"\bOR\b", "or", expr)
    expr = re.sub(r"\bNOT\b", "not", expr)
    tree = ast.parse(expr, mode="eval")
    return _eval(tree.body, statuses)


STATUS_TO_VALUE = {"met": True, "not_met": False, "unverified": None}


def decide(expression: str, criteria_verdicts: list[dict]) -> str:
    """Map criterion verdicts + a boolean expression to a decision.

    Returns one of: "appeal" | "uphold" | "more_info"
    """
    statuses = {v["criterion_id"]: STATUS_TO_VALUE[v["status"]] for v in criteria_verdicts}
    result = evaluate(expression, statuses)
    if result is True:
        return "appeal"
    if result is False:
        return "uphold"
    return "more_info"
