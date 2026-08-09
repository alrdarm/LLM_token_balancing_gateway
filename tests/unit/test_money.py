"""Money coercion rules.

The spec requires money to be parsed and enforced with ``Decimal``, never
binary floating point. These tests pin that at the boundary, before any value
reaches the database.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from gateway.persistence.types import MONEY_SCALE, MoneyError, to_money


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0.050000000", Decimal("0.050000000")),
        ("0.003120000", Decimal("0.003120000")),
        ("0.20", Decimal("0.200000000")),
        (Decimal("1.5"), Decimal("1.500000000")),
        (0, Decimal("0")),
        (7, Decimal("7.000000000")),
        ("0.000000001", Decimal("0.000000001")),
    ],
)
def test_accepts_exact_values(value, expected):
    assert to_money(value) == expected


def test_quantizes_to_the_money_scale():
    assert to_money("0.2").as_tuple().exponent == -MONEY_SCALE


@pytest.mark.parametrize("value", [0.1, 1.0, -2.5, float("nan"), float("inf")])
def test_rejects_float_outright(value):
    """Accepting float would silently reintroduce binary floating point."""
    with pytest.raises(MoneyError, match="float"):
        to_money(value)


def test_rejects_bool():
    """bool is an int subclass; treating True as 1 dollar would be absurd."""
    with pytest.raises(MoneyError):
        to_money(True)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_rejects_non_finite_strings(value):
    with pytest.raises(MoneyError, match="finite"):
        to_money(value)


def test_rejects_excess_precision():
    """Silently rounding a caller's ceiling could let spend exceed it."""
    with pytest.raises(MoneyError, match="decimal places"):
        to_money("0.0000000001")


@pytest.mark.parametrize("value", ["", "abc", "1.2.3", None, object()])
def test_rejects_garbage(value):
    with pytest.raises(MoneyError):
        to_money(value)


def test_exact_decimal_arithmetic_is_preserved():
    """The classic float failure: 0.1 + 0.2 != 0.3."""
    total = to_money("0.1") + to_money("0.2")
    assert total == to_money("0.3")
