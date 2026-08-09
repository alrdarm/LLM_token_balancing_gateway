"""Storage types that preserve domain invariants across SQLite and PostgreSQL.

The spec requires money to be parsed and enforced with ``Decimal``, never
binary floating point. SQLAlchemy's :class:`~sqlalchemy.Numeric` does not
deliver that on SQLite: SQLite has no exact decimal type, so values round-trip
through Python ``float`` and lose exactness. Storing money as a scaled integer
is therefore not an optimisation but a correctness requirement.

:class:`Money` stores nanodollars (scale 9) as a 64-bit integer on SQLite and
as ``NUMERIC(20, 9)`` on PostgreSQL. Both are exact, and both keep SQL-side
arithmetic and comparison working, which the budget reservation algorithm in
§6 depends on.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy import JSON, BigInteger, DateTime, Dialect, Numeric
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TypeDecorator

#: Decimal places retained for monetary values. The spec's examples carry nine
#: ("0.003120000"), which sub-cent per-token pricing genuinely needs.
MONEY_SCALE = 9

#: Multiplier between a dollar amount and its stored integer representation.
MONEY_SCALE_FACTOR = Decimal(10) ** MONEY_SCALE

#: Total digits for the PostgreSQL column: 11 integral + 9 fractional. That
#: caps a single value near 10^11 USD, far above any real ceiling, while
#: staying inside a signed 64-bit integer once scaled for SQLite.
MONEY_PRECISION = 20

_QUANTUM = Decimal(1).scaleb(-MONEY_SCALE)


class MoneyError(ValueError):
    """A value cannot be represented exactly as money."""


def to_money(value: Decimal | int | str) -> Decimal:
    """Coerce ``value`` to an exact money :class:`~decimal.Decimal`.

    ``float`` is rejected outright rather than converted: accepting it would
    silently reintroduce the binary floating point the spec forbids, and the
    caller almost certainly has an exact source string available.
    """
    if isinstance(value, bool) or isinstance(value, float):
        raise MoneyError(
            f"money must not be constructed from {type(value).__name__}; use Decimal, int, or str"
        )

    try:
        amount = Decimal(value)
    except (InvalidOperation, TypeError) as exc:
        raise MoneyError(f"not a valid money value: {value!r}") from exc

    if not amount.is_finite():
        raise MoneyError(f"money must be finite, got {value!r}")

    quantized = amount.quantize(_QUANTUM)
    if quantized != amount:
        raise MoneyError(f"money supports at most {MONEY_SCALE} decimal places, got {value!r}")
    return quantized


class Money(TypeDecorator[Decimal]):
    """Exact decimal money, portable across SQLite and PostgreSQL."""

    impl = Numeric
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        if dialect.name == "sqlite":
            # Scaled integer: SQLite would otherwise round-trip via float.
            return dialect.type_descriptor(BigInteger())
        return dialect.type_descriptor(
            Numeric(precision=MONEY_PRECISION, scale=MONEY_SCALE, asdecimal=True)
        )

    def process_bind_param(
        self, value: Decimal | int | str | None, dialect: Dialect
    ) -> Decimal | int | None:
        if value is None:
            return None
        amount = to_money(value)
        if dialect.name == "sqlite":
            scaled = (amount * MONEY_SCALE_FACTOR).to_integral_exact()
            return int(scaled)
        return amount

    def process_result_value(self, value: Any, dialect: Dialect) -> Decimal | None:
        if value is None:
            return None
        if dialect.name == "sqlite":
            return (Decimal(int(value)) / MONEY_SCALE_FACTOR).quantize(_QUANTUM)
        return Decimal(value).quantize(_QUANTUM)


class JSONDocument(TypeDecorator[Any]):
    """A JSON document, stored as ``JSONB`` on PostgreSQL and ``JSON`` on SQLite.

    Defined as a type rather than an inline ``with_variant`` so migrations
    render one portable name instead of dialect-specific imports, keeping every
    generated migration runnable on both backends.
    """

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())


class UTCDateTime(TypeDecorator[datetime]):
    """Timezone-aware timestamps normalised to UTC.

    SQLite has no timezone support and would hand back naive datetimes, which
    compare incorrectly against aware ones and silently corrupt deadline
    arithmetic. Naive input is rejected rather than assumed to be UTC.
    """

    impl = DateTime
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        return dialect.type_descriptor(DateTime(timezone=True))

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime rejected; timestamps must be timezone-aware")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
