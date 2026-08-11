"""Validation gates (§8).

The framing the whole gateway rests on: **a request succeeds only when a
generation attempt passes its required validation gates. Provider success alone
is not gateway success.**

Aggregation follows §8's severity order strictly::

    FAIL_GROUNDING > FAIL_CAPABILITY > FAIL_QUALITY > FAIL_REPAIRABLE
                   > INDETERMINATE > PASS

Only *required* validators gate success. An advisory validator's verdict is
recorded for telemetry but cannot fail a request, because a request whose
outcome depended on an optional check would be unpredictable to the caller.

Validators return codes, never prose about the output: §6 forbids storing raw
content, and a validator that quoted the text it rejected would smuggle model
output into the database through the back door.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from gateway.domain.enums import VALIDATION_SEVERITY, ValidationResult
from gateway.domain.requests import CanonicalRequest
from gateway.providers.base import ProviderResult


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    """One validator's verdict."""

    validator: str
    result: ValidationResult
    #: Machine-readable codes only. Never the offending content.
    detail_codes: tuple[str, ...] = ()
    #: Whether this verdict gates success.
    required: bool = True
    duration_ms: int | None = None
    #: Guidance for a repair attempt, if the failure is repairable. Describes
    #: the *constraint*, never the output that violated it.
    repair_hint: str | None = None

    @property
    def passed(self) -> bool:
        return self.result is ValidationResult.PASS


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Every verdict for one attempt, plus the aggregate."""

    outcomes: tuple[ValidationOutcome, ...] = ()

    @property
    def aggregate(self) -> ValidationResult:
        """The most severe verdict among *required* validators (§8)."""
        required = [outcome for outcome in self.outcomes if outcome.required]
        if not required:
            return ValidationResult.PASS

        return min(
            (outcome.result for outcome in required),
            key=VALIDATION_SEVERITY.index,
        )

    @property
    def passed(self) -> bool:
        return self.aggregate is ValidationResult.PASS

    @property
    def deciding(self) -> ValidationOutcome | None:
        """The verdict that determined the aggregate, for telemetry."""
        aggregate = self.aggregate
        for outcome in self.outcomes:
            if outcome.required and outcome.result is aggregate:
                return outcome
        return None

    def repair_hints(self) -> tuple[str, ...]:
        """Concise failure descriptions for a repair prompt (§7).

        §7 says repair uses "original request + concise failures", so this
        returns the constraints that were violated -- not the output.
        """
        return tuple(
            outcome.repair_hint
            for outcome in self.outcomes
            if outcome.repair_hint and not outcome.passed
        )


@dataclass(frozen=True, slots=True)
class ValidationContext:
    """What a validator may see besides the candidate output."""

    attempt_number: int
    is_repair: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Validator(Protocol):
    """The contract every validator satisfies (§5)."""

    name: str

    def validate(
        self,
        request: CanonicalRequest,
        candidate: ProviderResult,
        context: ValidationContext,
    ) -> ValidationOutcome:
        """Judge ``candidate`` against ``request``."""
        ...


class ValidatorRegistry:
    """Validators available to this deployment."""

    def __init__(self) -> None:
        self._validators: dict[str, Validator] = {}

    def register(self, validator: Validator) -> None:
        self._validators[validator.name] = validator

    def get(self, name: str) -> Validator | None:
        return self._validators.get(name)

    def names(self) -> frozenset[str]:
        return frozenset(self._validators)

    def missing(self, required: frozenset[str]) -> frozenset[str]:
        """Required validators this deployment cannot run.

        §10 excludes every model when a required gate is unavailable: a request
        whose gates cannot be enforced must not be attempted at all.
        """
        return required - self.names()


def run_plan(
    registry: ValidatorRegistry,
    plan: tuple[str, ...],
    request: CanonicalRequest,
    candidate: ProviderResult,
    context: ValidationContext,
) -> ValidationReport:
    """Run the planned validators in order and aggregate their verdicts.

    Ordered rather than parallel because §7 says "ordered required validators",
    and because a cheap deterministic check failing makes an expensive judge
    call pointless -- so the loop stops at the first *grounding* or *capability*
    failure, which no later verdict could soften.
    """
    outcomes: list[ValidationOutcome] = []

    for name in plan:
        validator = registry.get(name)
        if validator is None:
            # Planning should have caught this; treat it as indeterminate
            # rather than a silent pass, so it can never look like success.
            outcomes.append(
                ValidationOutcome(
                    validator=name,
                    result=ValidationResult.INDETERMINATE,
                    detail_codes=("validator_unavailable",),
                )
            )
            continue

        outcome = validator.validate(request, candidate, context)
        outcomes.append(outcome)

        if outcome.required and outcome.result in (
            ValidationResult.FAIL_GROUNDING,
            ValidationResult.FAIL_CAPABILITY,
        ):
            # Nothing later can improve on this, and continuing would spend
            # judge budget on an already-doomed candidate.
            break

    return ValidationReport(outcomes=tuple(outcomes))
