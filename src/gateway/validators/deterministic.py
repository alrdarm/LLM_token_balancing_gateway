"""Deterministic validators (§10 acceptance gates).

Deterministic in the strict sense: same request and same candidate always give
the same verdict, with no model call and no I/O. That matters beyond
reproducibility -- these run on every attempt, so anything slow or billable
here multiplies across repairs and escalations.

Each maps to a task class's "primary gate" in §10's matrix.

Severity choices are deliberate and follow §8's next-action table:

* **FAIL_REPAIRABLE** -- the model could plausibly fix this by trying again
  with the constraint restated (malformed JSON, wrong length).
* **FAIL_CAPABILITY** -- the model cannot do what was asked; retrying the same
  model is pointless, so §8 routes this to a compatible fallback instead.
* **FAIL_GROUNDING** -- a claim is unsupported; §8 sends this to a grounding
  route or escalation, never a naive repair.
"""

from __future__ import annotations

import json
import re
from typing import Any

from gateway.domain.enums import ValidationResult
from gateway.domain.requests import CanonicalRequest
from gateway.providers.base import ProviderResult
from gateway.validators.base import ValidationContext, ValidationOutcome

#: Statements that write or destroy data. A "review" or "generate" task that
#: emits one of these is a safety problem, not a style problem (§10).
_WRITE_STATEMENTS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE)\b", re.I
)

#: An UPDATE or DELETE without a WHERE clause: §9's "unbounded updates/deletes".
_UNBOUNDED_WRITE = re.compile(r"\b(UPDATE|DELETE)\b(?!.*\bWHERE\b)", re.I | re.S)

_CITATION = re.compile(r"\[(\d+)\]|\((https?://[^\s)]+)\)")
_CODE_FENCE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.S)


def _extract_json(text: str) -> tuple[Any, str | None]:
    """Parse JSON, tolerating a code fence around it.

    Models routinely wrap JSON in a fence even when asked not to. Treating that
    as a hard failure would burn a repair attempt on formatting rather than
    substance, so it is unwrapped first.
    """
    candidate = text.strip()

    fence = _CODE_FENCE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()

    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as exc:
        # The message names a position, not content, so it is safe to keep.
        return None, f"line {exc.lineno} column {exc.colno}"


class JSONParseValidator:
    """The output must be valid JSON."""

    name = "json_parse"

    def validate(
        self,
        request: CanonicalRequest,
        candidate: ProviderResult,
        context: ValidationContext,
    ) -> ValidationOutcome:
        parsed, error = _extract_json(candidate.text)
        if error is not None:
            return ValidationOutcome(
                validator=self.name,
                result=ValidationResult.FAIL_REPAIRABLE,
                detail_codes=("invalid_json",),
                repair_hint="Return only valid JSON, with no surrounding prose or code fence.",
            )
        del parsed
        return ValidationOutcome(validator=self.name, result=ValidationResult.PASS)


class SchemaCheckValidator:
    """The output must satisfy the caller's JSON schema.

    A hard gate: §2 says ``validation=none`` may never bypass a schema check,
    because the caller's own parser will fail on a violation regardless of what
    the gateway believes about validation depth.

    Implements the structural subset of JSON Schema the gateway needs --
    ``type``, ``required``, ``properties``, ``enum`` -- rather than pulling in a
    full validator, and reports which constraint failed without echoing values.
    """

    name = "schema_check"

    def validate(
        self,
        request: CanonicalRequest,
        candidate: ProviderResult,
        context: ValidationContext,
    ) -> ValidationOutcome:
        schema = request.output_format.json_schema
        if schema is None:
            # No schema demanded: nothing to check, and not a failure.
            return ValidationOutcome(
                validator=self.name,
                result=ValidationResult.PASS,
                detail_codes=("no_schema",),
            )

        parsed, error = _extract_json(candidate.text)
        if error is not None:
            return ValidationOutcome(
                validator=self.name,
                result=ValidationResult.FAIL_REPAIRABLE,
                detail_codes=("invalid_json",),
                repair_hint="Return only valid JSON matching the requested schema.",
            )

        codes = tuple(_check_schema(parsed, schema))
        if codes:
            return ValidationOutcome(
                validator=self.name,
                result=ValidationResult.FAIL_REPAIRABLE,
                detail_codes=codes,
                repair_hint="The JSON did not match the requested schema; correct the "
                "listed fields and return only JSON.",
            )

        return ValidationOutcome(validator=self.name, result=ValidationResult.PASS)


def _check_schema(value: Any, schema: dict[str, Any], path: str = "") -> list[str]:
    """Return violation codes. Codes name *fields*, never values."""
    codes: list[str] = []
    expected = schema.get("type")

    checkers: dict[str, type | tuple[type, ...]] = {
        "object": dict,
        "array": list,
        "string": str,
        "boolean": bool,
        "number": (int, float),
        "integer": int,
    }

    if expected in checkers:
        python_type = checkers[expected]
        # bool is an int subclass; a boolean is not an acceptable integer.
        if expected in ("number", "integer") and isinstance(value, bool):
            return [f"type_mismatch:{path or '$'}"]
        if not isinstance(value, python_type):
            return [f"type_mismatch:{path or '$'}"]

    if expected == "object" and isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                codes.append(f"missing_required:{path}{name}")

        for name, subschema in (schema.get("properties") or {}).items():
            if name in value:
                codes.extend(_check_schema(value[name], subschema, f"{path}{name}."))

    if expected == "array" and isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                codes.extend(_check_schema(item, item_schema, f"{path}{index}."))

    allowed = schema.get("enum")
    if allowed is not None and value not in allowed:
        codes.append(f"not_in_enum:{path or '$'}")

    return codes


class SQLParserValidator:
    """The output must contain something that parses as SQL."""

    name = "sql_parser"

    def validate(
        self,
        request: CanonicalRequest,
        candidate: ProviderResult,
        context: ValidationContext,
    ) -> ValidationOutcome:
        text = candidate.text
        fence = _CODE_FENCE.search(text)
        sql = (fence.group(1) if fence else text).strip()

        if not sql:
            return ValidationOutcome(
                validator=self.name,
                result=ValidationResult.FAIL_REPAIRABLE,
                detail_codes=("empty_output",),
                repair_hint="Return a SQL statement.",
            )

        keywords = ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE", "EXPLAIN")
        if not any(re.search(rf"\b{word}\b", sql, re.I) for word in keywords):
            return ValidationOutcome(
                validator=self.name,
                result=ValidationResult.FAIL_REPAIRABLE,
                detail_codes=("no_sql_statement",),
                repair_hint="Return a SQL statement, not prose.",
            )

        if sql.count("(") != sql.count(")"):
            return ValidationOutcome(
                validator=self.name,
                result=ValidationResult.FAIL_REPAIRABLE,
                detail_codes=("unbalanced_parentheses",),
                repair_hint="Return syntactically complete SQL.",
            )

        return ValidationOutcome(validator=self.name, result=ValidationResult.PASS)


class SQLSafetyValidator:
    """Read/write safety for SQL (§10).

    A review or explain task must not emit statements that mutate data. This is
    ``FAIL_CAPABILITY`` rather than repairable: the model produced something
    categorically outside what was asked for, and §8 routes that to a
    compatible fallback rather than asking the same model again.
    """

    name = "sql_safety"

    def validate(
        self,
        request: CanonicalRequest,
        candidate: ProviderResult,
        context: ValidationContext,
    ) -> ValidationOutcome:
        text = candidate.text
        fence = _CODE_FENCE.search(text)
        sql = (fence.group(1) if fence else text).strip()

        codes: list[str] = []
        if _WRITE_STATEMENTS.search(sql):
            codes.append("write_statement_present")
        if _UNBOUNDED_WRITE.search(sql):
            codes.append("unbounded_write")

        if codes:
            return ValidationOutcome(
                validator=self.name,
                result=ValidationResult.FAIL_CAPABILITY,
                detail_codes=tuple(codes),
            )

        return ValidationOutcome(validator=self.name, result=ValidationResult.PASS)


class LengthCheckValidator:
    """Output length constraints (§10 summarization and rewriting gates)."""

    name = "length_check"

    def __init__(self, *, min_chars: int = 1, max_ratio: float = 4.0) -> None:
        self.min_chars = min_chars
        self.max_ratio = max_ratio

    def validate(
        self,
        request: CanonicalRequest,
        candidate: ProviderResult,
        context: ValidationContext,
    ) -> ValidationOutcome:
        length = len(candidate.text.strip())

        if length < self.min_chars:
            return ValidationOutcome(
                validator=self.name,
                result=ValidationResult.FAIL_REPAIRABLE,
                detail_codes=("output_too_short",),
                repair_hint="Return a non-empty response.",
            )

        if request.max_output_tokens:
            # Compare against the caller's own ceiling rather than a constant.
            ceiling = request.max_output_tokens * self.max_ratio
            if length > ceiling:
                return ValidationOutcome(
                    validator=self.name,
                    result=ValidationResult.FAIL_REPAIRABLE,
                    detail_codes=("output_too_long",),
                    repair_hint="Return a shorter response within the requested limit.",
                )

        return ValidationOutcome(validator=self.name, result=ValidationResult.PASS)


class LabelCheckValidator:
    """Classification output must be one of the allowed labels (§10)."""

    name = "label_check"

    def validate(
        self,
        request: CanonicalRequest,
        candidate: ProviderResult,
        context: ValidationContext,
    ) -> ValidationOutcome:
        schema = request.output_format.json_schema or {}
        allowed = schema.get("enum")

        if not allowed:
            # No label set declared, so there is nothing to enforce.
            return ValidationOutcome(
                validator=self.name,
                result=ValidationResult.PASS,
                detail_codes=("no_label_set",),
            )

        answer = candidate.text.strip().strip('"')
        if answer not in allowed:
            return ValidationOutcome(
                validator=self.name,
                result=ValidationResult.FAIL_REPAIRABLE,
                detail_codes=("label_not_allowed",),
                repair_hint="Answer with exactly one of the permitted labels.",
            )

        return ValidationOutcome(validator=self.name, result=ValidationResult.PASS)


class CitationCheckValidator:
    """Grounded answers must cite sources (§10).

    ``FAIL_GROUNDING``, the most severe verdict: §8 sends it to a grounding
    route or escalation rather than a repair, because asking the same model to
    "add citations" invites it to invent them.
    """

    name = "citation_check"

    def validate(
        self,
        request: CanonicalRequest,
        candidate: ProviderResult,
        context: ValidationContext,
    ) -> ValidationOutcome:
        if not _CITATION.search(candidate.text):
            return ValidationOutcome(
                validator=self.name,
                result=ValidationResult.FAIL_GROUNDING,
                detail_codes=("no_citations",),
            )
        return ValidationOutcome(validator=self.name, result=ValidationResult.PASS)


class CodeCompileValidator:
    """Generated Python must at least parse (§10).

    Parsing, not executing: running model-generated code inside the gateway
    would be a remote code execution path. Real compilation belongs in an
    isolated sandbox, which v0.1 does not have.
    """

    name = "code_compile"

    def validate(
        self,
        request: CanonicalRequest,
        candidate: ProviderResult,
        context: ValidationContext,
    ) -> ValidationOutcome:
        fence = _CODE_FENCE.search(candidate.text)
        code = (fence.group(1) if fence else candidate.text).strip()

        if not code:
            return ValidationOutcome(
                validator=self.name,
                result=ValidationResult.FAIL_REPAIRABLE,
                detail_codes=("empty_output",),
                repair_hint="Return the requested code.",
            )

        import ast

        try:
            ast.parse(code)
        except SyntaxError:
            return ValidationOutcome(
                validator=self.name,
                result=ValidationResult.FAIL_REPAIRABLE,
                detail_codes=("syntax_error",),
                repair_hint="Return syntactically valid code.",
            )

        return ValidationOutcome(validator=self.name, result=ValidationResult.PASS)


def default_registry() -> Any:
    """A registry holding every deterministic validator."""
    from gateway.validators.base import ValidatorRegistry

    registry = ValidatorRegistry()
    for validator in (
        JSONParseValidator(),
        SchemaCheckValidator(),
        SQLParserValidator(),
        SQLSafetyValidator(),
        LengthCheckValidator(),
        LabelCheckValidator(),
        CitationCheckValidator(),
        CodeCompileValidator(),
    ):
        registry.register(validator)
    return registry
