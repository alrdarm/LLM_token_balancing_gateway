"""Request classification (§5, §7).

v0.1 uses a deterministic heuristic classifier, not a model. That is a
deliberate choice, not a shortcut: classification feeds the risk floor and the
validation plan, so a non-deterministic classifier would make identical
requests route differently and break the determinism the inspect contract
requires (§4).

The classifier must never fail a request. When signals are weak it falls back
to the configured safe defaults and marks ``used_fallback``, because §7 says
CLASSIFYING exits to "frozen features or safe configured fallback" -- never to
an error.

Caller hints raise but never lower: a caller may declare higher risk than was
inferred, never less (§2).
"""

from __future__ import annotations

import re

from gateway.domain.enums import Capability, Quality, Risk, TaskClass
from gateway.domain.requests import (
    CanonicalRequest,
    resolve_strictest_quality,
    resolve_strictest_risk,
)
from gateway.domain.routing import RequestFeatures

CLASSIFIER_NAME = "heuristic"
CLASSIFIER_VERSION = "v0.1"

#: Fallback used when nothing matches. General reasoning with medium risk is
#: the conservative choice: it neither skips validation nor forces the most
#: expensive tier.
FALLBACK_TASK_CLASS = TaskClass.GENERAL_REASONING
FALLBACK_RISK = Risk.MEDIUM

#: Ordered keyword signals. First match wins, so more specific classes are
#: listed before the general ones they would otherwise be absorbed into.
_TASK_SIGNALS: tuple[tuple[TaskClass, re.Pattern[str]], ...] = (
    (TaskClass.SQL_REVIEW, re.compile(r"\b(sql|query plan|explain|select .*\bfrom\b)\b", re.I)),
    (TaskClass.CODE_REVIEW, re.compile(r"\b(review|refactor|debug|stack ?trace|bug)\b", re.I)),
    (
        TaskClass.CODE_GENERATION,
        re.compile(r"\b(implement|write (a )?(function|class|script)|code)\b", re.I),
    ),
    (TaskClass.GROUNDED_QA, re.compile(r"\b(cite|citation|according to|source[sd]?\b)", re.I)),
    (TaskClass.EXTRACTION, re.compile(r"\b(extract|parse|pull out|fields?)\b", re.I)),
    (TaskClass.SUMMARIZATION, re.compile(r"\b(summar(y|ise|ize)|tl;?dr|condense)\b", re.I)),
    (TaskClass.TRANSFORMATION, re.compile(r"\b(translate|convert|transform|rewrite as)\b", re.I)),
    (TaskClass.CLASSIFICATION, re.compile(r"\b(classif|categor|label|sentiment)\b", re.I)),
    (TaskClass.REWRITING, re.compile(r"\b(rewrite|rephrase|edit|proofread)\b", re.I)),
    (TaskClass.CREATIVE_WRITING, re.compile(r"\b(poem|story|fiction|creative|lyrics)\b", re.I)),
    (
        TaskClass.FACTUAL_QA,
        re.compile(r"\b(who|what|when|where|why|how) (is|are|was|were|did)\b", re.I),
    ),
)

#: Task classes whose failures carry real-world consequence, so they start
#: higher on the risk scale (§10 acceptance gates).
_ELEVATED_RISK: dict[TaskClass, Risk] = {
    TaskClass.SQL_REVIEW: Risk.HIGH,
    TaskClass.CODE_REVIEW: Risk.HIGH,
    TaskClass.CODE_GENERATION: Risk.MEDIUM,
    TaskClass.GROUNDED_QA: Risk.HIGH,
    TaskClass.STRUCTURED_DATA: Risk.MEDIUM,
    TaskClass.TOOL_SELECTION: Risk.HIGH,
    TaskClass.CREATIVE_WRITING: Risk.LOW,
    TaskClass.CLASSIFICATION: Risk.LOW,
}

#: How outputs of each class can be checked. Drives the validation plan.
_VERIFIABILITY: dict[TaskClass, str] = {
    TaskClass.SQL_REVIEW: "deterministic",
    TaskClass.CODE_GENERATION: "deterministic",
    TaskClass.CODE_REVIEW: "semi_deterministic",
    TaskClass.STRUCTURED_DATA: "deterministic",
    TaskClass.EXTRACTION: "deterministic",
    TaskClass.TOOL_SELECTION: "deterministic",
    TaskClass.CLASSIFICATION: "deterministic",
    TaskClass.TRANSFORMATION: "deterministic",
    TaskClass.GROUNDED_QA: "grounded",
    TaskClass.FACTUAL_QA: "grounded",
    TaskClass.SUMMARIZATION: "semi_deterministic",
    TaskClass.REWRITING: "semi_deterministic",
    TaskClass.CREATIVE_WRITING: "subjective",
    TaskClass.GENERAL_REASONING: "subjective",
}

_FRESHNESS = re.compile(
    r"\b(today|current|latest|recent|now|202[5-9]|this (week|month|year))\b", re.I
)

#: Ratio of expected output to input when the caller sets no ceiling.
DEFAULT_OUTPUT_RATIO = 0.5
DEFAULT_EXPECTED_OUTPUT_TOKENS = 512


def _classification_text(request: CanonicalRequest) -> str:
    """The text the heuristics read.

    Only the instructions and the final user turn: earlier turns describe past
    context rather than the task now being asked for.
    """
    parts = [request.system_instructions or ""]
    for message in reversed(request.conversation):
        if message.role == "user":
            parts.append(message.content)
            break
    return "\n".join(parts)


def _infer_task_class(text: str, structured_output: bool) -> tuple[TaskClass, float]:
    """Return the inferred class and a confidence in [0, 1]."""
    if structured_output:
        # A schema is a far stronger signal than any keyword.
        return TaskClass.STRUCTURED_DATA, 0.9

    for task_class, pattern in _TASK_SIGNALS:
        if pattern.search(text):
            return task_class, 0.7

    return FALLBACK_TASK_CLASS, 0.3


def _infer_complexity(request: CanonicalRequest, text: str) -> int:
    """Complexity on the spec's 1..5 scale.

    Driven by observable size rather than semantics: input length, turn count,
    and whether tools are in play.
    """
    score = 1
    if len(text) > 500:
        score += 1
    if len(text) > 4000:
        score += 1
    if len(request.conversation) > 4:
        score += 1
    if request.tools:
        score += 1
    return min(score, 5)


def classify(request: CanonicalRequest) -> RequestFeatures:
    """Derive frozen features for ``request``.

    Never raises: an unclassifiable request falls back to safe defaults with
    ``used_fallback`` set, as §7 requires.
    """
    text = _classification_text(request)
    controls = request.controls

    structured = request.output_format.requires_schema_validation
    inferred_class, confidence = _infer_task_class(text, structured)

    # A caller-declared class is a hint that wins over inference, since the
    # caller knows their own intent better than a keyword match does.
    task_class = controls.task_class or inferred_class
    used_fallback = controls.task_class is None and inferred_class is FALLBACK_TASK_CLASS
    if controls.task_class is not None:
        confidence = 1.0

    inferred_risk = _ELEVATED_RISK.get(task_class, FALLBACK_RISK)
    # Strictest wins: the caller may raise risk, never lower it (§2).
    risk = resolve_strictest_risk(inferred_risk, controls.risk) or inferred_risk

    expected_output = controls_expected_output(request)

    return RequestFeatures(
        task_class=task_class,
        complexity=_infer_complexity(request, text),
        risk=risk,
        privacy=controls.privacy,
        verifiability=_VERIFIABILITY.get(task_class, "subjective"),
        freshness_sensitive=bool(_FRESHNESS.search(text)),
        required_capabilities=frozenset(controls.required_capabilities),
        expected_output_tokens=expected_output,
        classifier_name=CLASSIFIER_NAME,
        classifier_version=CLASSIFIER_VERSION,
        confidence=confidence,
        used_fallback=used_fallback,
    )


def controls_expected_output(request: CanonicalRequest) -> int:
    """Expected output tokens, used for cost estimation.

    Prefers the caller's explicit ceiling. Estimating high is the safe
    direction: it reserves more budget than needed and releases the remainder,
    whereas underestimating risks exceeding the caller's cost ceiling.
    """
    if request.max_output_tokens is not None:
        return request.max_output_tokens

    scaled = int(request.estimated_input_tokens * DEFAULT_OUTPUT_RATIO)
    return max(scaled, DEFAULT_EXPECTED_OUTPUT_TOKENS)


def quality_floor_for(risk: Risk, requested: Quality) -> Quality:
    """Raise the quality floor to match risk (§2, §10).

    Risk sets a floor the caller cannot undercut; a caller asking for economy
    on a critical-risk task gets the risk-appropriate tier instead.
    """
    by_risk = {
        Risk.LOW: Quality.ECONOMY,
        Risk.MEDIUM: Quality.STANDARD,
        Risk.HIGH: Quality.HIGH,
        Risk.CRITICAL: Quality.CRITICAL,
    }
    return resolve_strictest_quality(by_risk[risk], requested)


def required_capabilities_for(features: RequestFeatures) -> frozenset[Capability]:
    """Capabilities implied by the classification itself."""
    required = set(features.required_capabilities)
    if features.task_class is TaskClass.STRUCTURED_DATA:
        required.add(Capability.JSON_SCHEMA)
    return frozenset(required)
