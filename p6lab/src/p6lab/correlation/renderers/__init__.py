"""
p6lab.correlation.renderers
===========================

MatchBroker subscribers that turn engine matches into observable side effects.

Every renderer follows the same contract:

    class SomeRenderer:
        def __call__(self, match: PatternMatch) -> None: ...

Plug into a broker with:

    broker.subscribe(SomeRenderer(**config))

The renderer package is intentionally tiny — each module is single-purpose
and independently deletable. Adding a new renderer (webhook target, metrics
backend, ...) is a new file here, not a refactor of existing ones.
"""
from p6lab.correlation.renderers.audit_log import AuditLogRenderer
from p6lab.correlation.renderers.metrics import MetricsRenderer
from p6lab.correlation.renderers.outcome_tracker import OutcomeTrackerRenderer
from p6lab.correlation.renderers.webhook import WebhookRenderer

__all__ = [
    "AuditLogRenderer",
    "MetricsRenderer",
    "OutcomeTrackerRenderer",
    "WebhookRenderer",
]
