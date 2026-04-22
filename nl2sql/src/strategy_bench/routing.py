"""Rule-based routing and SQL catalog support."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class MatchRule:
    """Single matching rule within a catalog entry."""

    type: str
    pattern: str | None = None
    keywords: list[str] = field(default_factory=list)
    priority: int = 0


@dataclass(slots=True)
class CatalogEntry:
    """One reusable/adaptable SQL catalog item."""

    id: str
    route_type: str
    match_rules: list[MatchRule]
    sql: str | None = None
    template: str | None = None
    placeholders: list[str] | dict[str, str] = field(default_factory=list)
    priority: int = 0
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    match_mode: str = "all"


@dataclass(slots=True)
class RouteDecision:
    """Decision made by the router for one query."""

    strategy: str
    catalog_entry_id: str | None = None
    sql: str | None = None
    placeholders: dict[str, str] = field(default_factory=dict)
    reason: str | None = None


class RoutingCatalog:
    """Load and resolve route candidates from YAML catalog files."""

    def __init__(self, entries: list[CatalogEntry]) -> None:
        self.entries = entries

    @classmethod
    def from_yaml(cls, path: Path) -> "RoutingCatalog":
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or []
        rows = payload.get("entries", []) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("Routing catalog must be a list or {entries: [...]} object")
        entries: list[CatalogEntry] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"Catalog row {index} must be an object")
            entries.append(_normalize_entry(row, index))
        return cls(entries)


class RuleBasedRouter:
    """Deterministic router using keywords/regex rules plus catalog priority."""

    _ROUTE_WEIGHT = {"reuse": 2, "adapt": 1}

    def __init__(self, catalog: RoutingCatalog) -> None:
        self._catalog = catalog

    def route(self, query: str) -> RouteDecision:
        candidates: list[tuple[int, int, CatalogEntry, dict[str, str]]] = []
        for entry in self._catalog.entries:
            matched, extracted, rule_priority = self._match_entry(entry, query)
            if matched:
                candidates.append(
                    (
                        self._ROUTE_WEIGHT.get(entry.route_type, 0),
                        max(entry.priority, rule_priority),
                        entry,
                        extracted,
                    )
                )
        if not candidates:
            return RouteDecision(strategy="generate", reason="no catalog rule matched")
        _, _, entry, extracted = max(candidates, key=lambda item: (item[0], item[1]))
        if entry.route_type == "reuse":
            return RouteDecision(
                strategy="reuse",
                catalog_entry_id=entry.id,
                sql=entry.sql,
                reason="matched catalog reuse rule",
            )
        placeholders = _resolve_placeholders(entry.placeholders, extracted)
        template = entry.template or ""
        adapted_sql = _apply_template(template, placeholders)
        if _has_unresolved_placeholder(adapted_sql):
            return RouteDecision(
                strategy="generate",
                catalog_entry_id=entry.id,
                placeholders=placeholders,
                reason="adaptation left unresolved placeholders",
            )
        return RouteDecision(
            strategy="adapt",
            catalog_entry_id=entry.id,
            sql=adapted_sql,
            placeholders=placeholders,
            reason="matched catalog adapt rule",
        )

    def _match_entry(self, entry: CatalogEntry, query: str) -> tuple[bool, dict[str, str], int]:
        extracted: dict[str, str] = {}
        priorities: list[int] = []
        matches: list[bool] = []
        for rule in entry.match_rules:
            matched, captured = _match_rule(rule, query)
            matches.append(matched)
            if matched:
                extracted.update(captured)
                priorities.append(rule.priority)
        if not matches:
            return False, {}, 0
        if entry.match_mode == "any":
            return any(matches), extracted, max(priorities, default=0)
        return all(matches), extracted, max(priorities, default=0)


def _normalize_entry(row: dict[str, Any], index: int) -> CatalogEntry:
    entry_id = row.get("id")
    route_type = row.get("route_type")
    if not isinstance(entry_id, str) or not entry_id.strip():
        raise ValueError(f"Catalog row {index} has invalid id")
    if route_type not in {"reuse", "adapt"}:
        raise ValueError(f"Catalog entry {entry_id!r} must have route_type reuse|adapt")
    raw_rules = row.get("match_rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError(f"Catalog entry {entry_id!r} must define non-empty match_rules")
    rules = []
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            raise ValueError(f"Catalog entry {entry_id!r} has invalid match rule")
        rules.append(
            MatchRule(
                type=str(raw_rule.get("type", "")).strip(),
                pattern=str(raw_rule["pattern"]).strip() if raw_rule.get("pattern") is not None else None,
                keywords=[str(keyword).strip().lower() for keyword in raw_rule.get("keywords", [])],
                priority=int(raw_rule.get("priority", 0) or 0),
            )
        )
    return CatalogEntry(
        id=entry_id.strip(),
        route_type=route_type,
        match_rules=rules,
        sql=str(row["sql"]).strip() if row.get("sql") is not None else None,
        template=str(row["template"]).strip() if row.get("template") is not None else None,
        placeholders=row.get("placeholders", []),
        priority=int(row.get("priority", 0) or 0),
        description=str(row["description"]).strip() if row.get("description") is not None else None,
        tags=[str(tag) for tag in row.get("tags", [])],
        match_mode=str(row.get("match_mode", "all")).strip().lower() or "all",
    )


def _match_rule(rule: MatchRule, query: str) -> tuple[bool, dict[str, str]]:
    lowered_query = query.lower()
    if rule.type == "keyword":
        matched = all(keyword in lowered_query for keyword in rule.keywords)
        return matched, {}
    if rule.type == "regex":
        if rule.pattern is None:
            return False, {}
        match = re.search(rule.pattern, query, flags=re.IGNORECASE)
        if match is None:
            return False, {}
        return True, {key: value for key, value in match.groupdict().items() if value is not None}
    raise ValueError(f"Unsupported match rule type: {rule.type}")


def _resolve_placeholders(
    spec: list[str] | dict[str, str],
    extracted: dict[str, str],
) -> dict[str, str]:
    if isinstance(spec, dict):
        merged = dict(spec)
        merged.update(extracted)
        return merged
    return {name: extracted[name] for name in spec if name in extracted}


def _apply_template(template: str, placeholders: dict[str, str]) -> str:
    rendered = template
    for key, value in placeholders.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value).replace(f"{{{key}}}", value)
    return rendered


def _has_unresolved_placeholder(sql: str) -> bool:
    return bool(re.search(r"\{\{[^{}]+\}\}|\{[^{}]+\}", sql))
