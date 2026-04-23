"""Sample schema for code-generation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CodeSample:
    """Unified benchmark sample for HumanEval+ and MBPP+."""

    id: str
    benchmark: str
    prompt_text: str
    entry_point: str
    canonical_solution: str
    contract: str
    metadata: dict[str, Any] = field(default_factory=dict)
