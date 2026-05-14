"""Vector-store schema for ECL co-learning memories.

A *memory point* represents the durable artifact of one ECL step (or whole
run). The vector is typically a pooled hidden state (or the post-bus shared
state ``S``) and the payload carries the structured spec/tests/E* trace so
downstream agents can retrieve "what we tried before" and condition new runs
on it.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

COLLECTION_DEFAULT = "ecl_colearning_memory"
DEFAULT_VECTOR_SIZE = 64  # matches the default EntanglementBus state_dim


class PointPayload(BaseModel):
    """Structured metadata stored alongside each vector."""

    run_id: str
    step: int = 0
    e_star: float = 0.0
    spec: Dict[str, Any] = Field(default_factory=dict)
    tests: List[Dict[str, Any]] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    domain: Optional[str] = None
    created_at: float = Field(default_factory=lambda: time.time())


class MemoryPoint(BaseModel):
    """A single (id, vector, payload) tuple ready for upsert."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vector: List[float]
    payload: PointPayload
