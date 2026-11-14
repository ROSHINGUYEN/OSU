"""
Permission Engine with Leased Permissions and Active Revocation.
Enforces strict capability leases for all gameplay actions.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.state_store import SafetyState, state_store


class PermissionDecisionType(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    EXPIRED = "expired"
    REVALIDATE = "revalidate"


@dataclass(frozen=True)
class PermissionDecision:
    """Structured response to a permission request."""
    decision: PermissionDecisionType
    reason_codes: List[str]
    lease_id: str
    lease_deadline_ns: int
    policy_version: str = "1.0"
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_allowed(self) -> bool:
        return self.decision == PermissionDecisionType.ALLOW


@dataclass
class PermissionLease:
    lease_id: str
    issued_at_ns: int
    expires_at_ns: int
    is_revoked: bool = False


class PermissionEngine:
    """Manages short-lived permission leases and active revocation."""

    def __init__(self, default_lease_ms: int = 50):
        self._default_lease_ms = default_lease_ms
        self._leases: Dict[str, PermissionLease] = {}

    def grant_lease(self, duration_ms: Optional[int] = None) -> PermissionDecision:
        """Issue a short-lived permission lease."""
        duration = duration_ms or self._default_lease_ms
        now_ns = clock.now_ns()
        deadline_ns = now_ns + int(duration * 1_000_000)
        lease_id = f"lease_{uuid.uuid4().hex[:8]}"

        lease = PermissionLease(
            lease_id=lease_id,
            issued_at_ns=now_ns,
            expires_at_ns=deadline_ns,
            is_revoked=False,
        )
        self._leases[lease_id] = lease

        self._sync_to_store()

        return PermissionDecision(
            decision=PermissionDecisionType.ALLOW,
            reason_codes=["LEASE_GRANTED"],
            lease_id=lease_id,
            lease_deadline_ns=deadline_ns,
        )

    def validate_lease(self, lease_id: str) -> bool:
        """Check if lease exists, is not revoked, and has not expired."""
        lease = self._leases.get(lease_id)
        if lease is None or lease.is_revoked:
            return False
        return clock.now_ns() <= lease.expires_at_ns

    def revoke_all(self, reason: str = "Emergency revocation") -> int:
        """Instantly revoke all active leases."""
        count = 0
        for lease in self._leases.values():
            if not lease.is_revoked:
                lease.is_revoked = True
                count += 1
        self._leases.clear()
        self._sync_to_store()
        return count

    def prune_expired(self) -> int:
        """Remove leases that have expired."""
        now_ns = clock.now_ns()
        expired = [k for k, v in self._leases.items() if now_ns > v.expires_at_ns]
        for k in expired:
            del self._leases[k]
        return len(expired)

    def _sync_to_store(self) -> None:
        active_summaries = {
            k: v.expires_at_ns for k, v in self._leases.items() if not v.is_revoked
        }
        state_store.mutate_domain(
            "safety",
            lambda curr: SafetyState(
                dependency_health=curr.dependency_health,
                active_leases=active_summaries,
                emergency_stop_triggered=curr.emergency_stop_triggered,
                updated_at_ns=clock.now_ns(),
            ),
            caller="safety_supervisor",
        )


permission_engine = PermissionEngine()
