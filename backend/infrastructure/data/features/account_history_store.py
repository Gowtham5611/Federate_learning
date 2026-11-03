"""Persistent Multi-Tenant Account History Feature Store for FedSecureAI.

Enables causal, cross-upload transaction history tracking while guaranteeing:
1. Strict Multi-Tenant / Session Isolation:
   State is strictly partitioned by `tenant_id` (or `session_id`). Cross-tenant
   or cross-session leakage is mathematically impossible.
2. Causal Temporal Ordering:
   Features for transaction t use only information strictly preceding t.
   State is advanced only after features have been computed.
3. Thread-Safe Concurrency:
   Thread-level locking guards all mutating operations.
4. TTL Expiration & Memory Boundedness:
   Configurable eviction policy removes expired accounts or trims excess capacity.
5. Opt-in Persistence:
   Default behavior remains upload-local (enable_persistent_history=False).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AccountProfile:
    """Historical behavior profile for a single financial account."""

    account_id: str
    sent_count: float = 0.0
    sent_total: float = 0.0
    sent_last_epoch: Optional[float] = None
    received_count: float = 0.0
    received_total: float = 0.0
    received_last_epoch: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to serializable dictionary."""
        return {
            "account_id": self.account_id,
            "sent_count": self.sent_count,
            "sent_total": round(self.sent_total, 2),
            "sent_last_epoch": self.sent_last_epoch,
            "received_count": self.received_count,
            "received_total": round(self.received_total, 2),
            "received_last_epoch": self.received_last_epoch,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class AccountHistoryFeatureStore:
    """Thread-safe, multi-tenant in-memory feature store for account prior histories."""

    def __init__(self, default_ttl_seconds: float = 86400.0, max_accounts_per_tenant: int = 100000) -> None:
        """Initialize feature store with isolation partitions."""
        self._tenants: Dict[str, Dict[str, AccountProfile]] = {}
        self._lock = threading.Lock()
        self._default_ttl_seconds = default_ttl_seconds
        self._max_accounts_per_tenant = max_accounts_per_tenant

    def get_profile(self, tenant_id: str, account_id: str) -> Optional[AccountProfile]:
        """Retrieve existing account profile for a specific tenant."""
        with self._lock:
            tenant_store = self._tenants.get(tenant_id)
            if not tenant_store:
                return None
            return tenant_store.get(account_id)

    def get_all_profiles(self, tenant_id: str) -> Dict[str, AccountProfile]:
        """Retrieve all account profiles for a given tenant."""
        with self._lock:
            tenant_store = self._tenants.get(tenant_id)
            if not tenant_store:
                return {}
            return dict(tenant_store)

    def update_profile(
        self,
        tenant_id: str,
        account_id: str,
        *,
        sent_count_delta: float = 0.0,
        sent_amount_delta: float = 0.0,
        sent_latest_epoch: Optional[float] = None,
        received_count_delta: float = 0.0,
        received_amount_delta: float = 0.0,
        received_latest_epoch: Optional[float] = None,
    ) -> AccountProfile:
        """Atomically update an account profile under a tenant."""
        now = time.time()
        with self._lock:
            if tenant_id not in self._tenants:
                self._tenants[tenant_id] = {}

            tenant_store = self._tenants[tenant_id]
            profile = tenant_store.get(account_id)
            if profile is None:
                profile = AccountProfile(account_id=account_id, created_at=now, updated_at=now)
                tenant_store[account_id] = profile

            profile.sent_count += sent_count_delta
            profile.sent_total += sent_amount_delta
            if sent_latest_epoch is not None:
                if profile.sent_last_epoch is None or sent_latest_epoch > profile.sent_last_epoch:
                    profile.sent_last_epoch = sent_latest_epoch

            profile.received_count += received_count_delta
            profile.received_total += received_amount_delta
            if received_latest_epoch is not None:
                if profile.received_last_epoch is None or received_latest_epoch > profile.received_last_epoch:
                    profile.received_last_epoch = received_latest_epoch

            profile.updated_at = now
            return profile

    def save_profiles(self, tenant_id: str, profiles: Dict[str, AccountProfile]) -> None:
        """Batch save or update account profiles for a tenant."""
        now = time.time()
        with self._lock:
            if tenant_id not in self._tenants:
                self._tenants[tenant_id] = {}

            tenant_store = self._tenants[tenant_id]
            for acc_id, prof in profiles.items():
                prof.updated_at = now
                tenant_store[acc_id] = prof

            # Enforce max accounts limit per tenant if exceeded
            if len(tenant_store) > self._max_accounts_per_tenant:
                self._prune_tenant_locked(tenant_id)

    def clear_tenant(self, tenant_id: str) -> None:
        """Explicitly purge all historical state for a specific tenant."""
        with self._lock:
            if tenant_id in self._tenants:
                del self._tenants[tenant_id]
                logger.info("Cleared account history state for tenant: %s", tenant_id)

    def clear_all(self) -> None:
        """Purge all state across all tenants."""
        with self._lock:
            self._tenants.clear()
            logger.info("Cleared all tenant account history states.")

    def evict_expired(self, max_age_seconds: Optional[float] = None) -> int:
        """Evict account records older than max_age_seconds across all tenants."""
        ttl = max_age_seconds if max_age_seconds is not None else self._default_ttl_seconds
        cutoff = time.time() - ttl
        evicted_count = 0

        with self._lock:
            for tenant_id, tenant_store in list(self._tenants.items()):
                expired_keys = [
                    acc for acc, prof in tenant_store.items() if prof.updated_at < cutoff
                ]
                for k in expired_keys:
                    del tenant_store[k]
                    evicted_count += 1
                if len(tenant_store) == 0:
                    del self._tenants[tenant_id]

        if evicted_count > 0:
            logger.info("Evicted %d expired account profiles across tenants.", evicted_count)
        return evicted_count

    def _prune_tenant_locked(self, tenant_id: str) -> None:
        """Prune oldest accounts in tenant when capacity is exceeded (lock must be held)."""
        tenant_store = self._tenants.get(tenant_id)
        if not tenant_store or len(tenant_store) <= self._max_accounts_per_tenant:
            return

        sorted_accs = sorted(tenant_store.items(), key=lambda kv: kv[1].updated_at)
        excess = len(tenant_store) - self._max_accounts_per_tenant
        for i in range(excess):
            acc_to_drop = sorted_accs[i][0]
            del tenant_store[acc_to_drop]

    def get_stats(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get telemetry summary of stored accounts."""
        with self._lock:
            if tenant_id:
                store = self._tenants.get(tenant_id, {})
                return {
                    "tenant_id": tenant_id,
                    "distinct_accounts": len(store),
                    "total_tenants": len(self._tenants),
                }
            total_accounts = sum(len(store) for store in self._tenants.values())
            return {
                "total_tenants": len(self._tenants),
                "total_distinct_accounts": total_accounts,
                "tenants": {t: len(store) for t, store in self._tenants.items()},
            }


# Module-level singleton
_GLOBAL_ACCOUNT_HISTORY_STORE: Optional[AccountHistoryFeatureStore] = None
_STORE_INIT_LOCK = threading.Lock()


def get_account_history_store() -> AccountHistoryFeatureStore:
    """Access the global singleton AccountHistoryFeatureStore."""
    global _GLOBAL_ACCOUNT_HISTORY_STORE
    if _GLOBAL_ACCOUNT_HISTORY_STORE is None:
        with _STORE_INIT_LOCK:
            if _GLOBAL_ACCOUNT_HISTORY_STORE is None:
                _GLOBAL_ACCOUNT_HISTORY_STORE = AccountHistoryFeatureStore()
    return _GLOBAL_ACCOUNT_HISTORY_STORE

