"""
Company Manager
Manages per-company Qdrant collection indices.

Each company maps to a dedicated Qdrant collection (index):
  e.g. company "ManageEngine"  -> collection "company_manageengine"
       company "ABC Corp"      -> collection "company_abc_corp"

The registry is persisted as  data/companies.json
"""
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Default path for company registry file
_DEFAULT_REGISTRY = os.path.join(
    os.path.dirname(__file__), "..", "data", "companies.json"
)


def _slugify(name: str) -> str:
    """Convert a company display name to a safe Qdrant collection name."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return f"company_{slug}"


class CompanyManager:
    """
    Manages the registry of companies and their Qdrant collection names.

    Registry JSON structure:
    {
      "ManageEngine": {
        "collection": "company_manageengine",
        "index": 0,
        "created_at": "2026-03-08T...",
        "doc_count": 0,
        "sources": ["pdf", "url"]
      },
      ...
    }
    """

    def __init__(self, registry_path: str = _DEFAULT_REGISTRY):
        self.registry_path = registry_path
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        self._data: Dict[str, Dict] = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> Dict:
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"[CompanyManager] Failed to read registry: {e}")
        return {}

    def _save(self):
        try:
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[CompanyManager] Failed to save registry: {e}")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get_or_create(self, company_name: str) -> Dict[str, Any]:
        """
        Return existing company entry or create a new one.
        Returns the company record dict.
        """
        # Normalize key
        key = company_name.strip()
        if key in self._data:
            return self._data[key]

        # Assign next available index
        existing_indices = [v["index"] for v in self._data.values()]
        next_index = max(existing_indices, default=-1) + 1

        record = {
            "collection": _slugify(key),
            "index": next_index,
            "created_at": datetime.utcnow().isoformat(),
            "doc_count": 0,
            "sources": [],
        }
        self._data[key] = record
        self._save()
        logger.info(
            f"[CompanyManager] Created company '{key}' -> "
            f"collection='{record['collection']}', index={next_index}"
        )
        return record

    def get(self, company_name: str) -> Optional[Dict[str, Any]]:
        """Return company record or None if not found."""
        return self._data.get(company_name.strip())

    def list_companies(self) -> List[Dict[str, Any]]:
        """Return all companies sorted by index."""
        result = []
        for name, rec in self._data.items():
            result.append({"name": name, **rec})
        result.sort(key=lambda x: x["index"])
        return result

    def update_doc_count(self, company_name: str, delta: int, source_type: str = ""):
        """Increment document count for a company."""
        key = company_name.strip()
        if key not in self._data:
            return
        self._data[key]["doc_count"] = self._data[key].get("doc_count", 0) + delta
        if source_type and source_type not in self._data[key].get("sources", []):
            self._data[key].setdefault("sources", []).append(source_type)
        self._save()

    def rename(self, old_name: str, new_name: str) -> bool:
        """Rename a company (keeps same collection / index)."""
        old_key = old_name.strip()
        new_key = new_name.strip()
        if old_key not in self._data:
            return False
        if new_key in self._data:
            return False
        self._data[new_key] = self._data.pop(old_key)
        self._save()
        logger.info(f"[CompanyManager] Renamed '{old_key}' -> '{new_key}'")
        return True

    def delete(self, company_name: str) -> Optional[str]:
        """
        Remove a company from the registry.
        Returns the Qdrant collection name (so caller can also drop the collection).
        """
        key = company_name.strip()
        rec = self._data.pop(key, None)
        if rec:
            self._save()
            logger.info(f"[CompanyManager] Deleted company '{key}'")
            return rec["collection"]
        return None

    def get_collection(self, company_name: str) -> Optional[str]:
        """Shortcut: get Qdrant collection name for a company."""
        rec = self._data.get(company_name.strip())
        return rec["collection"] if rec else None

    def all_collections(self) -> List[str]:
        """Return all existing collection names."""
        return [rec["collection"] for rec in self._data.values()]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager: Optional[CompanyManager] = None


def get_company_manager(registry_path: str = _DEFAULT_REGISTRY) -> CompanyManager:
    global _manager
    if _manager is None:
        _manager = CompanyManager(registry_path)
    return _manager
