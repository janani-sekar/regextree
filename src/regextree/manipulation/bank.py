from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .load import load_rule_from_module
from .rule import ManipulationRule

MANIFEST_NAME = "bank.jsonl"
MODULES_DIRNAME = "modules"

_SAFE_NAME = re.compile(r"[^0-9a-zA-Z_]+")


def _slug(name: str) -> str:
    return _SAFE_NAME.sub("_", name.strip()).strip("_").lower() or "manipulation"


@dataclass(frozen=True)
class BankEntry:
    """One stored manipulation: where its source lives, plus provenance and stats."""

    manipulation_id: str
    name: str
    description: str
    module_path: Path
    provenance: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_record(self, root: Path) -> dict[str, Any]:
        return {
            "manipulation_id": self.manipulation_id,
            "name": self.name,
            "description": self.description,
            "module": str(self.module_path.relative_to(root)),
            "provenance": self.provenance,
            "stats": self.stats,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any], root: Path) -> BankEntry:
        return cls(
            manipulation_id=record["manipulation_id"],
            name=record["name"],
            description=record.get("description", ""),
            module_path=root / record["module"],
            provenance=record.get("provenance", {}),
            stats=record.get("stats", {}),
        )


class ManipulationBank:
    """A directory of discovered manipulations plus a JSONL manifest.

    `ManipulationRule` carries a live `apply_fn`, so a rule cannot be pickled
    meaningfully. The bank therefore stores the exported module *source* and
    reloads rules through `load_rule_from_module`, which keeps discovery output
    auditable and reproducible.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.modules_dir = self.root / MODULES_DIRNAME
        self.manifest_path = self.root / MANIFEST_NAME

    def __len__(self) -> int:
        return len(self.entries())

    def __contains__(self, manipulation_id: str) -> bool:
        return any(entry.manipulation_id == manipulation_id for entry in self.entries())

    def entries(self) -> list[BankEntry]:
        if not self.manifest_path.is_file():
            return []
        records = [
            json.loads(line)
            for line in self.manifest_path.read_text().splitlines()
            if line.strip()
        ]
        return [BankEntry.from_record(record, self.root) for record in records]

    def entry(self, manipulation_id: str) -> BankEntry:
        for entry in self.entries():
            if entry.manipulation_id == manipulation_id:
                return entry
        raise KeyError(f"Manipulation not in bank: {manipulation_id}")

    def manifest(self) -> pd.DataFrame:
        """Flat manifest view: provenance and stats keys become prefixed columns."""
        entries = self.entries()
        if not entries:
            return pd.DataFrame(
                columns=["manipulation_id", "name", "description", "module"]
            )
        return pd.json_normalize([entry.to_record(self.root) for entry in entries], sep="_")

    def add(
        self,
        export_path: str | Path,
        *,
        provenance: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
        manipulation_id: str | None = None,
    ) -> BankEntry:
        """Copy a discovered export into the bank and record it in the manifest.

        The default id is `<rule name>_<8 char source hash>`, so re-adding an
        identical export is a no-op rather than a duplicate.
        """
        export_path = Path(export_path)
        rule = load_rule_from_module(export_path)
        source = export_path.read_text()

        if manipulation_id is None:
            digest = hashlib.sha256(source.encode()).hexdigest()[:8]
            manipulation_id = f"{_slug(rule.name)}_{digest}"

        existing = [entry for entry in self.entries() if entry.manipulation_id == manipulation_id]
        if existing:
            return existing[0]

        self.modules_dir.mkdir(parents=True, exist_ok=True)
        module_path = self.modules_dir / f"{manipulation_id}.py"
        shutil.copyfile(export_path, module_path)

        entry = BankEntry(
            manipulation_id=manipulation_id,
            name=rule.name,
            description=rule.description,
            module_path=module_path,
            provenance=provenance or {},
            stats=stats or {},
        )
        with self.manifest_path.open("a") as handle:
            handle.write(json.dumps(entry.to_record(self.root)) + "\n")
        return entry

    def load(self, manipulation_id: str) -> ManipulationRule:
        return load_rule_from_module(self.entry(manipulation_id).module_path)

    def rules(self) -> dict[str, ManipulationRule]:
        """Every banked manipulation, keyed by manipulation_id.

        Feed this straight into `label_transitions` for full-dataset labeling.
        """
        return {
            entry.manipulation_id: load_rule_from_module(entry.module_path)
            for entry in self.entries()
        }
