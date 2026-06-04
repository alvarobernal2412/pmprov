"""
Asynchronous provenance store — per-entity relational schema.

Backend selection (auto-detected at import time):
  - DuckDB  – preferred; handles Parquet natively and is OLAP-optimised.
  - SQLite  – fallback when DuckDB is not installed.

DataFrame artifacts are persisted as immutable Parquet snapshots on disk,
referenced by path from the metadata DB.
"""

from __future__ import annotations

import json
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import duckdb as _duckdb
    _BACKEND = "duckdb"
except ImportError:
    import sqlite3 as _sqlite3  # type: ignore[no-redef]
    _BACKEND = "sqlite"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _p(*args):
    return list(args) if _BACKEND == "duckdb" else tuple(args)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _commit(con) -> None:
    if hasattr(con, "commit"):
        con.commit()


# ------------------------------------------------------------------
# DDL — 14 per-entity tables
# ------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS analysis_histories (history_id VARCHAR PRIMARY KEY, name VARCHAR, created_at VARCHAR NOT NULL, active_state_id VARCHAR);
CREATE TABLE IF NOT EXISTS analysis_branches (branch_id VARCHAR PRIMARY KEY, history_id VARCHAR NOT NULL, name VARCHAR NOT NULL, starts_at_state_id VARCHAR NOT NULL);
CREATE TABLE IF NOT EXISTS analysis_states (state_id VARCHAR PRIMARY KEY, history_id VARCHAR NOT NULL, branch_id VARCHAR NOT NULL, produced_by_step_id VARCHAR, derived_from_state_id VARCHAR, timestamp VARCHAR NOT NULL);
CREATE TABLE IF NOT EXISTS agents (agent_id VARCHAR PRIMARY KEY, history_id VARCHAR NOT NULL, agent_type VARCHAR NOT NULL, username VARCHAR);
CREATE TABLE IF NOT EXISTS runtime_environments (env_id VARCHAR PRIMARY KEY, history_id VARCHAR NOT NULL, tool_version VARCHAR NOT NULL, library_versions VARCHAR NOT NULL, runtime VARCHAR);
CREATE TABLE IF NOT EXISTS operation_types (type_id VARCHAR PRIMARY KEY, name VARCHAR NOT NULL);
CREATE TABLE IF NOT EXISTS operations (operation_id VARCHAR PRIMARY KEY, name VARCHAR NOT NULL, operation_type_id VARCHAR NOT NULL, step_category_id VARCHAR);
CREATE TABLE IF NOT EXISTS analysis_steps (step_id VARCHAR PRIMARY KEY, history_id VARCHAR NOT NULL, input_state_id VARCHAR NOT NULL, output_state_id VARCHAR NOT NULL, agent_id VARCHAR NOT NULL, env_id VARCHAR NOT NULL, operation_id VARCHAR NOT NULL, func_name VARCHAR NOT NULL, raw_line VARCHAR, param_fingerprint VARCHAR, timestamp VARCHAR NOT NULL);
CREATE TABLE IF NOT EXISTS parameter_values (pv_id VARCHAR PRIMARY KEY, step_id VARCHAR NOT NULL, param_id VARCHAR NOT NULL, value_type VARCHAR NOT NULL, value_json VARCHAR NOT NULL);
CREATE TABLE IF NOT EXISTS artifacts (artifact_id VARCHAR PRIMARY KEY, history_id VARCHAR NOT NULL, name VARCHAR NOT NULL, artifact_type VARCHAR NOT NULL);
CREATE TABLE IF NOT EXISTS artifact_states (artifact_state_id VARCHAR PRIMARY KEY, artifact_id VARCHAR NOT NULL, analysis_state_id VARCHAR NOT NULL, mime_type VARCHAR NOT NULL, checksum VARCHAR NOT NULL, content_ref VARCHAR NOT NULL, size_bytes INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS deltas (delta_id VARCHAR PRIMARY KEY, step_id VARCHAR NOT NULL, kind VARCHAR NOT NULL, mutation_type VARCHAR, modification_type VARCHAR, rows_delta INTEGER, columns_added VARCHAR, columns_removed VARCHAR, dtype_changes VARCHAR);
CREATE TABLE IF NOT EXISTS pipelines (pipeline_id VARCHAR PRIMARY KEY, history_id VARCHAR NOT NULL, name VARCHAR NOT NULL, created_at VARCHAR NOT NULL);
CREATE TABLE IF NOT EXISTS pipeline_fragments (fragment_id VARCHAR PRIMARY KEY, pipeline_id VARCHAR NOT NULL, step_ids VARCHAR NOT NULL, position INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS step_categories (category_id VARCHAR PRIMARY KEY, name VARCHAR NOT NULL);
"""


class StorageBackend:
    """
    Thin async façade over DuckDB / SQLite + Parquet.

    All DB writes are serialised through a single background thread so the
    notebook's execution thread is never blocked by I/O.
    """

    def __init__(
        self,
        db_path: str | Path = "provenance.db",
        artifact_dir: str | Path = "artifacts",
    ) -> None:
        self.db_path = Path(db_path)
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        # Single-worker executor serialises all writes to the same DB file.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="provtrack-io")
        # Block until schema is ready before accepting any async writes.
        self._executor.submit(self._init_schema).result()

    # ------------------------------------------------------------------
    # Internal connection helper
    # ------------------------------------------------------------------

    def _connect(self, read_only: bool = False):
        if _BACKEND == "duckdb":
            return _duckdb.connect(str(self.db_path), read_only=read_only)
        return _sqlite3.connect(str(self.db_path))

    def _init_schema(self) -> None:
        con = self._connect()
        try:
            for stmt in _DDL.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    con.execute(stmt)
            _commit(con)
            # Migration: add step_category_id column to operations if absent
            try:
                con.execute("ALTER TABLE operations ADD COLUMN step_category_id VARCHAR")
                _commit(con)
            except Exception:
                pass
        finally:
            con.close()

    # ------------------------------------------------------------------
    # Public async save API
    # ------------------------------------------------------------------

    def save_history_async(self, history) -> None:
        self._executor.submit(self._write_history, history)

    def save_branch_async(self, branch) -> None:
        self._executor.submit(self._write_branch, branch)

    def save_state_async(self, state) -> None:
        self._executor.submit(self._write_state, state)

    def save_step_async(self, step, func_name, raw_line, param_fingerprint, history_id) -> None:
        self._executor.submit(self._write_step, step, func_name, raw_line, param_fingerprint, history_id)

    def save_operation_async(self, op, op_type, step_category=None) -> None:
        self._executor.submit(self._write_operation, op, op_type, step_category)

    def save_agent_async(self, agent, history_id) -> None:
        self._executor.submit(self._write_agent, agent, history_id)

    def save_env_async(self, env, history_id) -> None:
        self._executor.submit(self._write_env, env, history_id)

    def save_param_values_async(self, pvs: list) -> None:
        self._executor.submit(self._write_param_values, pvs)

    def save_artifact_records_async(self, artifact, artifact_state, history_id) -> None:
        self._executor.submit(self._write_artifact_records, artifact, artifact_state, history_id)

    def save_delta_async(self, delta: dict, step_id: str) -> None:
        self._executor.submit(self._write_delta, delta, step_id)

    def update_history_active_state_async(self, history_id: str, active_state_id: str) -> None:
        self._executor.submit(self._update_history_active_state, history_id, active_state_id)

    def save_pipeline_async(self, pipeline_id: str, history_id: str, name: str) -> None:
        self._executor.submit(self._write_pipeline, pipeline_id, history_id, name)

    def save_fragment_async(self, fragment_id: str, pipeline_id: str, step_ids: list, position: int) -> None:
        self._executor.submit(self._write_fragment, fragment_id, pipeline_id, step_ids, position)

    # ------------------------------------------------------------------
    # Artifact persistence (synchronous)
    # ------------------------------------------------------------------

    def save_artifact(self, node_id: str, df: Any) -> Optional[str]:
        """
        Write *df* to ``<artifact_dir>/<node_id>.parquet``.
        Returns the path string, or None if writing fails.
        """
        path = self.artifact_dir / f"{node_id}.parquet"
        try:
            module = type(df).__module__ or ""
            if "polars" in module:
                df.write_parquet(str(path))
            else:
                df.to_parquet(str(path))
            return str(path)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def load_graph(self, history_id: Optional[str] = None) -> dict:
        """Return states and steps as plain Python dicts."""
        where = "WHERE history_id = ?" if history_id else ""
        params = [history_id] if history_id else []
        con = self._connect(read_only=True)
        try:
            states = con.execute(
                f"SELECT state_id, history_id, branch_id, produced_by_step_id, derived_from_state_id, timestamp FROM analysis_states {where}",
                params).fetchall()
            steps = con.execute(
                f"SELECT step_id, history_id, input_state_id, output_state_id, func_name, raw_line, timestamp FROM analysis_steps {where}",
                params).fetchall()
        finally:
            con.close()
        return {
            "states": [{"state_id": r[0], "history_id": r[1], "branch_id": r[2],
                        "produced_by_step_id": r[3], "derived_from_state_id": r[4], "timestamp": r[5]}
                       for r in states],
            "steps": [{"step_id": r[0], "history_id": r[1], "input_state_id": r[2],
                       "output_state_id": r[3], "func_name": r[4], "raw_line": r[5], "timestamp": r[6]}
                      for r in steps],
        }

    def load_state_detail(self, state_id: str) -> dict:
        """
        Return full metadata for a state: step, operation type, agent, environment,
        parameter values, and delta. Returns {} when state_id is not found.
        """
        con = self._connect(read_only=True)
        try:
            row = con.execute("""
                SELECT s.step_id, s.func_name, s.raw_line, s.timestamp,
                       s.param_fingerprint,
                       o.name AS op_name, ot.name AS op_type_name,
                       COALESCE(sc.name, '') AS category_name,
                       ag.agent_type, ag.username,
                       e.tool_version, e.library_versions, e.runtime,
                       b.name AS branch_name
                FROM analysis_states st
                JOIN analysis_steps s ON s.output_state_id = st.state_id
                JOIN operations o ON o.operation_id = s.operation_id
                JOIN operation_types ot ON ot.type_id = o.operation_type_id
                LEFT JOIN step_categories sc ON sc.category_id = o.step_category_id
                JOIN agents ag ON ag.agent_id = s.agent_id
                JOIN runtime_environments e ON e.env_id = s.env_id
                JOIN analysis_branches b ON b.branch_id = st.branch_id
                WHERE st.state_id = ?
            """, _p(state_id)).fetchone()

            if row is None:
                return {}

            (step_id, func_name, raw_line, timestamp, param_fingerprint,
             op_name, op_type_name, category_name, agent_type, username,
             tool_version, library_versions_json, runtime, branch_name) = row

            pvs = con.execute(
                "SELECT param_id, value_type, value_json FROM parameter_values WHERE step_id = ?",
                _p(step_id),
            ).fetchall()

            delta_row = con.execute(
                """SELECT kind, mutation_type, modification_type, rows_delta,
                          columns_added, columns_removed, dtype_changes
                   FROM deltas WHERE step_id = ?""",
                _p(step_id),
            ).fetchone()

        finally:
            con.close()

        params = [
            {"param_id": r[0], "value_type": r[1], "value": json.loads(r[2])}
            for r in pvs
        ]

        delta = None
        if delta_row:
            delta = {
                "kind": delta_row[0],
                "mutation_type": delta_row[1],
                "modification_type": delta_row[2],
                "rows_delta": delta_row[3],
                "columns_added": json.loads(delta_row[4] or "[]"),
                "columns_removed": json.loads(delta_row[5] or "[]"),
                "dtype_changes": json.loads(delta_row[6] or "{}"),
            }

        return {
            "state_id": state_id,
            "step_id": step_id,
            "timestamp": timestamp,
            "branch_name": branch_name,
            "func_name": func_name,
            "raw_line": raw_line,
            "param_fingerprint": param_fingerprint,
            "operation": {"name": op_name, "type": op_type_name, "category": category_name or None},
            "agent": {"agent_type": agent_type, "username": username},
            "environment": {
                "tool_version": tool_version,
                "library_versions": json.loads(library_versions_json or "{}"),
                "runtime": runtime,
            },
            "params": params,
            "delta": delta,
        }

    def load_branches(self, history_id: str) -> list[dict]:
        """
        Return all branches for a history with step counts and divergence points.
        divergence_point_id is None for the root branch (starts_at has no producing step).
        """
        con = self._connect(read_only=True)
        try:
            rows = con.execute("""
                SELECT b.branch_id, b.name, b.starts_at_state_id,
                       COUNT(DISTINCT s.step_id) AS step_count,
                       st.produced_by_step_id
                FROM analysis_branches b
                LEFT JOIN analysis_states ast ON ast.branch_id = b.branch_id
                LEFT JOIN analysis_steps s ON s.output_state_id = ast.state_id
                LEFT JOIN analysis_states st ON st.state_id = b.starts_at_state_id
                WHERE b.history_id = ?
                GROUP BY b.branch_id, b.name, b.starts_at_state_id, st.produced_by_step_id
            """, _p(history_id)).fetchall()
        finally:
            con.close()

        results = []
        for branch_id, name, starts_at, step_count, produced_by in rows:
            divergence = starts_at if (produced_by and produced_by.strip()) else None
            results.append({
                "branch_id": branch_id,
                "name": name,
                "starts_at_state_id": starts_at,
                "step_count": step_count,
                "divergence_point_id": divergence,
            })
        return results

    def load_operations_by_category(self, history_id: str) -> list[dict]:
        """
        Return all steps for a history with their StepCategory names.
        Steps with no registered category appear with category=None.
        """
        con = self._connect(read_only=True)
        try:
            rows = con.execute("""
                SELECT s.func_name, ot.name AS op_type, sc.name AS category
                FROM analysis_steps s
                JOIN operations o ON o.operation_id = s.operation_id
                JOIN operation_types ot ON ot.type_id = o.operation_type_id
                LEFT JOIN step_categories sc ON sc.category_id = o.step_category_id
                WHERE s.history_id = ?
            """, _p(history_id)).fetchall()
        finally:
            con.close()
        return [{"func_name": r[0], "op_type": r[1], "category": r[2]} for r in rows]

    def load_artifact_lifecycle(self, state_id: str) -> list[dict]:
        """
        Return the ordered evolution chain of the artifact associated with state_id.
        Returns [] when no artifact is recorded for state_id.
        """
        con = self._connect(read_only=True)
        try:
            art_row = con.execute(
                "SELECT artifact_id FROM artifact_states WHERE analysis_state_id = ?",
                _p(state_id),
            ).fetchone()
            if art_row is None:
                return []
            artifact_id = art_row[0]

            rows = con.execute("""
                SELECT ast.analysis_state_id, st.timestamp, s.func_name, s.raw_line,
                       ast.content_ref, ast.size_bytes, ast.checksum,
                       d.modification_type, d.rows_delta,
                       d.columns_added, d.columns_removed
                FROM artifact_states ast
                JOIN analysis_states st ON st.state_id = ast.analysis_state_id
                LEFT JOIN analysis_steps s ON s.output_state_id = ast.analysis_state_id
                LEFT JOIN deltas d ON d.step_id = s.step_id
                WHERE ast.artifact_id = ?
                ORDER BY st.timestamp
            """, _p(artifact_id)).fetchall()
        finally:
            con.close()

        return [
            {
                "state_id": r[0],
                "timestamp": r[1],
                "func_name": r[2],
                "raw_line": r[3],
                "content_ref": r[4],
                "size_bytes": r[5],
                "checksum": r[6],
                "modification_type": r[7],
                "rows_delta": r[8],
                "columns_added": json.loads(r[9] or "[]"),
                "columns_removed": json.loads(r[10] or "[]"),
            }
            for r in rows
        ]

    def load_artifact_path(self, artifact_state_id: str) -> Optional[str]:
        """Return the Parquet content_ref for an artifact_state_id, or None."""
        con = self._connect(read_only=True)
        try:
            row = con.execute(
                "SELECT content_ref FROM artifact_states WHERE artifact_state_id = ?",
                _p(artifact_state_id),
            ).fetchone()
        finally:
            con.close()
        return row[0] if row else None

    def load_pipeline_steps(self, pipeline_id: str) -> list[dict]:
        """Return step records for a pipeline in fragment order."""
        con = self._connect(read_only=True)
        try:
            fragments = con.execute(
                "SELECT step_ids, position FROM pipeline_fragments WHERE pipeline_id = ? ORDER BY position",
                _p(pipeline_id),
            ).fetchall()

            ordered_step_ids: list[str] = []
            for frag in fragments:
                ordered_step_ids.extend(json.loads(frag[0]))

            if not ordered_step_ids:
                return []

            placeholders = ",".join(["?"] * len(ordered_step_ids))
            rows = con.execute(
                f"SELECT step_id, func_name, raw_line FROM analysis_steps WHERE step_id IN ({placeholders})",
                ordered_step_ids,
            ).fetchall()
        finally:
            con.close()

        step_map = {r[0]: {"step_id": r[0], "func_name": r[1], "raw_line": r[2]} for r in rows}
        return [step_map[sid] for sid in ordered_step_ids if sid in step_map]

    def to_networkx(self, history_id: Optional[str] = None):
        """Build an in-memory ``networkx.DiGraph`` from the stored provenance DAG."""
        import networkx as nx  # optional dependency

        data = self.load_graph(history_id=history_id)
        G = nx.DiGraph()
        for s in data["states"]:
            G.add_node(s["state_id"], **{k: v for k, v in s.items() if k != "state_id"})
        for st in data["steps"]:
            G.add_edge(st["input_state_id"], st["output_state_id"],
                       func_name=st["func_name"], step_id=st["step_id"])
        return G

    # ------------------------------------------------------------------
    # Private write helpers (run inside the worker thread)
    # ------------------------------------------------------------------

    def _write_history(self, history) -> None:
        con = self._connect()
        try:
            con.execute("INSERT OR REPLACE INTO analysis_histories VALUES (?,?,?,?)",
                        _p(history.history_id, history.name or "", _now(), history.active_state_id or ""))
            _commit(con)
        finally:
            con.close()

    def _write_branch(self, branch) -> None:
        con = self._connect()
        try:
            con.execute("INSERT OR REPLACE INTO analysis_branches VALUES (?,?,?,?)",
                        _p(branch.branch_id, branch.history_id, branch.name, branch.starts_at_state_id))
            _commit(con)
        finally:
            con.close()

    def _write_state(self, state) -> None:
        con = self._connect()
        try:
            con.execute("INSERT OR REPLACE INTO analysis_states VALUES (?,?,?,?,?,?)",
                        _p(state.state_id, state.history_id, state.branch_id,
                           state.produced_by_step_id or "", state.derived_from_state_id or "", _now()))
            _commit(con)
        finally:
            con.close()

    def _write_step(self, step, func_name, raw_line, param_fingerprint, history_id) -> None:
        con = self._connect()
        try:
            con.execute(
                "INSERT OR REPLACE INTO analysis_steps VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                _p(step.step_id, history_id, step.input_state_id, step.output_state_id,
                   step.agent_id, step.env_id, step.operation_id,
                   func_name, raw_line or "", param_fingerprint or "", _now()))
            _commit(con)
        finally:
            con.close()

    def _write_operation(self, op, op_type, step_category=None) -> None:
        con = self._connect()
        try:
            if step_category is not None:
                con.execute("INSERT OR REPLACE INTO step_categories VALUES (?,?)",
                            _p(step_category.category_id, step_category.name))
            con.execute("INSERT OR REPLACE INTO operation_types VALUES (?,?)", _p(op_type.type_id, op_type.name))
            con.execute("INSERT OR REPLACE INTO operations (operation_id, name, operation_type_id, step_category_id) VALUES (?,?,?,?)",
                        _p(op.operation_id, op.name, op.operation_type_id, getattr(op, "step_category_id", None)))
            _commit(con)
        finally:
            con.close()

    def _write_agent(self, agent, history_id) -> None:
        con = self._connect()
        try:
            con.execute("INSERT OR REPLACE INTO agents VALUES (?,?,?,?)",
                        _p(agent.agent_id, history_id, agent.agent_type.value, getattr(agent, "username", "")))
            _commit(con)
        finally:
            con.close()

    def _write_env(self, env, history_id) -> None:
        con = self._connect()
        try:
            con.execute("INSERT OR REPLACE INTO runtime_environments VALUES (?,?,?,?,?)",
                        _p(env.env_id, history_id, env.tool_version,
                           json.dumps(env.library_versions), env.runtime or ""))
            _commit(con)
        finally:
            con.close()

    def _write_param_values(self, pvs: list) -> None:
        con = self._connect()
        try:
            for pv in pvs:
                con.execute("INSERT OR REPLACE INTO parameter_values VALUES (?,?,?,?,?)",
                            _p(pv["parameter_value_id"], pv["step_id"],
                               pv.get("parameter_id", ""), pv["value_type"],
                               json.dumps(pv, default=str)))
            _commit(con)
        finally:
            con.close()

    def _write_artifact_records(self, artifact, artifact_state, history_id) -> None:
        con = self._connect()
        try:
            if artifact is not None:
                con.execute("INSERT OR REPLACE INTO artifacts VALUES (?,?,?,?)",
                            _p(artifact.artifact_id, history_id, artifact.name, artifact.artifact_type.value))
            con.execute("INSERT OR REPLACE INTO artifact_states VALUES (?,?,?,?,?,?,?)",
                        _p(artifact_state.artifact_state_id, artifact_state.artifact_id,
                           artifact_state.analysis_state_id, artifact_state.mime_type,
                           artifact_state.checksum, artifact_state.content_ref, artifact_state.size_bytes))
            _commit(con)
        finally:
            con.close()

    def _write_delta(self, delta: dict, step_id: str) -> None:
        con = self._connect()
        try:
            con.execute("INSERT OR REPLACE INTO deltas VALUES (?,?,?,?,?,?,?,?,?)",
                        _p(str(uuid.uuid4()), step_id,
                           delta.get("kind", "unknown"), delta.get("mutation_type", ""),
                           delta.get("modification_type", ""), delta.get("rows_delta"),
                           json.dumps(delta.get("columns_added", [])),
                           json.dumps(delta.get("columns_removed", [])),
                           json.dumps(delta.get("dtype_changes", {}))))
            _commit(con)
        finally:
            con.close()

    def _update_history_active_state(self, history_id: str, active_state_id: str) -> None:
        con = self._connect()
        try:
            con.execute("UPDATE analysis_histories SET active_state_id=? WHERE history_id=?",
                        _p(active_state_id, history_id))
            _commit(con)
        finally:
            con.close()

    def _write_pipeline(self, pipeline_id, history_id, name) -> None:
        con = self._connect()
        try:
            con.execute("INSERT OR REPLACE INTO pipelines VALUES (?,?,?,?)",
                        _p(pipeline_id, history_id, name, _now()))
            _commit(con)
        finally:
            con.close()

    def _write_fragment(self, fragment_id, pipeline_id, step_ids, position) -> None:
        con = self._connect()
        try:
            con.execute("INSERT OR REPLACE INTO pipeline_fragments VALUES (?,?,?,?)",
                        _p(fragment_id, pipeline_id, json.dumps(step_ids), position))
            _commit(con)
        finally:
            con.close()
