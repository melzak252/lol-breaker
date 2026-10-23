"""pytest: isolated PostgreSQL test DB per test."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import pytest

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://betting:betting_local_password@localhost:5432/betting_test",
)
os.environ["DATABASE_URL"] = TEST_DATABASE_URL


from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, pool, text
from sqlalchemy.orm import Session

from betting_app.core.db import dispose_engine
from betting_app.models.base import Base
import betting_app.models  # ensure all models are registered
from betting_app.api.main import app

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_artifacts(*paths): skip when declared optional local research inputs are absent",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        marker = item.get_closest_marker("requires_artifacts")
        if marker is None:
            continue
        missing = [str(path) for path in marker.args if not Path(path).exists()]
        if missing:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"missing optional research artifact(s): {', '.join(missing)}"
                )
            )


@pytest.fixture(scope="module")
def require_native_artifacts():
    """Skip only genuine native-worker cases when their optional inputs are absent."""
    import json

    project_root = Path(__file__).resolve().parents[2]
    configured = os.environ.get("C0_SERVING_CONFIG")
    config_path = Path(configured).expanduser() if configured else project_root / "conf/base/c0_serving.json"
    if not config_path.is_absolute():
        config_path = project_root / config_path
    missing = []
    if not config_path.is_file():
        missing.append(str(config_path))
    else:
        config = json.loads(config_path.read_text())
        research_root = next((os.environ[name] for name in (
            "C0_RESEARCH_ROOT", "ENSEMBLE_RESEARCH_ROOT", "RESEARCH_ROOT",
        ) if os.environ.get(name)), None)
        if research_root is None:
            pointer = project_root / "data/research_root.txt"
            if pointer.is_file():
                research_root = pointer.read_text().strip()
        research_root = Path(research_root).expanduser() if research_root else None
        if research_root is not None and not research_root.is_absolute():
            research_root = project_root / research_root
        if research_root is None:
            missing.append(str(project_root / "data/research_root.txt"))
        elif not research_root.exists():
            missing.append(str(research_root))
        else:
            release_map = Path(os.environ.get("C0_RELEASE_MAP_PATH", "")).expanduser()
            if not os.environ.get("C0_RELEASE_MAP_PATH"):
                release_map = research_root / config["release_map_relative"]
            elif not release_map.is_absolute():
                release_map = project_root / release_map
            if not release_map.is_file():
                missing.append(str(release_map))
        model_root = Path(config["model_root"]).expanduser()
        if not model_root.is_absolute():
            model_root = project_root / model_root
        from betting_app.core.models import native_c0
        required_native_files = [
            config["completion_report"],
            *native_c0._ARTIFACT_SHA256,
            str(Path(config["model_root"]) / "execution.json"),
        ]
        fresh_protocol = project_root / config["fresh_protocol_path"]
        required_native_files.append(fresh_protocol)
        if fresh_protocol.is_file():
            fresh_run = json.loads(fresh_protocol.read_text())["run_id"]
            fresh_root = (
                project_root / "data/06_models/c0_distilled_verification"
                / fresh_run / "fresh" / config["fresh_candidate"]
            )
            required_native_files.extend([
                fresh_root / "bundle_manifest.json",
                fresh_root / "calibration.json",
                *(fresh_root / f"seed_{seed}/checkpoint_epoch16.pt"
                  for seed in (20260910, 20260911, 20260912)),
            ])
        for relative in required_native_files:
            path = Path(relative).expanduser()
            if not path.is_absolute():
                path = project_root / path
            if not path.is_file():
                missing.append(str(path))
        if not model_root.exists():
            missing.append(str(model_root))
        raw_history = Path(os.environ.get("C0_RAW_HISTORY_PATH", config["serving_raw_history_path"])).expanduser()
        if not raw_history.is_absolute():
            raw_history = project_root / raw_history
        if not raw_history.is_file():
            missing.append(str(raw_history))
    if missing:
        pytest.skip(f"missing optional native artifact(s): {', '.join(missing)}")

_test_engine = None

def _get_test_engine():
    global _test_engine
    if _test_engine is None:
        _test_engine = create_engine(TEST_DATABASE_URL, poolclass=pool.NullPool)
        Base.metadata.create_all(_test_engine)
        with _test_engine.begin() as conn:
            conn.execute(text("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_odds_unique_snapshot
                ON odds_snapshots (canonical_match_id, bookmaker_id, scraped_at)
                WHERE market_type = 'match_winner'
            """))
            conn.execute(text("ALTER TABLE prop_odds_snapshots ADD COLUMN IF NOT EXISTS outcomes_payload TEXT;"))
            conn.execute(text("ALTER TABLE model_ev_signals ADD COLUMN IF NOT EXISTS evaluated_at TIMESTAMP WITH TIME ZONE;"))
            conn.execute(text("ALTER TABLE model_ev_signals ADD COLUMN IF NOT EXISTS market_context_json TEXT;"))
    return _test_engine

@pytest.fixture(scope="function")
def client() -> Generator[TestClient, None, None]:
    engine = _get_test_engine()

    # Registry services also create tables outside Base.metadata. Clear those
    # between tests and repeated suite runs rather than leaking prior evaluations.
    quote = engine.dialect.identifier_preparer.quote
    table_names = [quote(name) for name in inspect(engine).get_table_names()]
    truncate_sql = f"TRUNCATE TABLE {', '.join(table_names)} RESTART IDENTITY CASCADE;"

    with engine.begin() as conn:
        conn.execute(text(truncate_sql))
        conn.execute(
            text(
                "INSERT INTO bookmakers(id, name, base_url) VALUES "
                "(1,'manual',NULL),(2,'sts','https://www.sts.pl/'),(3,'betclic','https://www.betclic.pl/'),"
                "(4,'superbet','https://superbet.pl/'),(5,'efortuna','https://www.efortuna.pl/'),"
                "(6,'fortuna','https://www.efortuna.pl/'),(7,'betfan','https://betfan.pl/'),"
                "(8,'totalbet','https://totalbet.pl/'),(9,'lebull','https://www.lebull.pl/'),"
                "(10,'pinnacle','https://www.pinnacle.com/'),(11,'kalshi','https://kalshi.com/') "
                "ON CONFLICT (id) DO NOTHING"
            )
        )

    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    dispose_engine()

    with TestClient(app) as c:
        yield c

    dispose_engine()
