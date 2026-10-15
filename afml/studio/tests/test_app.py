"""Streamlit AppTest checks: every page renders against a seeded ledger without errors."""

import os
import sys
from pathlib import Path

import pytest

from afml.studio import paths

APP = paths.repo_root() / "app"
PAGES = sorted((APP / "views").glob("*.py"))

streamlit_testing = pytest.importorskip("streamlit.testing.v1")


@pytest.fixture(scope="module")
def seeded_home(tmp_path_factory):
    home = tmp_path_factory.mktemp("studio_app")
    old = os.environ.get("AFML_STUDIO_HOME")
    os.environ["AFML_STUDIO_HOME"] = str(home)
    from afml.studio import jobs, vault
    from afml.studio.ledger import get_ledger
    from afml.studio.worker import run_worker

    from .conftest import base_config

    led = get_ledger()
    vault.set_research_end("SYNTH", "2024-06-30", ledger=led)
    for stage in ("bars", "labels", "features", "importance"):
        jobs.submit(base_config(stage=stage), ledger=led)
    jobs.submit(base_config(
        stage="model",
        model={"estimator": {"name": "random_forest", "params": {"n_estimators": 10, "n_jobs": 1}},
               "param_grid": {"max_depth": [2, 4]}, "cv_splits": 3},
    ), ledger=led)
    run_worker(poll_s=0.2, once=True, ledger=led)
    failed = [r for r in led.list_runs() if r["status"] != "done"]
    assert not failed, [(r["stage"], r["error"]) for r in failed]
    if str(APP) not in sys.path:
        sys.path.insert(0, str(APP))
    yield home
    if old is None:
        os.environ.pop("AFML_STUDIO_HOME", None)
    else:
        os.environ["AFML_STUDIO_HOME"] = old


@pytest.mark.parametrize("page", PAGES, ids=[p.stem for p in PAGES])
def test_page_renders(seeded_home, page):
    at = streamlit_testing.AppTest.from_file(str(page), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_control_room_shows_scoreboard(seeded_home):
    at = streamlit_testing.AppTest.from_file(str(APP / "views" / "control_room.py"), default_timeout=120)
    at.run()
    assert not at.exception
    assert any("Family scoreboard" in h.value for h in at.subheader)
    assert len(at.dataframe) >= 2


def test_launch_without_research_end_is_refused(seeded_home):
    at = streamlit_testing.AppTest.from_file(str(APP / "views" / "data_bars.py"), default_timeout=120)
    at.run()
    at.text_input[0].set_value("NOEND").run()
    assert any("No research_end" in e.value for e in at.error)
