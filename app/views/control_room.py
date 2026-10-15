"""0 · Control room: jobs, recent runs, family scoreboard, cache health."""

import time

import pandas as pd
import streamlit as st

import studio_ui as ui
from afml.studio import jobs
from afml.studio.worker import spawn_detached

st.title("Control room")
led = ui.ledger()

# --------------------------------------------------------------------------- worker


@st.fragment(run_every="3s")
def worker_panel():
    live = led.live_workers()
    started = st.session_state.get("worker_started")
    starting = (not live and started is not None and jobs.is_alive(started[0])
                and time.time() - started[1] < 60)
    c1, c2 = st.columns([3, 1])
    with c1:
        if live:
            w = live[0]
            st.success(f"Worker pid {w['pid']} on {w['host']} is {w['status']} "
                       f"(heartbeat {w['age_s']:.0f}s ago)")
        elif starting:
            st.info(f"Worker pid {started[0]} is starting (importing afml takes about 10 s)…")
        else:
            st.warning("No worker is running. Queued jobs wait until one starts.")
    with c2:
        if live:
            if st.button("Stop worker", use_container_width=True):
                for w in live:
                    jobs.kill_tree(w["pid"])
                    led.heartbeat(w["pid"], w["host"], "stopped")
                for r in led.list_runs(statuses=["running"]):
                    led.set_status(r["run_id"], "failed", error="worker stopped from the Control room")
                st.rerun()
        elif st.button("Start worker", type="primary", use_container_width=True, disabled=starting):
            st.session_state["worker_started"] = (spawn_detached(), time.time())
            st.rerun()


worker_panel()


# --------------------------------------------------------------------------- active jobs


@st.fragment(run_every="2s")
def active_jobs():
    st.subheader("Running and queued")
    active = led.list_runs(statuses=["running", "queued"], limit=50)
    if not active:
        st.caption("Nothing running.")
        return
    for run in reversed(active):
        st_ = jobs.status(run["run_id"])
        p = st_["progress"]
        with st.container(border=True):
            top = st.columns([4, 1])
            top[0].markdown(f"**{run['stage']}** · `{run['run_id']}`  \n{run['family_name']}")
            if top[1].button("Cancel", key=f"cancel.{run['run_id']}", use_container_width=True):
                jobs.cancel(run["run_id"])
                st.rerun()
            if run["status"] == "queued":
                st.caption("queued")
            else:
                st.progress(float(p["progress"] or 0.0),
                            text=f"{p['stage'] or 'starting'} · {p['message'] or ''}")
            if st_.get("stale"):
                st.warning("The process is gone but the run is still marked running.")


active_jobs()

# --------------------------------------------------------------------------- scoreboard

st.subheader("Family scoreboard")
fams = led.list_families()
if not fams:
    st.caption("No families yet. A family is one research question: symbol, bars, labels, strategy.")
else:
    rows = []
    for f in fams:
        s = led.family_stats(f["family_id"])
        # Phone-first column order: what you need to judge a score comes before the detail
        rows.append({
            "family": f["name"],
            "flags": " ".join(x for x in [
                "CONTAMINATED" if f["contaminated"] else "",
                "rank-limited" if s["rank_limited"] else "",
                "frozen" if f["frozen_config_hash"] else "",
            ] if x),
            "best score": s["best_score"],
            "N": s["n_raw"],
            "N eff (eigen)": s["n_eigen"],
            "null threshold": s["null_threshold"],
            "metric": s["metric"] or "",
            "± noise floor": s["best_noise_floor"],
            "N eff (Kish)": s["n_kish"],
            "id": f["family_id"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={k: st.column_config.NumberColumn(format="%.4f")
                                for k in ["best score", "± noise floor", "null threshold"]}
                 | {k: st.column_config.NumberColumn(format="%.1f") for k in ["N eff (eigen)", "N eff (Kish)"]})
    if any(f["contaminated"] for f in fams):
        st.error("At least one family is contaminated: its hold-out was accessed more than once.")

# --------------------------------------------------------------------------- recent runs

st.subheader("Recent runs")
runs = led.list_runs(limit=40)
if not runs:
    st.caption("No runs yet. Start on page 1 (Data and bars).")
else:
    def score_of(r):
        s = r.get("summary") or {}
        if "importance" in s:
            return s["importance"].get("cv_score_mean")
        if "model" in s:
            return s["model"].get("best_score")
        return None

    def cache_of(r):
        c = (r.get("summary") or {}).get("cache") or {}
        hits = sum(v["hits"] for v in c.values())
        total = hits + sum(v["misses"] for v in c.values())
        return f"{hits}/{total}" if total else ""

    df = pd.DataFrame([{
        "run": r["run_id"], "stage": r["stage"], "status": r["status"], "purpose": r["purpose"],
        "family": r["family_name"], "score": score_of(r), "cache hits": cache_of(r),
        "created": r["created_at"][:19].replace("T", " "), "error": r["error"] or "",
    } for r in runs])
    st.dataframe(df, hide_index=True, use_container_width=True,
                 column_config={"score": st.column_config.NumberColumn(format="%.4f")})

    sel = st.selectbox("Run", [r["run_id"] for r in runs], key="cr.sel")
    run = next(r for r in runs if r["run_id"] == sel)
    a, b, c = st.columns(3)
    if a.button("Clone config into draft", use_container_width=True):
        ui.load_into_draft(jobs.clone_spec(sel))
        st.toast("Draft replaced with this run's config")
    if b.button("Resume as new run", use_container_width=True,
                disabled=run["status"] not in ("failed", "cancelled")):
        st.toast(f"Submitted {jobs.resume(sel)}")
        st.rerun()
    if c.button("Cancel", use_container_width=True, disabled=run["status"] not in ("queued", "running")):
        jobs.cancel(sel)
        st.rerun()
    with st.expander("Log"):
        ui.show_log(run)
    with st.expander("Run folder"):
        st.code(run["run_dir"])
        st.dataframe(pd.DataFrame(led.artifacts(sel)), hide_index=True, use_container_width=True)

# --------------------------------------------------------------------------- cache

with st.expander("Cache health"):
    hits = misses = 0
    for r in led.list_runs(statuses=["done"], limit=200):
        for v in ((r.get("summary") or {}).get("cache") or {}).values():
            hits += v["hits"]
            misses += v["misses"]
    st.metric("afml.cache hit rate in studio runs (last 200)",
              f"{hits / (hits + misses):.0%}" if hits + misses else "—")
    try:
        from afml import cache_status

        st.caption(f"afml.cache_status(): {cache_status()}")
    except Exception as exc:
        st.caption(f"cache_status unavailable: {exc}")
