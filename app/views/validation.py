"""6 · Validation (Phase 3): the battery that can only reject."""

import json
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import studio_ui as ui
from afml.studio import calibration, jobs
from afml.studio import validation as SV

st.title("6 · Validation")
st.caption("Validation can only reject: a backtest exists to discard models, not to improve them (AFML 11.4). "
           "Nothing on this page changes the model, and every battery run is logged as a trial.")
led = ui.ledger()

# --------------------------------------------------------------------------- calibration

cal = calibration.latest()
with st.expander("Battery calibration" + (" (not run yet)" if not cal else ""), expanded=False):
    if not cal:
        st.info("Not calibrated yet. Run `python -m afml.studio calibrate` to measure the battery's false-pass rate "
                "on synthetic noise families and its power on planted ones.")
    else:
        noise, signal = cal["kinds"]["noise"], cal["kinds"]["signal"]
        m = st.columns(3)
        m[0].metric("False-pass rate on noise", f"{cal['false_pass_rate']:.1%}" if cal["false_pass_rate"] is not None
                    else "—", help=f"{noise['n_families']} noise families; 95% CI "
                    f"{noise['ci'][0]:.0%} to {noise['ci'][1]:.0%}")
        m[1].metric("Registered level", f"{cal['alpha']:.0%}")
        m[2].metric("Power on planted signal", f"{cal['power']:.0%}" if cal["power"] is not None else "—",
                    help=f"{signal['n_families']} families, AR(1) strength {cal['signal_strength']}; 95% CI "
                    f"{signal['ci'][0]:.0%} to {signal['ci'][1]:.0%}")
        if cal["false_pass_ok"]:
            st.success("On noise the battery passes no more often than the registered level.")
        else:
            st.error("On noise the battery passes more often than the registered level: do not trust a pass.")
        rows = []
        for kind in ("noise", "signal"):
            for test, rate in cal["kinds"][kind]["tests"].items():
                rows.append({"test": int(test), "families": kind,
                             "pass rate": f"{rate['pass_rate']:.0%} ({rate['ci'][0]:.0%} to {rate['ci'][1]:.0%})"
                             if rate["pass_rate"] is not None else "—"})
        if rows:
            tab = pd.DataFrame(rows).pivot(index="test", columns="families", values="pass rate")
            tab.columns = [f"{c} families (95% CI)" for c in tab.columns]
            st.dataframe(tab, use_container_width=True)
        st.caption(f"Calibrated {cal['created_at']} on {cal['months']} months of synthetic M15 bars per family.")




# --------------------------------------------------------------------------- choose a frozen model run

models = [r for r in led.list_runs(stage="model", limit=200)
          if r["status"] == "done" and (r.get("summary") or {}).get("model", {}).get("search") == "optuna"]
if not models:
    st.info("No completed Optuna model run yet. Run one on page 5; validation starts from its frozen config.")
    st.stop()
by_id = {r["run_id"]: r for r in models}
frozen_first = sorted(models, key=lambda r: (led.get_family(r["family_id"])["frozen_config_hash"]
                                             != r["config_hash"], r["run_id"]))
source_id = st.selectbox("Model run", [r["run_id"] for r in frozen_first], key="val.source",
                         format_func=lambda i: f"{i} · {by_id[i]['family_name']}")
source = by_id[source_id]
fam = led.get_family(source["family_id"])
if fam["contaminated"]:
    st.error("This family is contaminated: its hold-out was accessed more than once.")
if fam["closed"]:
    st.warning("This family is closed after a failed battery. Its reports stay below.")

problems = SV.entry_gate(source_id, led)
with st.container(border=True):
    st.markdown("**Entry gate**")
    if problems:
        for p in problems:
            st.markdown(f":red[✗] {p}")
    else:
        st.markdown(":green[✓] Frozen, pre-registered, per-fold scores logged, data inside the research window.")

# --------------------------------------------------------------------------- pre-registration

regs = SV.preregistrations(source["family_id"], led)
with st.expander("Pre-registration" + (f" (version {regs[-1]['version']})" if regs else ""), expanded=not regs):
    st.caption("Freezes this run's config and records every threshold, the significance level and the hold-out "
               "decision rule before the first battery run. Results are judged only against it. Registering "
               "again creates a new version; earlier versions stay listed.")
    base = SV.Preregistration(source_run=source_id)
    cur = regs[-1]["payload"] if regs else base.model_dump(mode="json")
    th, ho, acct = cur["thresholds"], cur["holdout"], cur["account"]
    with st.form("prereg"):
        c1, c2, c3 = st.columns(3)
        dsr_min = c1.number_input("DSR at least", 0.5, 0.999, float(th["dsr_min"]), format="%.3f")
        p_max = c2.number_input("Permutation p at most (tests 1, 7)", 0.001, 0.5, float(th["permutation_p_max"]),
                                format="%.3f")
        pbo_max = c3.number_input("PBO at most", 0.01, 0.99, float(th["pbo_max"]))
        c1, c2, c3 = st.columns(3)
        med_min = c1.number_input("CPCV median Sharpe above", -5.0, 5.0, float(th["cpcv_median_sharpe_min"]))
        pos_min = c2.number_input("CPCV share of paths positive", 0.0, 1.0, float(th["cpcv_positive_share_min"]))
        breach_max = c3.number_input("Breach probability at most", 0.001, 0.5, float(th["breach_probability_max"]),
                                     format="%.3f")
        from afml.prop_firms import PRESETS

        c1, c2, c3 = st.columns(3)
        rule_set = c1.selectbox("Prop-firm rules (tests 2 and 6)", sorted(PRESETS),
                                index=sorted(PRESETS).index(cur["rule_set"]))
        lot = c2.number_input("Lots at full signal", 0.01, 100.0, float(acct["lot_size"]))
        balance = c3.number_input("Account balance", 1_000.0, 10_000_000.0, float(acct["initial_balance"]), step=1000.0)
        c1, c2, c3 = st.columns(3)
        psr_min = c1.number_input("Hold-out PSR at least", 0.5, 0.999, float(ho["psr_min"]), format="%.3f")
        ho_end = c2.date_input("Hold-out end", value=date.fromisoformat(ho["holdout_end"]) if ho["holdout_end"]
                               else date.today())
        n_perm = c3.number_input("Process permutations (test 7)", 5, 1000, int(cur["n_perm_process"]))
        submitted = st.form_submit_button("Freeze and pre-register", type="primary")
    if submitted:
        try:
            reg_id = SV.preregister(SV.Preregistration(
                source_run=source_id, rule_set=rule_set, n_perm_process=int(n_perm),
                thresholds=SV.Thresholds(dsr_min=dsr_min, permutation_p_max=p_max, pbo_max=pbo_max,
                                         cpcv_median_sharpe_min=med_min, cpcv_positive_share_min=pos_min,
                                         breach_probability_max=breach_max),
                holdout=SV.HoldoutRule(psr_min=psr_min, holdout_end=ho_end),
                account={**acct, "lot_size": lot, "initial_balance": balance}), led)
            st.success(f"Pre-registration {reg_id} recorded")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if regs:
        st.dataframe(pd.DataFrame([{"version": r["version"], "id": r["id"], "config": r["config_hash"][:12],
                                    "created": r["created_at"][:19], "rule set": r["payload"]["rule_set"],
                                    "notes": r["payload"].get("notes", "")} for r in regs]),
                     hide_index=True, use_container_width=True)

# --------------------------------------------------------------------------- actions

c1, c2 = st.columns(2)
if c1.button("Run the battery (tests 1–6)", type="primary", disabled=bool(problems), use_container_width=True):
    try:
        st.toast(f"Queued {SV.submit_validation(source_id, led)}")
    except Exception as exc:
        st.error(str(exc))
if c2.button("Run process permutation (test 7, overnight)", disabled=bool(problems), use_container_width=True):
    try:
        st.toast(f"Queued {SV.submit_process_permutation(source_id, led)}")
    except Exception as exc:
        st.error(str(exc))
if not led.live_workers():
    st.caption(":orange[No worker running: queued jobs wait until one starts in the Control room.]")

# --------------------------------------------------------------------------- report

st.divider()
st.subheader("Report")
val_runs = [r for r in led.list_runs(family_id=source["family_id"], stage="validation", limit=100)]
if not val_runs:
    st.info("No battery run for this family yet.")
else:
    vrun_id = st.selectbox("Battery run", [r["run_id"] for r in val_runs], key="val.run",
                           format_func=lambda i: f"{i} · {next(r['status'] for r in val_runs if r['run_id'] == i)}")
    vrun = jobs.status(vrun_id)
    if vrun["status"] in ("queued", "running"):
        ui._live_status(vrun_id)
    elif vrun["status"] != "done":
        st.error(f"Run {vrun['status']}: {vrun.get('error') or ''}")
    else:
        bv = SV.battery_verdict(vrun_id, led)
        v = bv["summary"]
        verdict = bv["verdict"].copy()
        # Verdict panel: one row per test, readable at phone width
        if bv["passed"]:
            st.success("Every test passed.")
        elif not bv["complete"]:
            st.warning("Not every test has run yet (test 7 is its own overnight job). "
                       + ("Failures so far: " + ", ".join(verdict.loc[~verdict['passed'].astype(bool), 'name'])
                          if (~verdict["passed"].astype(bool)).any() else ""))
        else:
            st.error("Rejected: " + ", ".join(verdict.loc[~verdict["passed"].astype(bool), "name"]))
        show = verdict.assign(result=np.where(verdict["passed"].astype(bool), "pass", "FAIL"))
        st.dataframe(show[["name", "value", "threshold", "result", "note"]], use_container_width=True,
                     column_config={"value": st.column_config.NumberColumn(format="%.4f")})
        st.caption(f"Pre-registration version {v['preregistration_version']} · frozen trial {v['frozen_trial']} · "
                   f"{v['n_events']:,} events · rules {v['rule_set']}")

        tests = v["tests"]

        def t(n):
            return tests.get(str(n)) or tests.get(n) or {}

        def vline(fig, x, label, color="crimson"):
            fig.add_vline(x=x, line_color=color, annotation_text=label, annotation_position="top")

        if t(1) and "error" not in t(1):
            with st.expander("1 · Signal permutation", expanded=True):
                null = ui.table(vrun, "t1_null")
                fig = px.histogram(null, x="sharpe", nbins=40, title="Null: shifted probabilities")
                vline(fig, t(1)["observed_sharpe"], "observed")
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"p = {t(1)['p_value']:.4f} over {t(1)['n_perm']} circular shifts")
        if t(2) and "error" not in t(2):
            with st.expander("2 · CPCV path distribution", expanded=True):
                paths = ui.table(vrun, "t2_paths")
                c1, c2 = st.columns(2)
                c1.plotly_chart(px.histogram(paths, x="sharpe", nbins=20, title="Path Sharpe ratios"),
                                use_container_width=True)
                eq = ui.table(vrun, "t2_equity")
                med = paths["sharpe"].sort_values().index[len(paths) // 2]
                fig = go.Figure()
                for pid, g in eq.groupby("path"):
                    fig.add_scatter(x=g["time"], y=g["equity"], mode="lines", name=f"path {pid}",
                                    line=dict(width=3 if pid == med else 1),
                                    opacity=1.0 if pid == med else 0.35)
                fig.update_layout(title="Normalized equity (bold: median path)", showlegend=False)
                c2.plotly_chart(fig, use_container_width=True)
                st.dataframe(paths, use_container_width=True)
        if t(3) and "error" not in t(3):
            with st.expander("3 · Deflated Sharpe", expanded=True):
                curve = ui.table(vrun, "t3_expected_max")
                fig = px.line(curve, x="n_trials", y="expected_max_sharpe",
                              title="Expected maximum Sharpe under no skill, by number of trials")
                fig.add_hline(y=t(3)["observed_sharpe"], line_color="crimson", annotation_text="observed")
                vline(fig, t(3)["n_eff"], "N eff", "seagreen")
                vline(fig, t(3)["n_raw"], "N raw", "gray")
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"DSR {t(3)['dsr']:.4f} with N eff {t(3)['n_eff']:.1f} of {t(3)['n_raw']} trials; "
                           f"skew {t(3)['skewness']:.2f}, excess kurtosis {t(3)['excess_kurtosis']:.2f}")
        if t(4) and "error" not in t(4):
            with st.expander("4 · PBO", expanded=True):
                c1, c2 = st.columns(2)
                c1.plotly_chart(px.histogram(ui.table(vrun, "t4_logits"), x="logit", nbins=30,
                                             title="Logit of the IS winner's OOS rank"), use_container_width=True)
                sc = ui.table(vrun, "t4_scatter")
                c2.plotly_chart(px.scatter(sc, x="is_sharpe", y="oos_sharpe",
                                           title="IS winner: in-sample vs out-of-sample Sharpe"),
                                use_container_width=True)
                st.caption(f"PBO {t(4)['pbo']:.3f} across {t(4)['n_trials']} trials, {t(4)['n_splits']} CSCV splits")
        if t(5) and "error" not in t(5):
            with st.expander("5 · Walk-forward degradation", expanded=True):
                w = ui.table(vrun, "t5_windows").reset_index()
                st.plotly_chart(px.line(w, x="window", y=["is_score", "oos_score"], markers=True,
                                        title=f"In-sample and out-of-sample {t(5)['metric']} per window"),
                                use_container_width=True)
                st.caption(f"Mean degradation {t(5)['mean_degradation']:.4f}, 90% interval "
                           f"[{t(5)['ci_low']:.4f}, {t(5)['ci_high']:.4f}] over {t(5)['n_windows']} windows")
        if t(6) and "error" not in t(6):
            with st.expander("6 · Sequence risk", expanded=True):
                dd = ui.table(vrun, "t6_drawdowns")
                fig = px.histogram(dd, x="max_drawdown", nbins=40, title="Bootstrap maximum drawdown")
                vline(fig, t(6)["realized_max_drawdown"], "realized")
                vline(fig, t(6)["max_drawdown_limit"], "account limit", "black")
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Breach probability {t(6)['p_any_breach']:.4f} (drawdown {t(6)['p_drawdown_breach']:.4f}"
                           + (f", daily {t(6)['p_daily_breach']:.4f}" if "p_daily_breach" in t(6) else "")
                           + f"); block {t(6)['mean_block']:.0f} events, {t(6)['trades_per_day']} trades per day")
        for n, s in tests.items():
            if "error" in s:
                st.error(f"Test {n} did not run: {s['error']}")
        if bv["process_permutation_run"]:
            prun = led.get_run(bv["process_permutation_run"])
            p7 = prun["summary"]["process_permutation"]
            with st.expander("7 · Process permutation", expanded=True):
                null = ui.table(prun, "t7_null")
                fig = px.histogram(null, x="sharpe", nbins=30, title="Null: the whole process on permuted bars")
                vline(fig, p7["observed_sharpe"], "observed")
                vline(fig, p7["spurious_center"], "spurious center", "darkorange")
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"p = {p7['p_value']:.4f}; null median {p7['spurious_center']:.3f} "
                           f"(95% CI {p7['center_ci_low']:.3f} to {p7['center_ci_high']:.3f}); "
                           f"{p7['n_perm']} replicates at a budget of {p7['trial_budget']} trials")

        # Actions: export, register for hold-out (after a pass), close (after a fail)
        st.divider()
        report = {"validation_run": vrun_id, "summary": v, "verdict": verdict.reset_index().to_dict("records"),
                  "process_permutation_run": bv["process_permutation_run"]}
        c1, c2, c3 = st.columns(3)
        c1.download_button("Export report", json.dumps(report, indent=2, default=str),
                           file_name=f"validation-{vrun_id}.json", mime="application/json",
                           use_container_width=True)
        if c2.button("Register for hold-out", disabled=not bv["passed"], use_container_width=True):
            try:
                rec = SV.register_holdout(vrun_id, led)
                st.success(f"Registered: hold-out {rec['holdout_start']} to {rec['holdout_end']}, "
                           f"about {rec['expected_events']:.0f} events (needs {rec['min_track_record_events']:.0f})")
            except Exception as exc:
                st.error(str(exc))
        failed = bv["complete"] and not bv["passed"]
        if c3.button("Close the family", disabled=not failed or bool(fam["closed"]), use_container_width=True):
            SV.close_family(source["family_id"], "failed: " + ", ".join(
                verdict.loc[~verdict["passed"].astype(bool), "name"]), vrun_id, led)
            st.rerun()
