# MetaTrader 5 Machine Learning Blueprint – Modules Repository

This repository contains the companion code modules, utilities, and assets for the **MetaTrader 5 Machine Learning Blueprint** article series by *Patrick Murimi Njoroge*.  
It is designed to be a clean, production‑ready implementation of advanced financial machine learning techniques — from robust data handling to adaptive, probabilistic trade execution.

---

## 📚 Series Context

This repo accompanies the following articles:

1. **Part 1 – Data Integrity & Tick‑Based Bars**  
   - Eliminating data leakage with proper tick aggregation  
   - Timestamp correction and unbiased dataset preparation  

2. **Part 2 – Meta‑Labeling & Triple‑Barrier Method**  
   - Risk‑aware labeling with profit‑taking/stop‑loss logic  
   - Meta‑labels to improve classifier precision under realistic trading constraints  

3. **Part 3 – Advanced Labeling & Sample Weighting**  
   - Trend‑scanning labels with adaptive horizons  
   - Purged cross‑validation  
   - Sample weighting to address concurrency bias  
   - Probabilistic position sizing for real‑time execution

---

## 🔑 Key Features

- **Leakage‑Proof Labeling** – Triple‑barrier & adaptive trend‑scanning with volatility regime filtering

- **Numba‑Accelerated** – 100×–350× faster execution for live‑trading scenarios

- **Concurrency‑Aware Weighting** – Down‑weights overlapping observations for better generalization

- **Probabilistic Position Sizing** – Trade sizing aligned with model confidence and risk parameters

- **MT5 Integration** – Direct pipeline from Python model output to MetaTrader 5 execution
