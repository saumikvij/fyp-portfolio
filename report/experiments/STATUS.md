# Model-search experiment: status (branch `experiment/model-search`)

Design and every decision: DECISIONS.md, sections MS1–MS10 and WU1.

## Done
- Rounds 1–3: 420 setups on the 51 stocks (15 families, 5 windows, 2 targets, 3 signal sets, plus averages). Development only.
- Placebo (shuffled targets, 270 setups): real results are indistinguishable from luck so far
  (best IC 0.046 real vs 0.041 placebo; 5% vs 20% pass the 3-block rule).
- Wide universe built: 1,052 stocks with full 2010–2026 data in `data/wide/processed`.

- Round 4 done: 780 setups in total on the 51 stocks; finalists kernel ridge, KNN, LightGBM (risk signal set).
- Wide universe done: 60 setups + 36 placebo (see DECISIONS MS11).

## Not done yet (in this order)
1. Weekly test (E1), planned for 18 Oct.
2. Final exam on 2022–2026, once: `python -m notebooks.model_search --stage final`.
3. Bonus 2008–2009 crisis stress test (MS10).
4. Wider-universe portfolio with Ledoit–Wolf, only if wide forecasts show anything.

Results logs: `report/experiments/search_log.csv`, `placebo_log.csv`, `wide_search_log.csv`.
The holdout (forecasts made in 2022–2026) has not been looked at for any new setup.
