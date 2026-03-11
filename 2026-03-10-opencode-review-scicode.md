# SCIENTIFIC CODE REVIEW

**Date:** 2026-03-10  
**Files reviewed:** 12 Python files (src/*.py, *-main_*.py, *.py notebooks)  
**Library stack:** numpy 2.4.1, scipy 1.17.0, pandas 2.3.3, pingouin 0.5.5, mne 1.11.0, scikit-learn 1.8.0, matplotlib, seaborn 0.13.2

---

## Executive Summary

This review identified **41 total issues** across the psilocybin microstates analysis codebase. The most critical concerns involve: (1) **statistical test selection errors** where Tukey HSD is used for within-subjects comparisons and between-subjects ANOVA is applied to within-subjects data, (2) **numerical instability** including division-by-zero risks in transition matrices and algorithmic errors in midpoint filling, (3) **missing sphericity checks** in repeated-measures ANOVA, (4) **inconsistent multiple comparison corrections** (fdr_bh vs fdr_by), and (5) **publication-readiness issues** with insufficient DPI, oversized figures, and missing colorbar labels. Reproducibility is compromised by non-propagated random seeds and parallel processing non-determinism. These issues must be addressed before data publication or journal submission.

---

## Priority Issues (must fix before data publication or paper submission)

1. **[CRITICAL] Incorrect test type for microstate comparisons:** `diff_between_microstates()` in `2.5-microstate-stats.py` uses `pg.anova(between=["microstate"])` treating microstate as between-subjects when it is within-subjects. Use `pg.rm_anova(within=["microstate"], subject="subject")` instead.

2. **[CRITICAL] Tukey HSD inappropriate for repeated measures:** Tukey's HSD assumes independent groups. Replace with `pg.pairwise_tests(within="microstate", subject="subject", padjust="bonferroni")`.

3. **[CRITICAL] Division by zero in transition matrix:** `src/recording.py:273-275` normalizes without checking for zero row sums. Add epsilon guard or use `np.divide` with `where` clause.

4. **[CRITICAL] Midpoint fill boundary error:** `src/recording.py:177-185` crashes with IndexError when ≤1 GFP peak exists. Add edge case handling.

5. **[CRITICAL] No sphericity check in RM ANOVA:** All `pg.rm_anova()` calls in `2.5-microstate-stats.py` ignore Mauchly's test results. Apply Greenhouse-Geisser or Huynh-Feldt correction when sphericity is violated.

6. **[CRITICAL] Inconsistent FDR methods:** Notebook 5 uses `fdr_by` while others use `fdr_bh`. Standardize on one method across all analyses.

7. **[HIGH] Missing multiple comparison correction:** Experience questionnaire t-tests in `3-psychedelic-experience-data.py` and `5-psychedelic-experience-data_IP.py` perform ~20 paired t-tests without correction.

8. **[HIGH] DPI insufficient for print:** `src/microstates.py:372` uses `dpi=150` which is below the 300 dpi minimum for publication.

9. **[HIGH] Figure dimensions inappropriate:** All notebooks use `figsize=(20, 9)` inches which exceeds journal column widths. Reduce to `(6.7, 4)` for double-column or `(3.35, 3)` for single-column.

10. **[HIGH] Random seed not propagated:** Scripts 1 and 2 accept `n_inits` but not `random_state`. Results are non-reproducible across runs.

11. **[HIGH] Parallel processing non-determinism:** `imap_unordered` causes varying result order between runs. Set `assert_ordered=True` or document non-determinism.

12. **[MEDIUM] No normality assessment:** Parametric tests (ANOVA, t-tests) lack Shapiro-Wilk checks. Microstate statistics (coverage, lifespan) are bounded/non-normal and may violate assumptions.

---

## 1 · Statistical Consistency

### 1.1 Multiple Comparison Correction Consistency

**[CRITICAL] Inconsistent FDR Method Between Notebooks**

| Location | Correction Method | Context |
|----------|------------------|---------|
| `1.5-GFP_stats_and_ideal_no_mstates.py:39` | `fdr_bh` | GFP peak comparisons |
| `2.5-microstate-stats.py:56` | `fdr_bh` | All microstate statistics |
| `3-psychedelic-experience-data.py:60` | `fdr_bh` | Experience correlations |
| `5-psychedelic-experience-data_IP.py:57` | `fdr_by` | Experience correlations (Inga/Povilas data) |

Notebook 5 uses the more conservative Benjamini-Yekutieli procedure while others use Benjamini-Hochberg. Results from notebook 5 are not directly comparable to other analyses. **Recommendation:** Standardize on `fdr_bh` across all notebooks or document justification for using `fdr_by` in notebook 5.

**[MAJOR] Missing Correction in Experience Analysis**

Multiple paired t-tests are performed (4 ASC scales, multiple BPRS scales, 12 persisting effects scales) without multiple comparison correction in `3-psychedelic-experience-data.py:267-330` and `5-psychedelic-experience-data_IP.py:363-426`. With ~20 tests, family-wise error rate is substantially inflated. Apply `fdr_bh` correction across all t-tests within each questionnaire category.

### 1.2 Within-Subject vs Between-Subject Test Appropriateness

**[CRITICAL] Incorrect Test Type for Microstate Comparisons**

`2.5-microstate-stats.py:97-135` treats microstate as between-subjects factor:
```python
anova = pg.anova(data=df, dv=dv, between=["microstate"], detailed=False)
posthoc = pg.pairwise_tukey(data=df, dv=dv, between="microstate", ...)
```

Each subject has measurements for microstates A, B, C, D - this is a within-subjects factor. The `pg.anova(between=["microstate"])` violates independence assumption and reduces power. **Correct approach:**
```python
anova = pg.rm_anova(data=df, dv=dv, within=["microstate"], subject="subject")
posthoc = pg.pairwise_tests(data=df, dv=dv, within="microstate", subject="subject", padjust="fdr_bh")
```

**[MAJOR] Tukey HSD Inappropriate for Repeated Measures**

Tukey's HSD is designed for between-subjects comparisons with equal variances. For repeated-measures/paired comparisons, use Bonferroni-corrected paired t-tests via `pg.pairwise_tests` with `within=` and `subject=` parameters.

### 1.3 Effect Size Reporting

Effect sizes are inconsistently reported:
- ANOVA tests do not report effect sizes (partial eta-squared available via pingouin)
- Pairwise tests consistently report Cohen's d via `effsize="cohen"`
- Experience questionnaire t-tests lack effect sizes entirely

**Recommendation:** Add `effsize="ng2"` to all `rm_anova()` calls and `effsize="cohen"` to all `ttest()` calls.

### 1.4 Alpha Level Consistency

While all thresholds use 0.05, implementation varies between `P_THRESH = 0.05` constant and hard-coded literals. Minor consistency issue that doesn't affect validity.

---

## 2 · Numerical & Algorithmic Correctness

### 2.1 Division by Zero Risk

**[CRITICAL] Transition Matrix Normalization**

`src/recording.py:273-275`:
```python
self.transition_mat = prob_matrix / np.nansum(prob_matrix, axis=1, keepdims=True)
```

If a microstate never transitions, the row sum is zero, producing NaN values. **Fix:**
```python
row_sums = prob_matrix.sum(axis=1, keepdims=True)
self.transition_mat = np.divide(prob_matrix, row_sums, 
                                 out=np.zeros_like(prob_matrix, dtype=float),
                                 where=row_sums!=0)
```

### 2.2 Algorithm Logic Errors

**[CRITICAL] Midpoint Fill Boundary Error**

`src/recording.py:177-185` has off-by-one errors and crashes with `IndexError` when ≤1 GFP peak exists:
```python
for idx in range(len(midpoints) - 1):  # Skips last segment!
segmentation[: midpoints[0]] = segmentation[peaks[0]]  # Crashes if midpoints empty
segmentation[midpoints[-1] :] = segmentation[peaks[-1]]  # Crashes if midpoints empty
```

**[HIGH] Empty Cluster Handling in K-means**

`src/microstates.py:261-265`: When a microstate is never assigned, it's set to zero and becomes permanently dead. **Fix:** Reinitialize empty clusters with random data points instead of zeroing them.

**[HIGH] GMD Assignment with Unnecessary np.abs()**

`src/recording.py:167-171`: Global Map Dissimilarity is already non-negative by definition, but `np.abs()` is applied redundantly. Remove `np.abs()` to avoid masking potential numerical issues.

### 2.3 Performance Issues

**[MEDIUM] Exhaustive Permutation Search**

`src/microstates.py:376-421`: `match_reorder_microstates()` uses `permutations()` which is O(n!). Fine for n≤4 but dangerous if extended. Current usage is acceptable given n_states ≤ 6 constraint.

**[MEDIUM] Dunn Index Memory Usage**

`src/clustering_scores.py:55-95`: Computes full O(n²) distance matrix. For EEG data with many timepoints, this can exhaust memory. Replace magic number `1000000` with `np.inf`.

### 2.4 Private API Usage

`src/microstates.py:354` uses `mne.channels.layout._find_topomap_coords` which is private and may break on MNE upgrades. Monitor MNE changelog or implement coordinate extraction directly.

### 2.5 Random Number Generation

`np.random.default_rng()` is used correctly, but `random_state` is not propagated through all function calls. Each parallel worker uses independent RNG state, making results non-reproducible.

### 2.6 Convergence Criteria

`src/microstates.py:268` uses relative convergence threshold `(prev_residual - residual) < (thresh * residual)` which becomes vanishingly small when residual approaches zero. Use hybrid absolute/relative tolerance.

---

## 3 · Statistical Assumptions & Test Validity

### 3.1 Sphericity Assumption

**[CRITICAL] No Sphericity Check in RM ANOVA**

All `pg.rm_anova()` calls in `2.5-microstate-stats.py` (lines 138-144, 197-206, 449-455, 461-467) return `eps` (epsilon) and `sphericity` columns from Mauchly's test but **never check or act on them**. With 5 time points × 2 conditions, sphericity violations are likely. **Fix:** Check `anova.loc[0, 'sphericity']` and apply Greenhouse-Geisser or Huynh-Feldt correction when violated.

### 3.2 Normality Assumption

**[MEDIUM] No Normality Assessment**

No Shapiro-Wilk tests, Q-Q plots, or skewness/kurtosis checks are performed. Microstate statistics have known non-normal distributions:
- **Coverage**: Bounded [0, 1] — beta-distributed
- **Lifespan**: Positive, right-skewed — gamma/log-normal
- **Occurrence**: Count data — Poisson/negative-binomial

With N ≈ 15-17 subjects after exclusions, CLT cannot be relied upon. **Recommendation:** Add normality checks; use non-parametric alternatives (Friedman instead of rm_anova, Wilcoxon instead of paired t-test) when assumptions violated.

### 3.3 Pearson vs Spearman

- **Pearson** for microstate template matching (`src/recording.py:155`): Acceptable for continuous spatial correlation
- **Spearman** for experience correlations: Appropriate for ordinal/non-normal questionnaire data

Usage is justified and consistent. Document assumptions in methods.

### 3.4 Bounded Dependent Variables

**[MEDIUM] Coverage Analyzed with ANOVA**

Coverage is a proportion bounded [0, 1] analyzed with ANOVA which assumes unbounded normality. Near boundaries, variance depends on mean (heteroscedasticity). **Recommendation:** Apply logit transform or use beta regression.

**[LOW] Transition Probabilities**

Transition probabilities bounded [0, 1] with simplex constraint (sum to 1). These are correctly excluded from correlation analysis via `.drop(ms_stats.filter(regex="transition"), axis=1)`.

### 3.5 Sample Size

N ≈ 15-17 subjects after exclusions ([4, 13, 14, 20, 22]). Small sample for 2-way RM ANOVA with low power for interaction effects. Report effect sizes and acknowledge power limitations.

### 3.6 Covariate Omission

No covariates included (age, sex, order effects). The `order` column exists in experience data but is not used. Consider including order as covariate if counterbalancing was incomplete.

---

## 4 · Visualisation & Publication Readiness

### 4.1 Figure DPI

**[CRITICAL] Insufficient DPI for Print**

| Location | Current | Required |
|----------|---------|----------|
| `src/microstates.py:372` | `dpi=150` | `dpi=300` minimum |
| PNG saves in `5-psychedelic-experience-data_IP.py` | Not set | `dpi=300` |

Journals require ≥300 dpi for raster images. Vector formats (EPS) are used but DPI should still be set for consistency.

### 4.2 Font Sizes

**[HIGH] Wrong Seaborn Context**

All notebooks use `sns.set_context("notebook", font_scale=1.75/2)` which is designed for screen display. **Fix:** Use `sns.set_context("paper", font_scale=1.5)` for journal figures.

### 4.3 Figure Dimensions

**[HIGH] Oversized Figures**

All files use `plt.rcParams["figure.figsize"] = (20, 9)` inches which exceeds journal maximum widths:
- Single column: 8.5 cm ≈ 3.35 inches
- Double column: 17 cm ≈ 6.7 inches

**Fix:** Reduce to `(6.7, 4)` for double-column or `(3.35, 3)` for single-column layouts.

### 4.4 Color Accessibility

- `cmap="coolwarm"` for correlations is acceptable diverging colormap
- Verify grayscale conversion for print - test figures in black and white

### 4.5 Axis Labels and Colorbars

**[MEDIUM] Missing Labels**

- Heatmaps lack colorbar labels (`cbar_kws={"label": "Correlation coefficient"}`)
- Boxplots rely on seaborn defaults without explicit axis labels
- Displot in `1.5-GFP_stats_and_ideal_no_mstates.py` has empty x/y labels

### 4.6 Annotations

**[MEDIUM] Non-Standard P-value Notation**

`"p < 0.000"` should be `"p < 0.001"` per standard convention in `1.5-GFP_stats_and_ideal_no_mstates.py:88` and `2.5-microstate-stats.py:286`.

**[MEDIUM] Non-Standard Thresholds in Notebook 5**

`5-psychedelic-experience-data_IP.py:59` uses `{0.01: "***", 0.05: "**", 0.1: "*"}` which includes 0.1 threshold. Standard is `{0.001: "***", 0.01: "**", 0.05: "*"}`.

### 4.7 File Formats

- `.eps` used for publication - good for vector graphics
- `5-psychedelic-experience-data_IP.py` uses `.png` without DPI specification
- `transparent=True` may cause issues with some journal submission systems

### 4.8 Multi-Panel Figures

No panel letters (a, b, c...) added for multi-panel figures in `2.5-microstate-stats.py`. Add annotations for clarity.

---

## 5 · Reproducibility & Scientific Workflow

### 5.1 Random Seed Handling

**[CRITICAL] Seeds Not Propagated**

| Issue | Location |
|-------|----------|
| `segment()` accepts `random_state` but loops create identical initializations if seed is int | `src/microstates.py:179-209` |
| `run_microstates()` lacks `random_state` parameter | `src/recording.py:111-132` |
| Scripts 1 and 2 don't accept `random_state` CLI argument | `1-main_*.py`, `2-main_*.py` |
| Each parallel worker uses independent RNG state | `src/helpers.py:67` |

**Critical bug:** When `random_state` is an integer, `np.random.default_rng(random_state)` creates the same generator state each iteration, causing identical initializations across `n_inits`. **Fix:** Create base RNG once, then derive per-iteration seeds.

### 5.2 Parallel Processing Non-Determinism

**[HIGH] `imap_unordered` Breaks Result Ordering**

`src/helpers.py:67` uses `pool.imap_unordered` by default (`assert_ordered=False`). Results appended in completion order, not input order. This causes non-deterministic CSV row ordering between runs. **Fix:** Set `assert_ordered=True` by default or document non-determinism.

### 5.3 Configuration Consistency

**[MEDIUM] `EXCLUDE_SUBJECTS` Duplicated**

Defined in 4 locations:
- `1-main_gfp_stats_and_ideal_no_mstates.py:30`
- `2-main_compute_microstates.py:41`
- `4-comparison-me-vs-IngaPovilas.py:27`
- `src/recording.py:358-376` (accepts as parameter)

**Risk:** Inconsistency if exclusion criteria change. **Fix:** Define once in `src/helpers.py` and import everywhere.

**[MEDIUM] Magic Numbers**

- `n_states` defaults differ: `segment()`=4, `run_microstates()`=200, pipeline=500
- `MS_OPTIONS` filter/state combinations documented but hardcoded
- `TARGET_LENGTH = 40` seconds, `RESAMPLE_TO = 256.0` Hz - well-named but inflexible

### 5.4 Hardcoded Paths

`0-main_preprocess_data.py:26-28`:
```python
RAW_DATA = "/Volumes/Q/science-brain/UI-microstates/data_v2 - Palenicek-raw/raw_continuous"
```

Absolute path to external volume is user-specific. **Fix:** Move to CLI argument or environment variable.

### 5.5 Version Pinning

`pyproject.toml` uses minimum versions (`>=`) but `uv.lock` provides full reproducibility. Python 3.14 requirement (`requires-python = ">=3.14"`) should be verified as Python 3.14 is not yet released.

### 5.6 Pipeline Dependencies

- Notebook `1.5` requires output from script `1`
- Notebooks `2.5`, `3` require output from script `2` with hardcoded folder name `"20260116-new-recompute"`
- Output filenames like `ms_stats_run2.csv` suggest manual versioning

**Fix:** Document pipeline execution order in CLAUDE.md; use timestamped or git-hashed output names.

### 5.7 Output Determinism

CSV outputs silently overwrite previous runs. **Fix:** Add `--output-name` CLI option with default including timestamp or git hash.

### 5.8 Missing Tests

No unit tests for core algorithms (`segment()`, clustering scores) or integration tests for pipeline. Consider adding tests with known inputs/outputs.

---

## Appendix: Quick-Fix Checklist

| # | File | Line | Issue | Fix |
|---|------|------|-------|-----|
| 1 | `2.5-microstate-stats.py` | 97 | Between-subjects ANOVA for within-subjects factor | Change to `pg.rm_anova(within=["microstate"], subject="subject")` |
| 2 | `2.5-microstate-stats.py` | 106 | Tukey HSD for repeated measures | Change to `pg.pairwise_tests(within="microstate", subject="subject", padjust="bonferroni")` |
| 3 | `src/recording.py` | 273 | Division by zero in transition matrix | Add `where=row_sums!=0` to `np.divide` |
| 4 | `src/recording.py` | 177 | Midpoint fill crashes with ≤1 GFP peak | Add edge case handling for empty midpoints |
| 5 | `2.5-microstate-stats.py` | 138 | No sphericity check in RM ANOVA | Check `anova['sphericity']` and apply GG/HF correction |
| 6 | `5-psychedelic-experience-data_IP.py` | 57 | Inconsistent FDR method | Change `fdr_by` to `fdr_bh` or document justification |
| 7 | `3-psychedelic-experience-data.py` | 267 | Missing multiple comparison correction for experience t-tests | Apply `fdr_bh` across all questionnaire t-tests |
| 8 | `src/microstates.py` | 372 | Insufficient DPI for print | Change `dpi=150` to `dpi=300` |
| 9 | All notebooks | 30, 44, 47 | Oversized figures | Change `(20, 9)` to `(6.7, 4)` or `(3.35, 3)` |
| 10 | All notebooks | 30, 44, 45 | Wrong seaborn context | Change `set_context("notebook")` to `set_context("paper")` |
| 11 | `1-main_*.py`, `2-main_*.py` | 107, 99 | No random_state CLI parameter | Add `random_state: int | None = None` argument |
| 12 | `src/helpers.py` | 67 | Parallel processing non-determinism | Set `assert_ordered=True` by default |
| 13 | `src/microstates.py` | 179 | Identical RNG initializations | Derive per-iteration seeds from base RNG |
| 14 | `2.5-microstate-stats.py` | 56 | P_THRESH hardcoded vs constant | Use `P_THRESH` consistently |
| 15 | `src/recording.py` | 261 | Empty clusters become permanent zeros | Reinitialize with random data points |
| 16 | `1.5-GFP_stats_and_ideal_no_mstates.py` | 88 | Non-standard p-value notation | Change `"p < 0.000"` to `"p < 0.001"` |
| 17 | `5-psychedelic-experience-data_IP.py` | 59 | Non-standard star thresholds | Change to `{0.001: "***", 0.01: "**", 0.05: "*"}` |
| 18 | All heatmaps | various | Missing colorbar labels | Add `cbar_kws={"label": "..."}` |
| 19 | `src/microstates.py` | 354 | Private MNE API usage | Monitor MNE changelog or use public API |
| 20 | `0-main_preprocess_data.py` | 26 | Hardcoded absolute path | Move to CLI argument or env variable |
| 21 | `1-main_*.py`, `2-main_*.py` | 30, 41 | EXCLUDE_SUBJECTS duplicated | Define once in `helpers.py` and import |
| 22 | `src/clustering_scores.py` | 70 | Magic number in Dunn index | Change `1000000` to `np.inf` |
| 23 | `src/microstates.py` | 268 | Relative convergence threshold | Use hybrid absolute/relative tolerance |
| 24 | `src/recording.py` | 152 | Magic number for unassigned label | Define `UNASSIGNED_LABEL = -1` constant |
| 25 | `2.5-microstate-stats.py` | 106 | Tukey HSD missing padjust | Add explicit correction parameter |

---

*Generated by /review-scicode via OpenCode · Orchestrator: litellm/kimi-k2.5 · Sub-agents: litellm/qwen3.5 (×5)*
