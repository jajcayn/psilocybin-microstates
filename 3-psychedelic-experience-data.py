# ruff: noqa: ANN001, ANN202, E501

import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Psychedelic experience data

    1-40Hz filter, 4 microstates
    Correlations among experience data and microstate statistics
    """)
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo

    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pingouin as pg
    import seaborn as sns
    from pingouin.multicomp import multicomp
    from scipy.stats import pearsonr, spearmanr
    from sklearn.decomposition import PCA

    from src.helpers import DATA_ROOT, PLOTS_ROOT, RESULTS_ROOT, make_dirs

    plt.style.use("default")
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.figsize"] = (20, 9)
    sns.set_context("notebook", font_scale=1.75)

    MS_STATS_ROOT = os.path.join(RESULTS_ROOT, "20260325-final")
    WORKING_DIR = os.path.join(RESULTS_ROOT, "experience")
    if not os.path.exists(WORKING_DIR):
        make_dirs(WORKING_DIR)
    PLOTTING_DIR = os.path.join(PLOTS_ROOT, "final")
    if not os.path.exists(PLOTTING_DIR):
        make_dirs(PLOTTING_DIR)

    # if True, will save all statistical tests as csv and all plots as eps
    SAVE_RESULTS = True
    PLOT_EXT = ".eps"
    # which correction for multiple comparisons should be used
    MULTI_CORRECTION = "fdr_bh"
    PVAL_STARS = {0.001: "***", 0.01: "**", 0.05: "*"}
    return (
        DATA_ROOT,
        MS_STATS_ROOT,
        MULTI_CORRECTION,
        PCA,
        PLOTTING_DIR,
        PLOT_EXT,
        PVAL_STARS,
        SAVE_RESULTS,
        WORKING_DIR,
        multicomp,
        np,
        os,
        pd,
        pearsonr,
        pg,
        plt,
        sns,
        spearmanr,
    )


@app.cell
def _(DATA_ROOT, MS_STATS_ROOT, os, pd):
    experience = pd.read_csv(
        os.path.join(DATA_ROOT, "experience_data", "experience_processed.csv")
    )
    experience = experience.rename(columns={"group": "condition"})

    persisting_effs = pd.read_csv(
        os.path.join(
            DATA_ROOT,
            "experience_data",
            "persisting_effects_scale_processed.csv",
        )
    )
    persisting_effs = persisting_effs.rename(columns={"group": "condition"})

    ms_stats = pd.read_csv(
        os.path.join(MS_STATS_ROOT, "ms_stats.csv"), index_col=[0]
    )
    # divide session into PLA/PSI and time for plotting purposes
    ms_stats[["condition", "time"]] = ms_stats["session"].str.split(
        "-", expand=True
    )
    # divide to filters
    # ms_stats_220 = ms_stats[ms_stats["ms_opts"] == "(2.0, 20.0, 4)"]
    # ms_stats_220 = ms_stats_220.sort_values(by=["condition", "time"])
    ms_stats = ms_stats[ms_stats["ms_opts"] == "(1.0, 40.0, 3)"]
    ms_stats = ms_stats.sort_values(by=["condition", "time"]).drop(
        ["template_corr"], axis=1
    )
    return experience, ms_stats, persisting_effs


@app.cell
def _(ms_stats):
    ms_stats
    return


@app.cell
def _(
    MULTI_CORRECTION,
    PCA,
    PLOTTING_DIR,
    PLOT_EXT,
    PVAL_STARS,
    SAVE_RESULTS,
    WORKING_DIR,
    np,
    os,
    pd,
    pearsonr,
    plt,
    sns,
    spearmanr,
):
    def pd_pca(df, n_comps, col_name=None):
        if col_name is None:
            col_name = ""
        else:
            col_name = col_name + "_"
        pca = PCA(n_components=n_comps)
        pcs = pca.fit_transform(df.values)
        loadings = pca.components_[0]
        if loadings.sum() < 0:
            pcs = pcs * -1
            loadings = loadings * -1
        return pd.DataFrame(
            pcs,
            index=df.index,
            columns=[
                f"{col_name}PC{i + 1}_{pca.explained_variance_ratio_[i]:.1%}"
                for i in range(n_comps)
            ],
        )

    def plot_corr(
        df,
        method="pearson",
        title="",
        mask_upper=True,
        mask_pval=True,
        fname=None,
        lines=None,
        rename_cols=None,
        xticklabels_rotation=0,
    ):
        import re

        df = df.rename(
            columns=lambda c: re.sub(r"_PC(\d+)_[\d.]+%$", r"_PC\1", c)
        )
        if rename_cols:
            df = df.rename(columns=rename_cols)
        plt.figure(figsize=(16, 14))
        corr = df.corr(method=method, numeric_only=True)
        lines = lines or []

        def pval_func(x, y):
            if method == "pearson":
                return pearsonr(x, y)[1]
            if method == "spearman":
                return spearmanr(x, y)[1]

        pval = df.corr(method=pval_func, numeric_only=True)
        mask_up = (
            np.triu(np.ones_like(corr, dtype=bool))
            if mask_upper
            else np.zeros_like(corr, dtype=bool)
        )
        mask_pval = (
            pval > 0.05 if mask_pval else np.zeros_like(corr, dtype=bool)
        )
        mask = np.logical_or(mask_up, mask_pval)
        sns.heatmap(
            corr,
            annot=df.rcorr(
                method=method,
                upper="pval",
                decimals=2,
                padjust=MULTI_CORRECTION,
                stars=True,
                pval_stars=PVAL_STARS,
            ),
            cmap="coolwarm",
            annot_kws={"size": 15},
            vmin=-1,
            vmax=1,
            mask=mask,
            fmt="",
            cbar_kws={"shrink": 0.5},
        )
        xlims = plt.gca().get_xlim()
        ylims = plt.gca().get_ylim()
        plt.plot(
            [xlims[0], xlims[1]],
            [ylims[1], ylims[0]],
            "--",
            color=plt.rcParams["text.color"],
            linewidth=1.0,
        )
        plt.hlines(
            lines,
            *plt.gca().get_xlim(),
            color=plt.rcParams["text.color"],
            linewidth=1.0,
        )
        plt.vlines(
            lines,
            *plt.gca().get_ylim(),
            color=plt.rcParams["text.color"],
            linewidth=1.0,
        )
        if xticklabels_rotation:
            plt.xticks(rotation=xticklabels_rotation, ha="right")
            plt.yticks(rotation=0)
        plt.title(title)
        if SAVE_RESULTS:
            plt.savefig(
                os.path.join(
                    PLOTTING_DIR, f"{fname}_{method}_correlation{PLOT_EXT}"
                ),
                bbox_inches="tight",
                transparent=True,
            )
            _p = pval.map(replace_pval)
            corr = corr.round(5).astype(str) + _p
            corr.to_csv(
                os.path.join(WORKING_DIR, f"{fname}_{method}_correlation.csv")
            )

    def replace_pval(x, pval_stars=PVAL_STARS):
        for key, value in pval_stars.items():
            if x < key:
                return value
        return ""

    return pd_pca, plot_corr, replace_pval


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PLA vs PSI differences in ASC
    """)
    return


@app.cell
def _(experience, ms_stats, pd, pg):
    df_experience = experience[experience["subject"].isin(ms_stats["subject"])]
    ASC = ["AIA", "OSE", "VUS", "VWB"]
    asc_ttest = pd.DataFrame()
    for asc in ASC:
        _tt = pg.ttest(
            df_experience[df_experience["condition"] == "PLA"][asc],
            df_experience[df_experience["condition"] == "PSI"][asc],
            paired=True,
        )
        _tt.index = [asc]
        asc_ttest = pd.concat([asc_ttest, _tt], axis=0)
    asc_ttest
    return (df_experience,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PLA vs PSI differences in BPRS
    """)
    return


@app.cell
def _(df_experience, experience, pd, pg):
    BP = list(experience.filter(like="BP").columns)
    bprs_ttest = pd.DataFrame()
    for bprs in BP:
        _tt = pg.ttest(
            df_experience[df_experience["condition"] == "PLA"][bprs],
            df_experience[df_experience["condition"] == "PSI"][bprs],
            paired=True,
        )
        _tt.index = [bprs]
        bprs_ttest = pd.concat([bprs_ttest, _tt], axis=0)
    bprs_ttest
    return


@app.cell
def _(ms_stats, pd, persisting_effs, pg):
    df_pers_effs = persisting_effs[
        persisting_effs["subject"].isin(ms_stats["subject"])
    ]
    cols = [
        "Attitudes about Life positive",
        "Attitudes about Life negative",
        "Attitudes about Self positive",
        "Attitudes about Self negative",
        "Mood Changes positive",
        "Mood Changes negative",
        "Relationships positive",
        "Relationships negative",
        "Behavioral Changes positive",
        "Behavioral Changes negative",
        "Spirituality positive",
        "Spirituality negative",
    ]
    pers_ttest = pd.DataFrame()
    for col in cols:
        _tt = pg.ttest(
            df_pers_effs[df_pers_effs["condition"] == "PLA"][col].astype(float),
            df_pers_effs[df_pers_effs["condition"] == "PSI"][col].astype(float),
            paired=True,
        )
        _tt.index = [col]
        pers_ttest = pd.concat([pers_ttest, _tt], axis=0)
    pers_ttest
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Correlations within experience data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Only psilocybin condition
    """)
    return


@app.cell
def _(MULTI_CORRECTION, experience, plot_corr):
    # only pval <= 0.05 unmasked
    corr_df = (
        experience[experience["condition"] == "PSI"]
        .drop("subject", axis=1)
        .drop(
            list(experience.filter(regex="BP.*T0").columns) + ["PSI_conc_T0"],
            axis=1,
        )
    )
    _rename_cols = {
        "BP-5:FKT-I_T70": "FI_T70",
        "BP-5:FKT-II_T70": "FII_T70",
        "BP-5:FKT-III_T70": "FIII_T70",
        "BP-5:FKT-IV_T70": "FIV_T70",
        "BP-5:FKT-V_T70": "FV_T70",
        "BP-5:FKT-I_T180": "FI_T180",
        "BP-5:FKT-II_T180": "FII_T180",
        "BP-5:FKT-III_T180": "FIII_T180",
        "BP-5:FKT-IV_T180": "FIV_T180",
        "BP-5:FKT-V_T180": "FV_T180",
        "PSI_conc_T60": "[PSI]T60",
        "PSI_conc_T120": "[PSI]T120",
        "PSI_conc_T240": "[PSI]T240",
        "PSI_conc_T360": "[PSI]T360",
    }
    plot_corr(
        corr_df,
        title=f"PSI only \n masked out non-significant ~ {MULTI_CORRECTION.upper()} correction",
        mask_upper=False,
        mask_pval=True,
        lines=[4, 9, 14, 15, 19, 20],
        method="spearman",
        fname="experience_PSIonly",
        rename_cols=_rename_cols,
        xticklabels_rotation=90,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Both psilocybin and placebo conditions
    """)
    return


@app.cell
def _(MULTI_CORRECTION, experience, plot_corr):
    # only pval <= 0.05 unmasked
    plot_corr(
        experience.drop("subject", axis=1),
        title=f"PLA & PSI \n masked out non-significant ~ {MULTI_CORRECTION.upper()} correction",
        mask_upper=False,
        mask_pval=True,
        lines=[4, 9, 14, 19, 20],
        method="spearman",
        fname="experience_PLAandPSI",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Correlations with ASC questionnaire
    """)
    return


@app.cell
def _(experience, ms_stats):
    ASC_1 = ["AIA", "OSE", "VUS", "VWB"]
    experience_asc = experience[["subject", "condition"] + ASC_1]
    # match subjects
    experience_asc = experience_asc.loc[
        experience_asc["subject"].isin(ms_stats["subject"])
    ]
    return ASC_1, experience_asc


@app.cell
def _(ASC_1, experience_asc, ms_stats, np, pd, spearmanr):
    grand = pd.DataFrame([])
    grand_pvals = pd.DataFrame([])
    for _time in np.unique(ms_stats["time"]):
        for _ms in np.unique(ms_stats["microstate"]):
            for _method in ["spearman", lambda x, y: spearmanr(x, y)[1]]:
                asc_corrs = pd.concat(
                    [
                        ms_stats[
                            (ms_stats["condition"] == "PSI")
                            & (ms_stats["microstate"] == _ms)
                            & (ms_stats["time"] == _time)
                        ]
                        .drop(ms_stats.filter(regex="transition"), axis=1)
                        .set_index("subject")
                        .corrwith(
                            experience_asc[
                                experience_asc["condition"] == "PSI"
                            ].set_index("subject")[asc],
                            method=_method,
                            numeric_only=True,
                        )
                        for asc in ASC_1
                    ],
                    axis=1,
                )
                asc_corrs.columns = ASC_1
                asc_corrs["microstate"] = _ms
                asc_corrs["time"] = _time
                asc_corrs = asc_corrs.dropna(axis=1)
                asc_corrs.index = asc_corrs.index.rename("stat")
                asc_corrs = asc_corrs.set_index(
                    ["time", "microstate", asc_corrs.index]
                )
                if _method == "spearman":
                    grand = pd.concat([grand, asc_corrs], axis=0)
                else:
                    grand_pvals = pd.concat([grand_pvals, asc_corrs], axis=0)
    return grand, grand_pvals


@app.cell
def _(
    MULTI_CORRECTION,
    PLOTTING_DIR,
    PLOT_EXT,
    SAVE_RESULTS,
    WORKING_DIR,
    grand,
    grand_pvals,
    multicomp,
    np,
    os,
    pd,
    plt,
    replace_pval,
    sns,
):
    _plot_times = ["T2", "T3", "T4", "T5"]
    plt.figure(figsize=(11, 15))
    _plot_df = grand[grand.index.get_level_values("time").isin(_plot_times)]
    _, _bonf_corrd = multicomp(
        grand_pvals[
            grand_pvals.index.get_level_values("time").isin(_plot_times)
        ].values,
        alpha=0.05,
        method=MULTI_CORRECTION,
    )
    _bonf_corrd = pd.DataFrame(
        _bonf_corrd, columns=_plot_df.columns, index=_plot_df.index
    ).map(replace_pval)
    sns.heatmap(
        _plot_df,
        cmap="coolwarm",
        annot=_plot_df.round(2).astype(str) + _bonf_corrd,
        vmin=-0.8,
        vmax=0.8,
        fmt="",
        mask=grand_pvals[grand.index.get_level_values("time").isin(_plot_times)]
        > 0.05,
        annot_kws={"size": 15},
    )
    plt.hlines(
        np.arange(3, grand.shape[0], 3).tolist(),
        *plt.gca().get_xlim(),
        color=plt.rcParams["text.color"],
        linewidth=1.0,
    )
    plt.vlines(
        [],
        *plt.gca().get_ylim(),
        color=plt.rcParams["text.color"],
        linewidth=1.0,
    )
    plt.title(
        f"MS stats vs. ASC / T2 - T5 \n masked out non-significant ~ {MULTI_CORRECTION.upper()} correction"
    )
    if SAVE_RESULTS:
        plt.savefig(
            os.path.join(
                PLOTTING_DIR,
                f"MSstats_w_ASC_T2-T5_1-40filt_spearman_correlation{PLOT_EXT}",
            ),
            bbox_inches="tight",
            transparent=True,
        )
        _p = grand_pvals.map(
            lambda x: "".join(["*" for t in [0.001, 0.01, 0.05] if x <= t])
        )
        _corr = grand.round(5).astype(str) + _p
        _corr.to_csv(
            os.path.join(
                WORKING_DIR, "MSstats_w_ASC_1-40filt_spearman_correlation.csv"
            )
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Correlations with BPRS questionnaire
    """)
    return


@app.cell
def _(experience, ms_stats):
    BP_1 = list(experience.filter(like="BP").columns)
    experience_bprs = experience[["subject", "condition"] + BP_1]
    # match subjects
    experience_bprs = experience_bprs.loc[
        experience_bprs["subject"].isin(ms_stats["subject"])
    ]
    return BP_1, experience_bprs


@app.cell
def _(BP_1, experience_bprs, ms_stats, np, pd, spearmanr):
    grand_1 = pd.DataFrame([])
    grand_pvals_1 = pd.DataFrame([])
    for _time in np.unique(ms_stats["time"]):
        for _ms in np.unique(ms_stats["microstate"]):
            for _method in ["spearman", lambda x, y: spearmanr(x, y)[1]]:
                bprs_corrs = pd.concat(
                    [
                        ms_stats[
                            (ms_stats["condition"] == "PSI")
                            & (ms_stats["microstate"] == _ms)
                            & (ms_stats["time"] == _time)
                        ]
                        .drop(ms_stats.filter(regex="transition"), axis=1)
                        .set_index("subject")
                        .corrwith(
                            experience_bprs[
                                experience_bprs["condition"] == "PSI"
                            ].set_index("subject")[bprs],
                            method=_method,
                            numeric_only=True,
                        )
                        for bprs in BP_1
                    ],
                    axis=1,
                )
                bprs_corrs.columns = BP_1
                bprs_corrs["microstate"] = _ms
                bprs_corrs["time"] = _time
                bprs_corrs = bprs_corrs.dropna(axis=1)
                bprs_corrs.index = bprs_corrs.index.rename("stat")
                bprs_corrs = bprs_corrs.set_index(
                    ["time", "microstate", bprs_corrs.index]
                )
                if _method == "spearman":
                    grand_1 = pd.concat([grand_1, bprs_corrs], axis=0)
                else:
                    grand_pvals_1 = pd.concat(
                        [grand_pvals_1, bprs_corrs], axis=0
                    )
    return grand_1, grand_pvals_1


@app.cell
def _(
    MULTI_CORRECTION,
    PLOTTING_DIR,
    PLOT_EXT,
    SAVE_RESULTS,
    WORKING_DIR,
    grand_1,
    grand_pvals_1,
    multicomp,
    np,
    os,
    pd,
    plt,
    replace_pval,
    sns,
):
    _plot_times = ["T2", "T3", "T4", "T5"]
    plt.figure(figsize=(11, 15))
    _plot_df = grand_1[grand_1.index.get_level_values("time").isin(_plot_times)]
    _, _bonf_corrd = multicomp(
        grand_pvals_1[
            grand_pvals_1.index.get_level_values("time").isin(_plot_times)
        ].values,
        alpha=0.05,
        method=MULTI_CORRECTION,
    )
    _bonf_corrd = pd.DataFrame(
        _bonf_corrd, columns=_plot_df.columns, index=_plot_df.index
    ).map(replace_pval)
    sns.heatmap(
        _plot_df,
        cmap="coolwarm",
        annot=_plot_df.round(2).astype(str) + _bonf_corrd,
        vmin=-0.8,
        vmax=0.8,
        fmt="",
        mask=grand_pvals_1[
            grand_1.index.get_level_values("time").isin(_plot_times)
        ]
        > 0.05,
        annot_kws={"size": 15},
    )
    plt.hlines(
        np.arange(3, grand_1.shape[0], 3).tolist(),
        *plt.gca().get_xlim(),
        color=plt.rcParams["text.color"],
        linewidth=1.0,
    )
    plt.vlines(
        [3, 8],
        *plt.gca().get_ylim(),
        color=plt.rcParams["text.color"],
        linewidth=1.0,
    )
    plt.title(
        f"MS stats vs. BPRS / T2 - T5 \n masked out non-significant ~ {MULTI_CORRECTION.upper()} correction"
    )
    if SAVE_RESULTS:
        plt.savefig(
            os.path.join(
                PLOTTING_DIR,
                f"MSstats_w_BPRS_T2-T5_1-40filt_spearman_correlation{PLOT_EXT}",
            ),
            bbox_inches="tight",
            transparent=True,
        )
        _p = grand_pvals_1.map(
            lambda x: "".join(["*" for t in [0.001, 0.01, 0.05] if x <= t])
        )
        _corr = grand_1.round(5).astype(str) + _p
        _corr.to_csv(
            os.path.join(
                WORKING_DIR, "MSstats_w_BPRS_1-40filt_spearman_correlation.csv"
            )
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Correlations with dosage
    """)
    return


@app.cell
def _(experience, ms_stats):
    CONC = list(experience.filter(like="conc").columns) + ["dose [mg]"]

    experience_conc = experience[["subject", "condition"] + CONC]
    # match subjects
    experience_conc = experience_conc.loc[
        experience_conc["subject"].isin(ms_stats["subject"])
    ]
    return CONC, experience_conc


@app.cell
def _(CONC, experience_conc, ms_stats, np, pd, spearmanr):
    grand_2 = pd.DataFrame([])
    grand_pvals_2 = pd.DataFrame([])
    for _time in np.unique(ms_stats["time"]):
        for _ms in np.unique(ms_stats["microstate"]):
            for _method in ["spearman", lambda x, y: spearmanr(x, y)[1]]:
                conc_corrs = pd.concat(
                    [
                        ms_stats[
                            (ms_stats["condition"] == "PSI")
                            & (ms_stats["microstate"] == _ms)
                            & (ms_stats["time"] == _time)
                        ]
                        .drop(ms_stats.filter(regex="transition"), axis=1)
                        .set_index("subject")
                        .corrwith(
                            experience_conc[
                                experience_conc["condition"] == "PSI"
                            ].set_index("subject")[conc],
                            method=_method,
                            numeric_only=True,
                        )
                        for conc in CONC
                    ],
                    axis=1,
                )
                conc_corrs.columns = CONC
                conc_corrs["microstate"] = _ms
                conc_corrs["time"] = _time
                conc_corrs = conc_corrs.dropna(axis=1)
                conc_corrs.index = conc_corrs.index.rename("stat")
                conc_corrs = conc_corrs.set_index(
                    ["time", "microstate", conc_corrs.index]
                )
                if _method == "spearman":
                    grand_2 = pd.concat([grand_2, conc_corrs], axis=0)
                else:
                    grand_pvals_2 = pd.concat(
                        [grand_pvals_2, conc_corrs], axis=0
                    )
    return grand_2, grand_pvals_2


@app.cell
def _(
    MULTI_CORRECTION,
    PLOTTING_DIR,
    PLOT_EXT,
    SAVE_RESULTS,
    WORKING_DIR,
    grand_2,
    grand_pvals_2,
    multicomp,
    np,
    os,
    pd,
    plt,
    replace_pval,
    sns,
):
    _plot_times = ["T2", "T3", "T4", "T5"]
    plt.figure(figsize=(11, 15))
    _plot_df = grand_2[grand_2.index.get_level_values("time").isin(_plot_times)]
    _, _bonf_corrd = multicomp(
        grand_pvals_2[
            grand_pvals_2.index.get_level_values("time").isin(_plot_times)
        ].values,
        alpha=0.05,
        method=MULTI_CORRECTION,
    )
    _bonf_corrd = pd.DataFrame(
        _bonf_corrd, columns=_plot_df.columns, index=_plot_df.index
    ).map(replace_pval)
    sns.heatmap(
        _plot_df,
        cmap="coolwarm",
        annot=_plot_df.round(2).astype(str) + _bonf_corrd,
        vmin=-0.8,
        vmax=0.8,
        fmt="",
        mask=grand_pvals_2[
            grand_2.index.get_level_values("time").isin(_plot_times)
        ]
        > 0.05,
        annot_kws={"size": 15},
    )
    plt.hlines(
        np.arange(3, grand_2.shape[0], 3).tolist(),
        *plt.gca().get_xlim(),
        color=plt.rcParams["text.color"],
        linewidth=1.0,
    )
    plt.vlines(
        [4],
        *plt.gca().get_ylim(),
        color=plt.rcParams["text.color"],
        linewidth=1.0,
    )
    plt.title(
        f"MS stats vs. concentration / T2 - T5 \n masked out non-significant ~ {MULTI_CORRECTION.upper()} correction"
    )
    if SAVE_RESULTS:
        plt.savefig(
            os.path.join(
                PLOTTING_DIR,
                f"MSstats_w_conc_T2-T5_1-40filt_spearman_correlation{PLOT_EXT}",
            ),
            bbox_inches="tight",
            transparent=True,
        )
        _p = grand_pvals_2.map(
            lambda x: "".join(["*" for t in [0.001, 0.01, 0.05] if x <= t])
        )
        _corr = grand_2.round(5).astype(str) + _p
        _corr.to_csv(
            os.path.join(
                WORKING_DIR, "MSstats_w_conc_1-40filt_spearman_correlation.csv"
            )
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Correlations with persistent effects
    """)
    return


@app.cell
def _(ms_stats, persisting_effs):
    # match subjects
    persisting = persisting_effs.drop("order", axis=1).loc[
        persisting_effs["subject"].isin(ms_stats["subject"])
    ]
    EFFECTS = list(persisting.columns[2:])
    return EFFECTS, persisting


@app.cell
def _(EFFECTS, ms_stats, np, pd, persisting, spearmanr):
    grand_3 = pd.DataFrame([])
    grand_pvals_3 = pd.DataFrame([])
    for _time in np.unique(ms_stats["time"]):
        for _ms in np.unique(ms_stats["microstate"]):
            for _method in ["spearman", lambda x, y: spearmanr(x, y)[1]]:
                pers_corrs = pd.concat(
                    [
                        ms_stats[
                            (ms_stats["condition"] == "PSI")
                            & (ms_stats["microstate"] == _ms)
                            & (ms_stats["time"] == _time)
                        ]
                        .drop(ms_stats.filter(regex="transition"), axis=1)
                        .set_index("subject")
                        .corrwith(
                            persisting[
                                persisting["condition"] == "PSI"
                            ].set_index("subject")[eff],
                            method=_method,
                            numeric_only=True,
                        )
                        for eff in EFFECTS
                    ],
                    axis=1,
                )
                pers_corrs.columns = EFFECTS
                pers_corrs["microstate"] = _ms
                pers_corrs["time"] = _time
                pers_corrs = pers_corrs.dropna(axis=1)
                pers_corrs.index = pers_corrs.index.rename("stat")
                pers_corrs = pers_corrs.set_index(
                    ["time", "microstate", pers_corrs.index]
                )
                if _method == "spearman":
                    grand_3 = pd.concat([grand_3, pers_corrs], axis=0)
                else:
                    grand_pvals_3 = pd.concat(
                        [grand_pvals_3, pers_corrs], axis=0
                    )
    return grand_3, grand_pvals_3


@app.cell
def _(
    MULTI_CORRECTION,
    PLOTTING_DIR,
    PLOT_EXT,
    SAVE_RESULTS,
    WORKING_DIR,
    grand_3,
    grand_pvals_3,
    multicomp,
    np,
    os,
    pd,
    plt,
    replace_pval,
    sns,
):
    _plot_times = ["T2", "T3", "T4", "T5"]
    plt.figure(figsize=(11, 15))
    _plot_df = grand_3[grand_3.index.get_level_values("time").isin(_plot_times)]
    _, _bonf_corrd = multicomp(
        grand_pvals_3[
            grand_pvals_3.index.get_level_values("time").isin(_plot_times)
        ].values,
        alpha=0.05,
        method=MULTI_CORRECTION,
    )
    _bonf_corrd = pd.DataFrame(
        _bonf_corrd, columns=_plot_df.columns, index=_plot_df.index
    ).map(replace_pval)
    sns.heatmap(
        _plot_df,
        cmap="coolwarm",
        annot=_plot_df.round(2).astype(str) + _bonf_corrd,
        vmin=-0.8,
        vmax=0.8,
        fmt="",
        mask=grand_pvals_3[
            grand_3.index.get_level_values("time").isin(_plot_times)
        ]
        > 0.05,
        annot_kws={"size": 15},
    )
    plt.hlines(
        np.arange(3, grand_3.shape[0], 3).tolist(),
        *plt.gca().get_xlim(),
        color=plt.rcParams["text.color"],
        linewidth=1.0,
    )
    plt.vlines(
        [2, 4, 6, 8, 9],
        *plt.gca().get_ylim(),
        color=plt.rcParams["text.color"],
        linewidth=1.0,
    )
    plt.title(
        f"MS stats vs. persistent effects / T2-T5 \n masked out non-significant ~ {MULTI_CORRECTION.upper()} correction"
    )
    if SAVE_RESULTS:
        plt.savefig(
            os.path.join(
                PLOTTING_DIR,
                f"MSstats_w_persistent_effects_T2-T5_1-40filt_spearman_correlation{PLOT_EXT}",
            ),
            bbox_inches="tight",
            transparent=True,
        )
        _p = grand_pvals_3.map(
            lambda x: "".join(["*" for t in [0.001, 0.01, 0.05] if x <= t])
        )
        _corr = grand_3.round(5).astype(str) + _p
        _corr.to_csv(
            os.path.join(
                WORKING_DIR,
                "MSstats_w_persistent_effects_1-40filt_spearman_correlation.csv",
            )
        )
    return


@app.cell
def _(ASC_1, experience, ms_stats, pd, pd_pca):
    experience_grp = experience[experience["condition"] == "PSI"].loc[
        experience["subject"].isin(ms_stats["subject"])
    ]
    experience_grp = experience_grp.drop(
        ["order", "usage time"], axis=1
    ).set_index(["subject", "condition"])
    _groupings = {
        "ASC": ASC_1,
        "BPRS_T70": list(experience_grp.filter(like="T70").columns),
        "BPRS_T180": list(experience_grp.filter(like="T180").columns),
    }
    for key, val in _groupings.items():
        experience_grp = pd.concat(
            [experience_grp, pd_pca(experience_grp[val], 1, key)], axis=1
        ).drop(val, axis=1)
    experience_grp = experience_grp.drop(
        list(experience_grp.filter(like="T0").columns), axis=1
    )
    experience_grp.index = experience_grp.index.droplevel("condition")
    return (experience_grp,)


@app.cell
def _(ms_stats, persisting_effs):
    persisting_effs_grp = persisting_effs[
        persisting_effs["condition"] == "PSI"
    ].loc[persisting_effs["subject"].isin(ms_stats["subject"])]
    persisting_effs_grp = persisting_effs_grp.drop(
        ["order", "condition"]
        + list(persisting_effs_grp.filter(like="negative").columns),
        axis=1,
    ).set_index("subject")
    return (persisting_effs_grp,)


@app.cell
def _(MULTI_CORRECTION, experience_grp, ms_stats, pd, pd_pca, plot_corr):
    _ms_stats_grp = (
        ms_stats[ms_stats["condition"] == "PSI"]
        .drop(["session", "ms_opts"], axis=1)
        .set_index(["subject", "condition", "time"])
    )
    _groupings = ["coverage", "lifespan", "occurrence"]
    for _time in ["T2", "T3", "T4", "T5"]:
        _ms_stats_grp_ = pd.DataFrame()
        for _grp in _groupings:
            _pca_mat = pd_pca(
                _ms_stats_grp[
                    _ms_stats_grp.index.get_level_values("time") == _time
                ]
                .reset_index()
                .pivot(index="subject", columns="microstate", values=_grp),
                1,
                f"{_grp}_{_time}",
            )
            _ms_stats_grp_ = pd.concat([_ms_stats_grp_, _pca_mat], axis=1)
        _grp_corrs = pd.concat([experience_grp, _ms_stats_grp_], axis=1)
        plot_corr(
            _grp_corrs,
            method="spearman",
            mask_upper=False,
            mask_pval=True,
            lines=[1, 5, 6, 8],
            title=f"Experience correlations {_time} \n masked out non-significant ~ {MULTI_CORRECTION.upper()} correction",
            fname=f"experience_agg_{_time}",
        )
    return


@app.cell
def _(MULTI_CORRECTION, ms_stats, pd, pd_pca, persisting_effs_grp, plot_corr):
    _ms_stats_grp = (
        ms_stats[ms_stats["condition"] == "PSI"]
        .drop(["session", "ms_opts"], axis=1)
        .set_index(["subject", "condition", "time"])
    )
    _groupings = ["coverage", "lifespan", "occurrence"]
    for _time in ["T2", "T3", "T4", "T5"]:
        _ms_stats_grp_ = pd.DataFrame()
        for _grp in _groupings:
            _pca_mat = pd_pca(
                _ms_stats_grp[
                    _ms_stats_grp.index.get_level_values("time") == _time
                ]
                .reset_index()
                .pivot(index="subject", columns="microstate", values=_grp),
                1,
                f"{_grp}_{_time}",
            )
            _ms_stats_grp_ = pd.concat([_ms_stats_grp_, _pca_mat], axis=1)
        _grp_corrs = pd.concat([persisting_effs_grp, _ms_stats_grp_], axis=1)
        _rename_pers = {
            "Attitudes about Life positive": "Att.Life+",
            "Attitudes about Self positive": "Att.Self+",
            "Mood Changes positive": "Mood+",
            "Relationships positive": "Relat+",
            "Behavioral Changes positive": "Behav+",
            "Spirituality positive": "Spirit+",
        }
        plot_corr(
            _grp_corrs,
            method="spearman",
            mask_upper=False,
            mask_pval=True,
            lines=[6],
            title=f"Persisting effects {_time} \n masked out non-significant ~ {MULTI_CORRECTION.upper()} correction",
            fname=f"persisting_effs_agg_{_time}",
            rename_cols=_rename_pers,
        )
    return


@app.cell
def _(experience_asc, experience_conc, pd):
    experience_asc_conc = pd.concat(
        [experience_asc, experience_conc["PSI_conc_T60"]], axis=1
    )
    experience_asc_conc = experience_asc_conc[
        experience_asc_conc["condition"] == "PSI"
    ].drop("condition", axis=1)
    return (experience_asc_conc,)


@app.cell
def _(
    MULTI_CORRECTION,
    PVAL_STARS,
    experience_asc_conc,
    ms_stats,
    np,
    pd,
    pg,
    plt,
    sns,
):
    for _time in np.unique(ms_stats["time"]):
        grand_r = pd.DataFrame()
        grand_annot = pd.DataFrame()
        for dv in ["occurrence", "lifespan", "coverage"]:
            df = ms_stats[ms_stats["condition"] == "PSI"][
                ["subject", "time", "microstate", dv]
            ]
            df = (
                df[df["time"] == _time]
                .pivot(index="subject", columns="microstate", values=dv)
                .add_prefix(f"{dv}_")
            )
            df = pd.concat(
                [experience_asc_conc.reset_index(), df.reset_index()], axis=1
            )
            columns = [
                ["AIA", "OSE", "VUS", "VWB"],
                [f"{dv}_A", f"{dv}_B", f"{dv}_C"],
            ]
            corrs = pg.pairwise_corr(
                df,
                columns,
                method="spearman",
                padjust=MULTI_CORRECTION,
                alternative="two-sided",
            )
            r_vals = corrs.pivot(index="X", columns="Y", values="r")
            corr_stars = (
                corrs.pivot(index="X", columns="Y", values="p_corr")
                .round(3)
                .map(
                    lambda x: "".join(
                        ["*" for st in PVAL_STARS.keys() if x < st]
                    )
                )
            )
            annot = r_vals.round(3).astype(str) + corr_stars
            grand_r = pd.concat([grand_r, r_vals], axis=1)
            grand_annot = pd.concat([grand_annot, annot], axis=1)
        plt.figure()
        sns.heatmap(
            grand_r,
            cmap="coolwarm",
            vmax=0.8,
            vmin=-0.8,
            annot=grand_annot,
            fmt="",
        )
        plt.ylabel("ACS")
        plt.xlabel("")
        plt.title(f"Filter: 1-40Hz; {_time} time")
        plt.show()
    return


if __name__ == "__main__":
    app.run()
