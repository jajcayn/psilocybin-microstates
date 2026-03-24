# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202

import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pingouin as pg
    import seaborn as sns

    from src.helpers import PLOTS_ROOT, RESULTS_ROOT, make_dirs

    plt.style.use("default")
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.figsize"] = (20, 9)
    sns.set_context("notebook", font_scale=1.6)

    WORKING_DIR = os.path.join(RESULTS_ROOT, "gfp_and_no_mstates")
    PLOTTING_DIR = os.path.join(PLOTS_ROOT, "new")
    make_dirs(PLOTTING_DIR)

    SAVE_RESULTS = True
    PLOT_EXT = ".eps"
    MULTI_CORRECTION = "fdr_bh"
    return (
        MULTI_CORRECTION,
        PLOTTING_DIR,
        PLOT_EXT,
        SAVE_RESULTS,
        WORKING_DIR,
        np,
        os,
        pd,
        pg,
        plt,
        sns,
    )


@app.cell
def _(np, pd, plt):
    # (best_val_fn, argbest_fn) — min-better vs max-better
    _score_fns = {
        "PM variance total": (
            lambda x: x.min(axis=1),
            lambda x: x.argmin(axis=1),
        ),
        "PM variance GFP": (
            lambda x: x.min(axis=1),
            lambda x: x.argmin(axis=1),
        ),
        "Davies-Bouldin": (lambda x: x.min(axis=1), lambda x: x.argmin(axis=1)),
        "Dunn": (lambda x: x.max(axis=1), lambda x: x.argmax(axis=1)),
        "Silhouette": (lambda x: x.max(axis=1), lambda x: x.argmax(axis=1)),
        "Calinski-Harabasz": (
            lambda x: x.max(axis=1),
            lambda x: x.argmax(axis=1),
        ),
    }

    def get_min(grp):
        return grp.set_index("# states")["PM variance total"].idxmin()

    def plot_ttest_signi(where, df_for_max, ax, x1_base=-0.2, x2_base=0.2):
        y = df_for_max.max() * 1.02
        h = df_for_max.max() * 0.05
        col = plt.rcParams["text.color"]
        for _, row in where.iterrows():
            pos = int(row["time"][-1]) - 1
            x1, x2 = x1_base + pos, x2_base + pos
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
            text = (
                "p < 0.001"
                if row["p_corr"] < 0.001
                else f"p={row['p_corr']:.3f}"
            )
            ax.text(
                (x1 + x2) * 0.5,
                y + h,
                text,
                ha="center",
                va="bottom",
                color=col,
            )

    def compute_ideal_ms(df_ns: pd.DataFrame) -> pd.DataFrame:
        no_states = df_ns.index.unique(level=-1).values
        rows = []
        for measure in df_ns.columns[3:]:
            for session, df_ in df_ns.groupby(level=[1, 2]):
                df_plot = df_[measure].reset_index()
                values_corr = (
                    df_plot[df_plot["method"] == "corr"]
                    .pivot(index="subject", columns="# states", values=measure)
                    .values
                )
                values_gmd = (
                    df_plot[df_plot["method"] == "GMD"]
                    .pivot(index="subject", columns="# states", values=measure)
                    .values
                )
                session_str = "-".join(session)
                argbest = _score_fns[measure][1]
                rows.extend(
                    [
                        {
                            "method": "corr",
                            "session": session_str,
                            "ideal #": np.median(
                                no_states[argbest(values_corr)]
                            ),
                            "measure": measure,
                        },
                        {
                            "method": "GMD",
                            "session": session_str,
                            "ideal #": np.median(
                                no_states[argbest(values_gmd)]
                            ),
                            "measure": measure,
                        },
                    ]
                )
        result = pd.DataFrame(rows)
        result[["condition", "time"]] = result["session"].str.split(
            "-", expand=True
        )
        return result

    return compute_ideal_ms, get_min, plot_ttest_signi


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GFP stats and ideal number of microstates

    Analysis of number GFP peaks and ideal number of canonical microstates.
    """)
    return


@app.cell
def _(WORKING_DIR, os, pd):
    # load data computed with `1-main_gfp_stats_and_ideal_no_mstates.py`
    df = pd.read_csv(
        os.path.join(WORKING_DIR, "gfp_peaks_var_test.csv"), index_col=[0]
    )
    df = df.drop_duplicates(subset=df.columns.difference(["filter"]))
    df[["condition", "time"]] = df["session"].str.split("-", expand=True)
    df_220 = df[df["filter"] == "2.0-20.0"]
    df_140 = df[df["filter"] == "1.0-40.0"]
    return df_140, df_220


@app.cell
def _(df_220):
    df_220
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Number of GFP peaks
    """)
    return


@app.cell
def _(
    MULTI_CORRECTION,
    PLOTTING_DIR,
    PLOT_EXT,
    SAVE_RESULTS,
    df_140,
    df_220,
    os,
    pg,
    plot_ttest_signi,
    plt,
    sns,
):
    _, axs = plt.subplots(1, 2, sharey=True)

    sns.boxplot(
        x="time", y="# GFP peaks", hue="condition", data=df_220, ax=axs[0]
    )
    axs[0].set_title("Filter: 2.0 - 20.0Hz", size=30)
    sign_220 = pg.pairwise_ttests(
        data=df_220,
        dv="# GFP peaks",
        within=["time", "condition"],
        subject="subject",
        padjust=MULTI_CORRECTION,
    )
    plot_ttest_signi(
        sign_220[
            (sign_220["p_corr"] <= 0.05)
            & (sign_220["Contrast"] == "time * condition")
        ],
        df_220["# GFP peaks"],
        ax=axs[0],
    )

    sns.boxplot(
        x="time", y="# GFP peaks", hue="condition", data=df_140, ax=axs[1]
    )
    axs[1].set_title("Filter: 1.0 - 40.0Hz", size=30)
    sign_140 = pg.pairwise_ttests(
        data=df_140,
        dv="# GFP peaks",
        within=["time", "condition"],
        subject="subject",
        padjust=MULTI_CORRECTION,
    )
    plot_ttest_signi(
        sign_140[
            (sign_140["p_corr"] <= 0.05)
            & (sign_140["Contrast"] == "time * condition")
        ],
        df_140["# GFP peaks"],
        ax=axs[1],
    )
    sns.despine(trim=True)
    if SAVE_RESULTS:
        plt.savefig(
            os.path.join(PLOTTING_DIR, f"num_gfp_boxplot{PLOT_EXT}"),
            bbox_inches="tight",
            transparent=True,
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ideal number of canonical microstates

    ### 2-20Hz filter
    """)
    return


@app.cell
def _(compute_ideal_ms, df_220):
    df_220_ns = df_220.set_index(
        ["method", "condition", "time", "subject", "# states"]
    ).sort_index()
    ideal_ms220 = compute_ideal_ms(df_220_ns)
    return (ideal_ms220,)


@app.cell
def _(ideal_ms220):
    ideal_ms220
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1-40Hz filter
    """)
    return


@app.cell
def _(compute_ideal_ms, df_140):
    df_140_ns = df_140.set_index(
        ["method", "condition", "time", "subject", "# states"]
    ).sort_index()
    ideal_ms140 = compute_ideal_ms(df_140_ns)
    return (ideal_ms140,)


@app.cell
def _(ideal_ms140):
    ideal_ms140
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Summary plot for ideal number of microstates
    """)
    return


@app.cell
def _(df_140, df_220, get_min, pd):
    ideal_220 = (
        df_220.groupby(["subject", "condition", "time", "method"])
        .apply(get_min, include_groups=False)
        .reset_index()
        .rename(columns={0: "ideal #"})
    )
    ideal_220["filter"] = "Filter: 2.0 - 20.0Hz"
    ideal_140 = (
        df_140.groupby(["subject", "condition", "time", "method"])
        .apply(get_min, include_groups=False)
        .reset_index()
        .rename(columns={0: "ideal #"})
    )
    ideal_140["filter"] = "Filter: 1.0 - 40.0Hz"
    ideal = pd.concat([ideal_220, ideal_140])
    ideal["ideal #"] = ideal["ideal #"].astype(int)
    return (ideal,)


@app.cell
def _(PLOTTING_DIR, PLOT_EXT, SAVE_RESULTS, ideal, os, plt, sns):
    g = sns.displot(
        ideal[ideal["method"] == "corr"],
        x="ideal #",
        col="time",
        row="filter",
        hue="condition",
        multiple="dodge",
        discrete=True,
        shrink=0.8,
        facet_kws=dict(margin_titles=True),
        height=5,
    )
    g.set_titles(col_template="", row_template="")
    g.figure.suptitle("Ideal # of microstates", size=35)
    g.figure.subplots_adjust(top=0.85)

    custom_yticks = [0, 2, 4, 6, 8, 10, 12]
    for ax in g.axes.flat:
        ax.set_yticks(custom_yticks)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for _i, ax in enumerate(g.axes[-1, :]):
        bbox = ax.get_position()
        g.figure.text(
            bbox.x0 + bbox.width / 2,
            bbox.y0 - 0.05,
            g.col_names[_i],
            ha="center",
            va="top",
            fontsize=25,
        )
    for _i, ax in enumerate(g.axes[:, 0]):
        ax.set_ylabel(g.row_names[_i], fontsize=25, rotation=90, labelpad=10)

    if SAVE_RESULTS:
        plt.savefig(
            os.path.join(PLOTTING_DIR, f"ideal_num_ms_summary{PLOT_EXT}"),
            bbox_inches="tight",
            transparent=True,
        )
    return


if __name__ == "__main__":
    app.run()
