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
    # Microstate stats and their analysis

    Analysis of microstate stats such as average lifespan, coverage, frequency of occurrence, and transition probabilities.
    """)
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo

    import os
    import string

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pingouin as pg
    import seaborn as sns

    from src.helpers import PLOTS_ROOT, RESULTS_ROOT, make_dirs

    plt.style.use("default")
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.figsize"] = (20, 9)
    sns.set_context("notebook", font_scale=2)

    WORKING_DIR = os.path.join(RESULTS_ROOT, "20260325-final")
    PLOTTING_DIR = os.path.join(PLOTS_ROOT, "final")
    if not os.path.exists(PLOTTING_DIR):
        make_dirs(PLOTTING_DIR)

    # if True, will save all statistical tests as csv and all plots as eps
    SAVE_RESULTS = True
    PLOT_EXT = ".eps"
    # which correction for multiple comparisons should be used
    MULTI_CORRECTION = "fdr_bh"
    P_THRESH = 0.05
    return (
        MULTI_CORRECTION,
        PLOTTING_DIR,
        PLOT_EXT,
        P_THRESH,
        SAVE_RESULTS,
        WORKING_DIR,
        np,
        os,
        pd,
        pg,
        plt,
        sns,
        string,
    )


@app.cell
def _(WORKING_DIR, os, pd):
    # load data computed with `1-main_gfp_stats_and_ideal_no_mstates.py`
    df = pd.read_csv(os.path.join(WORKING_DIR, "ms_stats.csv"), index_col=[0])
    # divide session into PLA/PSI and time for plotting purposes
    df[["condition", "time"]] = df["session"].str.split("-", expand=True)
    # divide to filters
    df_220 = df[df["ms_opts"] == "(2.0, 20.0, 4)"]
    df_220 = df_220.sort_values(by=["condition", "time"])
    df_140 = df[df["ms_opts"] == "(1.0, 40.0, 3)"]
    df_140 = df_140.sort_values(by=["condition", "time"])
    return df_140, df_220


@app.cell
def _(MULTI_CORRECTION, P_THRESH, SAVE_RESULTS, WORKING_DIR, np, os, pd, pg):
    def compute_power_from_anova(anova_df, n_subjects, n_measurements):
        power_results = []
        for _, row in anova_df.iterrows():
            eta_sq = row["ng2"]
            eps = row["eps"] if "eps" in row else 1.0
            power = pg.power_rm_anova(
                eta_squared=eta_sq,
                m=n_measurements,
                n=n_subjects,
                epsilon=eps,
                alpha=0.05,
            )
            power_results.append(
                {
                    "Source": row["Source"],
                    "ng2": round(eta_sq, 4),
                    "eps": round(eps, 3),
                    "power": round(power, 3),
                }
            )
        return pd.DataFrame(power_results)

    def diff_between_microstates(df, dv, filt_str):
        df = df.copy()
        print("==========")
        print(f"Microstate differences: {dv}")
        print("==========")
        anova = pg.anova(data=df, dv=dv, between=["microstate"], detailed=False)
        if SAVE_RESULTS:
            anova.round(5).to_csv(
                os.path.join(
                    WORKING_DIR, f"{dv}_{filt_str}filt_microstate_anova.csv"
                )
            )
        print("ANOVA")
        print(anova)
        posthoc = pg.pairwise_tukey(
            data=df, dv=dv, between="microstate", effsize="cohen"
        )
        if SAVE_RESULTS:
            posthoc.round(5).to_csv(
                os.path.join(
                    WORKING_DIR, f"{dv}_{filt_str}filt_microstate_posthoc.csv"
                )
            )
        print("Tukey HSD")
        print(posthoc)

        print("==========")
        print(f"Microstate differences w.r.t condition: {dv}")
        print("==========")
        posthoc = pg.pairwise_tests(
            data=df,
            dv=dv,
            between=["microstate"],
            within=["condition"],
            padjust=MULTI_CORRECTION,
            effsize="cohen",
            subject="subject",
            within_first=False,
        )
        print(posthoc[posthoc["p_corr"] <= P_THRESH])

    def diff_between_time(df, dv, filt_str):
        df = df.copy()
        print("==========")
        print(f"Time differences: {dv}")
        print("==========")
        anova = pg.rm_anova(
            data=df, dv=dv, within=["time", "condition"], subject="subject"
        )
        if SAVE_RESULTS:
            anova.round(5).to_csv(
                os.path.join(WORKING_DIR, f"{dv}_{filt_str}filt_time_anova.csv")
            )
        print("RM ANOVA")
        print(anova)

        power_df = compute_power_from_anova(
            anova, n_subjects=df["subject"].nunique(), n_measurements=5
        )
        if SAVE_RESULTS:
            power_df.to_csv(
                os.path.join(WORKING_DIR, f"{dv}_{filt_str}filt_power.csv"),
                index=False,
            )
        print("Power analysis")
        print(power_df)

        posthoc = pg.pairwise_tests(
            data=df,
            dv=dv,
            within=["time", "condition"],
            subject="subject",
            effsize="cohen",
            padjust=MULTI_CORRECTION,
            return_desc=True,
        )
        if SAVE_RESULTS:
            posthoc.round(5).to_csv(
                os.path.join(
                    WORKING_DIR, f"{dv}_{filt_str}filt_time_cond_posthoc.csv"
                )
            )
        print(
            f"Pairwise 'time x condition' t-tests (showing only significant after {MULTI_CORRECTION.upper()} correction < 0.05)"
        )
        print(posthoc[posthoc["p_corr"] <= P_THRESH])

        posthoc = pg.pairwise_tests(
            data=df,
            dv=dv,
            within=["condition", "time"],
            subject="subject",
            effsize="cohen",
            padjust=MULTI_CORRECTION,
            return_desc=True,
        )
        if SAVE_RESULTS:
            posthoc.round(5).to_csv(
                os.path.join(
                    WORKING_DIR, f"{dv}_{filt_str}filt_cond_time_posthoc.csv"
                )
            )
        print(
            f"Pairwise 'condition x time' t-tests (showing only significant after {MULTI_CORRECTION.upper()} correction < 0.05)"
        )
        print(posthoc[posthoc["p_corr"] <= P_THRESH])

    def diff_between_time_and_microstates(df, dv, filt_str):
        df = df.copy()
        print("==========")
        print(f"Time differences: {dv}")
        print("==========")
        for ms in np.unique(df["microstate"]):
            print("----------")
            print(f"microstate {ms}")
            print("----------")
            anova = pg.rm_anova(
                data=df[df["microstate"] == ms],
                dv=dv,
                within=["time", "condition"],
                subject="subject",
            )
            if SAVE_RESULTS:
                anova.round(5).to_csv(
                    os.path.join(
                        WORKING_DIR,
                        f"{dv}_microstate{ms}_{filt_str}filt_anova.csv",
                    )
                )
            print("RM ANOVA")
            print(anova)

            posthoc = pg.pairwise_tests(
                data=df[df["microstate"] == ms],
                dv=dv,
                within=["time", "condition"],
                subject="subject",
                effsize="cohen",
                padjust=MULTI_CORRECTION,
                return_desc=True,
            )
            if SAVE_RESULTS:
                posthoc.round(5).to_csv(
                    os.path.join(
                        WORKING_DIR,
                        f"{dv}_microstate{ms}_{filt_str}filt_time_cond_posthoc.csv",
                    )
                )
            print(
                f"Pairwise 'time x condition' t-tests (showing only significant after {MULTI_CORRECTION.upper()} correction < 0.05)"
            )
            print(posthoc[posthoc["p_corr"] <= P_THRESH])

            posthoc = pg.pairwise_tests(
                data=df[df["microstate"] == ms],
                dv=dv,
                within=["condition", "time"],
                subject="subject",
                effsize="cohen",
                padjust=MULTI_CORRECTION,
                return_desc=True,
            )
            if SAVE_RESULTS:
                posthoc.round(5).to_csv(
                    os.path.join(
                        WORKING_DIR,
                        f"{dv}_microstate{ms}_{filt_str}filt_cond_time_posthoc.csv",
                    )
                )
            print(
                f"Pairwise 'condition x time' t-tests (showing only significant after {MULTI_CORRECTION.upper()} correction < 0.05)"
            )
            print(posthoc[posthoc["p_corr"] <= P_THRESH])

    return (
        diff_between_microstates,
        diff_between_time,
        diff_between_time_and_microstates,
    )


@app.cell
def _(
    MULTI_CORRECTION,
    PLOTTING_DIR,
    PLOT_EXT,
    SAVE_RESULTS,
    np,
    os,
    pg,
    plt,
    sns,
    string,
):
    ## Functions for plotting and saving boxplots

    def _plot_ttest_signi(
        where, df_for_max, ax, plot_for="time", x1_base=-0.2, x2_base=0.2
    ):
        y = df_for_max.max() * 1.02
        h = df_for_max.max() * 0.05
        col = plt.rcParams["text.color"]
        for _, row in where.iterrows():
            if plot_for == "time":
                pos = int(row["time"][-1]) - 1
            elif plot_for == "microstate":
                pos = string.ascii_uppercase.index(row["microstate"])
            else:
                raise ValueError(f"Unkown for: {plot_for}")
            x1, x2 = x1_base + pos, x2_base + pos
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
            if row["p_corr"] < 0.001:
                text = "p < 0.001"
            else:
                text = f"p={row['p_corr']:.3f}"
            ax.text(
                (x1 + x2) * 0.5,
                y + h,
                text,
                ha="center",
                va="bottom",
                color=col,
                fontsize=18,
            )

    def plot_boxes(df, dv, filt_str, legend_pos, plot_signi=True):
        # individual "factors"
        _, axs = plt.subplots(1, 2, sharey=True, sharex=False)
        sns.boxplot(
            x="microstate",
            y=dv,
            data=df.copy(),
            hue="condition",
            ax=axs[0],
            legend=legend_pos == 1,
        )
        sns.stripplot(
            x="microstate",
            y=dv,
            data=df.copy(),
            hue="condition",
            ax=axs[0],
            dodge=True,
            size=4,
            alpha=0.5,
            legend=False,
            palette="dark:black",
        )
        sns.despine(trim=True, ax=axs[0])
        if plot_signi:
            sign = pg.pairwise_tests(
                data=df,
                dv=dv,
                between=["microstate"],
                within=["condition"],
                padjust=MULTI_CORRECTION,
                subject="subject",
                within_first=False,
            )
            # find significant
            where = sign[
                (sign["p_corr"] <= 0.05)
                & (sign["Contrast"] == "microstate * condition")
            ]
            _plot_ttest_signi(
                where,
                df[dv],
                ax=axs[0],
                plot_for="microstate",
                x1_base=-0.2,
                x2_base=0.2,
            )

        sns.boxplot(
            x="time",
            y=dv,
            data=df.copy(),
            hue="condition",
            ax=axs[1],
            legend=legend_pos == 2,
        )
        sns.stripplot(
            x="time",
            y=dv,
            data=df.copy(),
            hue="condition",
            ax=axs[1],
            dodge=True,
            size=4,
            alpha=0.5,
            legend=False,
            palette="dark:black",
        )
        sns.despine(trim=True, ax=axs[1])
        if plot_signi:
            sign = pg.pairwise_tests(
                data=df,
                dv=dv,
                within=["time", "condition"],
                subject="subject",
                padjust=MULTI_CORRECTION,
            )
            # find significant
            where = sign[
                (sign["p_corr"] <= 0.05)
                & (sign["Contrast"] == "time * condition")
            ]
            _plot_ttest_signi(
                where,
                df[dv],
                ax=axs[1],
                plot_for="time",
                x1_base=-0.2,
                x2_base=0.2,
            )

        if SAVE_RESULTS:
            plt.savefig(
                os.path.join(
                    PLOTTING_DIR,
                    f"{dv}_boxplot_factors_{filt_str}filt{PLOT_EXT}",
                ),
                bbox_inches="tight",
                transparent=True,
            )
        else:
            plt.show()

        # interaction
        _, axs = plt.subplots(
            1, len(np.unique(df["microstate"])), sharex=True, sharey=True
        )
        for i, ms in enumerate(np.unique(df["microstate"])):
            ax = axs[i]
            sns.boxplot(
                x="time",
                y=dv,
                data=df[df["microstate"] == ms],
                hue="condition",
                ax=ax,
                legend=legend_pos > 2,
            )
            sns.stripplot(
                x="time",
                y=dv,
                data=df[df["microstate"] == ms],
                hue="condition",
                ax=ax,
                dodge=True,
                size=4,
                alpha=0.5,
                legend=False,
                palette="dark:black",
            )
            ax.set_title(f"Microstate {ms}")
            sns.despine(trim=True, ax=ax)
            if plot_signi:
                sign = pg.pairwise_tests(
                    data=df[df["microstate"] == ms],
                    dv=dv,
                    within=["time", "condition"],
                    subject="subject",
                    padjust=MULTI_CORRECTION,
                )
                # find significant
                where = sign[
                    (sign["p_corr"] <= 0.05)
                    & (sign["Contrast"] == "time * condition")
                ]
                _plot_ttest_signi(
                    where,
                    df[df["microstate"] == ms][dv],
                    ax=ax,
                    plot_for="time",
                    x1_base=-0.2,
                    x2_base=0.2,
                )
            plt.tight_layout()
        if SAVE_RESULTS:
            plt.savefig(
                os.path.join(
                    PLOTTING_DIR,
                    f"{dv}_boxplot_interaction_{filt_str}filt{PLOT_EXT}",
                ),
                bbox_inches="tight",
                transparent=True,
            )
        else:
            plt.show()

    return (plot_boxes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explained variance
    """)
    return


@app.cell
def _(df_220, pg):
    # 2-20Hz
    pg.rm_anova(
        data=df_220.drop_duplicates(subset=["subject", "session"]),
        dv="var_GFP",
        within=["condition", "time"],
        subject="subject",
    )
    return


@app.cell
def _(df_140, pg):
    # 1-40Hz
    pg.rm_anova(
        data=df_140.drop_duplicates(subset=["subject", "session"]),
        dv="var_GFP",
        within=["condition", "time"],
        subject="subject",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Average lifespan ~ 2-20Hz filter ~ 4 microstates
    """)
    return


@app.cell
def _(df_220, plot_boxes):
    plot_boxes(df_220, "lifespan", "2-20", 1, plot_signi=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Differences between microstates
    """)
    return


@app.cell
def _(df_220, diff_between_microstates):
    diff_between_microstates(df_220, "lifespan", filt_str="2-20")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Differences between times
    """)
    return


@app.cell
def _(df_220, diff_between_time):
    diff_between_time(df_220, "lifespan", filt_str="2-20")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Significance of time / condition for each microstate
    """)
    return


@app.cell
def _(df_220, diff_between_time_and_microstates):
    diff_between_time_and_microstates(df_220, "lifespan", filt_str="2-20")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Coverage ~ 2-20Hz filter ~ 4 microstates
    """)
    return


@app.cell
def _(df_220, plot_boxes):
    plot_boxes(df_220, "coverage", "2-20", 1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Differences between microstates
    """)
    return


@app.cell
def _(df_220, diff_between_microstates):
    diff_between_microstates(df_220, "coverage", filt_str="2-20")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Differences between time
    """)
    return


@app.cell
def _(df_220, diff_between_time):
    diff_between_time(df_220, "coverage", filt_str="2-20")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Significance of time / condition for each microstate
    """)
    return


@app.cell
def _(df_220, diff_between_time_and_microstates):
    diff_between_time_and_microstates(df_220, "coverage", filt_str="2-20")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Frequency of occurrence ~ 2-20Hz filter ~ 4 microstates
    """)
    return


@app.cell
def _(df_220, plot_boxes):
    plot_boxes(df_220, "occurrence", "2-20", 1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Differences between microstates
    """)
    return


@app.cell
def _(df_220, diff_between_microstates):
    diff_between_microstates(df_220, "occurrence", filt_str="2-20")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Differences between time
    """)
    return


@app.cell
def _(df_220, diff_between_time):
    diff_between_time(df_220, "occurrence", filt_str="2-20")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Significance of time / condition for each microstate
    """)
    return


@app.cell
def _(df_220, diff_between_time_and_microstates):
    diff_between_time_and_microstates(df_220, "occurrence", filt_str="2-20")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Average lifespan ~ 1-40Hz filter ~ 3 microstates
    """)
    return


@app.cell
def _(df_140, plot_boxes):
    plot_boxes(df_140, "lifespan", "1-40", 2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Differences between microstates
    """)
    return


@app.cell
def _(df_140, diff_between_microstates):
    diff_between_microstates(df_140, "lifespan", filt_str="1-40")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Differences between time
    """)
    return


@app.cell
def _(df_140, diff_between_time):
    diff_between_time(df_140, "lifespan", filt_str="1-40")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Significance of time / condition for each microstate
    """)
    return


@app.cell
def _(df_140, diff_between_time_and_microstates):
    diff_between_time_and_microstates(df_140, "lifespan", filt_str="1-40")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Coverage ~ 1-40Hz filter ~ 3 microstates
    """)
    return


@app.cell
def _(df_140, plot_boxes):
    plot_boxes(df_140, "coverage", "1-40", 2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Differences between microstates
    """)
    return


@app.cell
def _(df_140, diff_between_microstates):
    diff_between_microstates(df_140, "coverage", filt_str="1-40")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Differences between time
    """)
    return


@app.cell
def _(df_140, diff_between_time):
    diff_between_time(df_140, "coverage", filt_str="1-40")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Significance of time / condition for each microstate
    """)
    return


@app.cell
def _(df_140, diff_between_time_and_microstates):
    diff_between_time_and_microstates(df_140, "coverage", filt_str="1-40")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Frequency of occurrence ~ 1-40Hz filter ~ 3 microstates
    """)
    return


@app.cell
def _(df_140, plot_boxes):
    plot_boxes(df_140, "occurrence", "1-40", 2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Differences between microstates
    """)
    return


@app.cell
def _(df_140, diff_between_microstates):
    diff_between_microstates(df_140, "occurrence", filt_str="1-40")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Differences between time
    """)
    return


@app.cell
def _(df_140, diff_between_time):
    diff_between_time(df_140, "occurrence", filt_str="1-40")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Significance of time / condition for each microstate
    """)
    return


@app.cell
def _(df_140, diff_between_time_and_microstates):
    diff_between_time_and_microstates(df_140, "occurrence", filt_str="1-40")
    return


if __name__ == "__main__":
    app.run()
