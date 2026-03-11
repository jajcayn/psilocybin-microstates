# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, E501

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
    # Comparison: me vs Inga/Povilas

    Correlating my microstate stats with Inga/Povilas results.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    plt.rcParams["figure.figsize"] = (20, 9)
    plt.style.use("papers")
    plt.rcParams["legend.frameon"] = False

    return pd, sns


@app.cell
def _(pd):
    ms_inga = pd.read_csv("../results-inga-povilas/temporal_results.csv")
    ms_inga[["cond", "subject", "time", "bull"]] = ms_inga["DataSet"].str.split(
        "_", n=3, expand=True
    )
    ms_inga = ms_inga.drop(
        [
            "DataSet",
            "bull",
            "Subject",
            "Group",
            "Condition",
            "Template",
            "SortInfo",
        ],
        axis=1,
    )
    ms_inga["subject"] = ms_inga["subject"].astype(int)
    ms_inga["session"] = ms_inga["cond"] + "-" + ms_inga["time"]
    ms_inga
    return (ms_inga,)


@app.cell
def _(pd):
    ms_me = pd.read_csv(
        "../results_csv/microstates/ms_stats_run.csv", index_col=0
    )
    ms_me
    return (ms_me,)


@app.cell
def _(ms_inga, ms_me, pd, sns):
    var1 = (
        ms_inga[["session", "subject", "ExpVar"]]
        .rename(columns={"ExpVar": "I/P"})
        .set_index(["subject", "session"])
    )
    var2 = (
        ms_me[["session", "subject", "var_GFP"]]
        .drop_duplicates()
        .rename(columns={"var_GFP": "N"})
        .set_index(["subject", "session"])
    )
    var = (
        pd.merge(var1, var2, left_index=True, right_index=True)
        .reset_index()
        .melt(
            id_vars=["subject", "session"],
            var_name="who",
            value_name="Explained Var",
        )
    )
    sns.boxplot(data=var, x="session", y="Explained Var", hue="who")
    sns.despine()
    return (var,)


@app.cell
def _(sns, var):
    sns.lineplot(
        data=var.pivot(
            index=["session", "subject"], columns="who", values="Explained Var"
        )
        .groupby("session")
        .corr()
        .loc[(slice(None), "N"), "I/P"]
        .reset_index()
        .drop("who", axis=1)
        .rename(columns={"I/P": "Expl. Var: Pearson"}),
        x="session",
        y="Expl. Var: Pearson",
    )
    return


@app.cell
def _(ms_inga, ms_me, pd, sns):
    lif1 = (
        ms_inga[
            [
                "session",
                "subject",
                "Duration_1",
                "Duration_2",
                "Duration_3",
                "Duration_4",
            ]
        ]
        .melt(
            id_vars=["session", "subject"],
            var_name="microstate",
            value_name="lifespan",
        )
        .replace(
            {
                "Duration_1": "A",
                "Duration_2": "B",
                "Duration_3": "C",
                "Duration_4": "D",
            }
        )
        .set_index(["subject", "session", "microstate"])
        .sort_index()
        .rename(columns={"lifespan": "I/P"})
        * 1000.0
    )
    lif2 = (
        ms_me[["session", "subject", "microstate", "lifespan"]]
        .set_index(["subject", "session", "microstate"])
        .sort_index()
        .rename(columns={"lifespan": "N"})
    )
    lif = (
        pd.merge(lif1, lif2, left_index=True, right_index=True)
        .reset_index()
        .melt(
            id_vars=["subject", "session", "microstate"],
            var_name="who",
            value_name="lifespan [ms]",
        )
    )
    sns.catplot(
        data=lif,
        x="who",
        y="lifespan [ms]",
        hue="microstate",
        col="session",
        col_wrap=5,
        kind="box",
    )
    return (lif,)


@app.cell
def _(lif, sns):
    sns.lineplot(
        data=lif.pivot(
            index=["session", "microstate", "subject"],
            columns="who",
            values="lifespan [ms]",
        )
        .groupby(["session", "microstate"])
        .corr()
        .loc[(slice(None), slice(None), "N"), "I/P"]
        .reset_index()
        .drop("who", axis=1)
        .rename(columns={"I/P": "lifepsan: Pearson"}),
        x="session",
        y="lifepsan: Pearson",
        hue="microstate",
        markers=True,
        style="microstate",
    )
    return


@app.cell
def _(ms_inga, ms_me, pd, sns):
    occ1 = (
        ms_inga[
            [
                "session",
                "subject",
                "Occurrence_1",
                "Occurrence_2",
                "Occurrence_3",
                "Occurrence_4",
            ]
        ]
        .melt(
            id_vars=["session", "subject"],
            var_name="microstate",
            value_name="occurrence",
        )
        .replace(
            {
                "Occurrence_1": "A",
                "Occurrence_2": "B",
                "Occurrence_3": "C",
                "Occurrence_4": "D",
            }
        )
        .set_index(["subject", "session", "microstate"])
        .sort_index()
        .rename(columns={"occurrence": "I/P"})
    )
    occ2 = (
        ms_me[["session", "subject", "microstate", "occurrence"]]
        .set_index(["subject", "session", "microstate"])
        .sort_index()
        .rename(columns={"occurrence": "N"})
    )
    occ = (
        pd.merge(occ1, occ2, left_index=True, right_index=True)
        .reset_index()
        .melt(
            id_vars=["subject", "session", "microstate"],
            var_name="who",
            value_name="occurrence [1/s]",
        )
    )
    sns.catplot(
        data=occ,
        x="who",
        y="occurrence [1/s]",
        hue="microstate",
        col="session",
        col_wrap=5,
        kind="box",
    )
    return (occ,)


@app.cell
def _(occ, sns):
    sns.lineplot(
        data=occ.pivot(
            index=["session", "microstate", "subject"],
            columns="who",
            values="occurrence [1/s]",
        )
        .groupby(["session", "microstate"])
        .corr()
        .loc[(slice(None), slice(None), "N"), "I/P"]
        .reset_index()
        .drop("who", axis=1)
        .rename(columns={"I/P": "occurrence: Pearson"}),
        x="session",
        y="occurrence: Pearson",
        hue="microstate",
        markers=True,
        style="microstate",
    )
    return


@app.cell
def _(ms_inga, ms_me, pd, sns):
    cov1 = (
        ms_inga[
            [
                "session",
                "subject",
                "Contribution_1",
                "Contribution_2",
                "Contribution_3",
                "Contribution_4",
            ]
        ]
        .melt(
            id_vars=["session", "subject"],
            var_name="microstate",
            value_name="contribution",
        )
        .replace(
            {
                "Contribution_1": "A",
                "Contribution_2": "B",
                "Contribution_3": "C",
                "Contribution_4": "D",
            }
        )
        .set_index(["subject", "session", "microstate"])
        .sort_index()
        .rename(columns={"contribution": "I/P"})
    )
    cov2 = (
        ms_me[["session", "subject", "microstate", "coverage"]]
        .set_index(["subject", "session", "microstate"])
        .sort_index()
        .rename(columns={"coverage": "N"})
    )
    cov = (
        pd.merge(cov1, cov2, left_index=True, right_index=True)
        .reset_index()
        .melt(
            id_vars=["subject", "session", "microstate"],
            var_name="who",
            value_name="coverage [%]",
        )
    )
    sns.catplot(
        data=cov,
        x="who",
        y="coverage [%]",
        hue="microstate",
        col="session",
        col_wrap=5,
        kind="box",
    )
    return (cov,)


@app.cell
def _(cov, sns):
    sns.lineplot(
        data=cov.pivot(
            index=["session", "microstate", "subject"],
            columns="who",
            values="coverage [%]",
        )
        .groupby(["session", "microstate"])
        .corr()
        .loc[(slice(None), slice(None), "N"), "I/P"]
        .reset_index()
        .drop("who", axis=1)
        .rename(columns={"I/P": "coverage: Pearson"}),
        x="session",
        y="coverage: Pearson",
        hue="microstate",
        markers=True,
        style="microstate",
    )
    return


if __name__ == "__main__":
    app.run()
