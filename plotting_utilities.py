""" Plotting functions for modeling analysis of project 8 in RU5389
base code by: Rasmus Bruckner
modified by: Johannes Lehnen
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from allinpy import cm2inch

def parameter_summary_group_old(
    parameters: pd.DataFrame,
    param_labels: list,
    grid_size: tuple,
    axis_labels: list | None = None,
) -> None:
    """Creates plots showing parameter values for two groups (0 and 1).

    Parameters
    ----------
    parameters : pd.DataFrame
        All parameters. Must contain column 'group' with levels 0 and 1.
    param_labels : list
        Labels (column names) to plot.
    grid_size : tuple
        Grid size for subplots (rows, cols).
    axis_labels : list, optional
        Y-axis labels.

    Returns
    -------
    None
    """

    fig_width = 15
    fig_height = 10

    plt.figure(figsize=cm2inch(fig_width, fig_height))

    for i, label in enumerate(param_labels):
        plt.subplot(grid_size[0], grid_size[1], i + 1)
        ax = plt.gca()

        # Boxplot per group
        sns.boxplot(
            x="group",
            y=label,
            data=parameters,
            hue="group",
            palette=["#3274A1", "#E1812C"],
            legend=False,
            notch=False,
            showfliers=False,
            linewidth=0.8,
            width=0.2,
            boxprops=dict(alpha=1),
            showcaps=False,
            ax=ax,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["control", "patient"])

        # Individual data points
        sns.swarmplot(
            x="group",
            y=label,
            data=parameters,
            color="gray",
            alpha=0.7,
            size=3,
            ax=ax,
        )

        # Two-sample t-test between groups
        g0 = parameters.loc[parameters["group"] == 0, label]
        g1 = parameters.loc[parameters["group"] == 1, label]

        if len(g0) > 0 and len(g1) > 0:
            ttest_result = stats.ttest_ind(g0, g1, nan_policy="omit")
            pval = np.round(ttest_result.pvalue, 3)
        else:
            pval = np.nan

        if axis_labels is None:
            plt.ylabel(label)
        else:
            plt.ylabel(axis_labels[i])

        plt.xlabel("group")
        plt.ylim(-0.75, 2)
        plt.title(f"p = {pval}")
        sns.despine()

    plt.tight_layout()

def parameter_summary_group(
    parameters: pd.DataFrame,
    param_labels: list,
    grid_size: tuple,
    axis_labels: list | None = None,
) -> None:
    """
    Creates violin plots with individual data points and Welch's t-test
    for two groups (0 = HC, 1 = AD) for each parameter.

    Parameters
    ----------
    parameters : pd.DataFrame
        One row per subject. Must contain column 'group' with levels 0 and 1.
    param_labels : list
        Column names in parameters to plot.
    grid_size : tuple
        Grid size for subplots (rows, cols).
    axis_labels : list, optional
        Y-axis labels; defaults to param_labels if not provided.
    """
    from scipy import stats

    #colors = {0: "#4C72B0", 1: "#DD8452"}
    colors = {0: "#3274A1", 1: "#E1812C"}
    group_labels = {0: "HC", 1: "AD"}
    rng = np.random.default_rng(seed=42)

    n_plots = len(param_labels)
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(4 * grid_size[1], 5 * grid_size[0]))
    axes = np.array(axes).flatten()  # works for both 1D and 2D grids

    for i, label in enumerate(param_labels):
        ax = axes[i]

        hc_vals = parameters.loc[parameters["group"] == 0, label].dropna().values
        ad_vals = parameters.loc[parameters["group"] == 1, label].dropna().values

        # --- Welch's two-sided t-test ---
        if len(hc_vals) > 1 and len(ad_vals) > 1:
            t_stat, p_val = stats.ttest_ind(hc_vals, ad_vals, equal_var=False)
        else:
            t_stat, p_val = np.nan, np.nan

        # --- Violin plot ---
        parts = ax.violinplot(
            [hc_vals, ad_vals],
            positions=[0, 1],
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for j, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(list(colors.values())[j])
            pc.set_alpha(0.4)
            pc.set_edgecolor("none")

        # --- Individual subject points (jittered) ---
        for j, (vals, color) in enumerate(zip([hc_vals, ad_vals], colors.values())):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(
                np.full(len(vals), j) + jitter,
                vals,
                color=color,
                alpha=0.7,
                s=25,
                zorder=3,
                linewidths=0,
            )

        # --- Group mean ± SEM ---
        for j, vals in enumerate([hc_vals, ad_vals]):
            ax.errorbar(
                j, np.mean(vals),
                yerr=stats.sem(vals),
                fmt="o",
                color="black",
                markersize=7,
                capsize=5,
                linewidth=2,
                zorder=5,
            )

        # --- Significance bar ---
        y_max = max(np.max(hc_vals), np.max(ad_vals))
        #y_max = 2.2
        y_bar = y_max * 1.07
        ax.plot([0, 0, 1, 1], [y_bar * 0.98, y_bar, y_bar, y_bar * 0.98],
                color="black", linewidth=1.2)

        if np.isnan(p_val):
            sig_label = "n.a."
        elif p_val < 0.001:
            sig_label = "***"
        elif p_val < 0.01:
            sig_label = "**"
        elif p_val < 0.05:
            sig_label = "*"
        else:
            sig_label = f"n.s.\nt = {t_stat:.2f}, p = {p_val:.3f}"

        ax.text(0.5, y_bar * 1.01, sig_label, ha="center", va="bottom", fontsize=12)

        # --- Formatting ---
        ax.set_xticks([0, 1])
        ax.set_xticklabels([group_labels[0], group_labels[1]], fontsize=12)
        ax.set_ylabel(axis_labels[i] if axis_labels is not None else label, fontsize=12)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.75, 2.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("")

        # Print stats to console
        print(f"\n{label}:")
        print(f"  Welch's t-test: t = {t_stat:.3f}, p = {p_val:.4f}")
        print(f"  HC: n={len(hc_vals)}, mean={np.mean(hc_vals):.3f}, SEM={stats.sem(hc_vals):.3f}")
        print(f"  AD: n={len(ad_vals)}, mean={np.mean(ad_vals):.3f}, SEM={stats.sem(ad_vals):.3f}")

    # Hide any unused subplots
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

def est_err_block_group(allsub: pd.DataFrame):
    """
    Plot mean estimation error per block with SEM for each group.

    Required columns in allsub:
        - subj_num
        - block
        - e_t
        - group (0 = HC, 1 = AD)
    """

    # Aggregate mean and SEM per block and group
    summary = (
        allsub
        .groupby(["group", "block"])["e_t"]
        .agg(mean="mean", sem="sem")
        .reset_index()
    )

    # Labels for groups
    group_labels = {0: "HC", 1: "AD"}

    # Plot
    plt.figure()
    for g, df_g in summary.groupby("group"):
        plt.errorbar(
            df_g["block"],
            df_g["mean"],
            yerr=df_g["sem"],
            marker="o",
            capsize=4,
            label=group_labels.get(g, f"Group {g}")
        )

    plt.xlabel("Block")
    plt.ylabel("Mean Estimation Error")
    plt.ylim(20, 35)
    plt.title("Estimation Error Across Blocks")
    plt.legend()
    plt.tight_layout()
    plt.show()

def est_err_group(allsub: pd.DataFrame):
    """
    Plot mean estimation error per subject as a violin plot with individual
    subject means and a two-sided Welch's t-test for group differences (HC vs AD).

    Required columns in allsub:
        - subj_num
        - e_t
        - group (0 = HC, 1 = AD)
    """
    from scipy import stats

    group_labels = {0: "HC", 1: "AD"}
    colors = {0: "#3274A1", 1: "#E1812C"}

    # --- Average e_t per subject first ---
    subj_means = (
        allsub
        .groupby(["subj_num", "group"])["e_t"]
        .mean()
        .reset_index()
        .rename(columns={"e_t": "mean_e_t"})
    )

    hc_vals = subj_means.loc[subj_means["group"] == 0, "mean_e_t"].values
    ad_vals = subj_means.loc[subj_means["group"] == 1, "mean_e_t"].values

    # --- Two-sided Welch's t-test on subject means ---
    t_stat, p_val = stats.ttest_ind(hc_vals, ad_vals, equal_var=False)

    fig, ax = plt.subplots(figsize=(4, 5))

    # --- Violin plot ---
    parts = ax.violinplot(
        [hc_vals, ad_vals],
        positions=[0, 1],
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(list(colors.values())[i])
        pc.set_alpha(0.4)
        pc.set_edgecolor("none")

    # --- Individual subject means (jittered) ---
    rng = np.random.default_rng(seed=42)
    for i, (vals, color) in enumerate(zip([hc_vals, ad_vals], colors.values())):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            color=color,
            alpha=0.7,
            s=25,
            zorder=3,
            linewidths=0,
        )

    # --- Group mean ± SEM markers ---
    for i, vals in enumerate([hc_vals, ad_vals]):
        mean = np.mean(vals)
        sem = stats.sem(vals)
        ax.errorbar(
            i, mean,
            yerr=sem,
            fmt="o",
            color="black",
            markersize=7,
            capsize=5,
            linewidth=2,
            zorder=5,
        )

    # --- Significance bar ---
    y_max = max(np.max(hc_vals), np.max(ad_vals))
    y_bar = y_max * 1.07
    ax.plot([0, 0, 1, 1], [y_bar * 0.98, y_bar, y_bar, y_bar * 0.98],
            color="black", linewidth=1.2)

    if p_val < 0.001:
        sig_label = "***"
    elif p_val < 0.01:
        sig_label = "**"
    elif p_val < 0.05:
        sig_label = "*"
    else:
        sig_label = f"n.s.\nt = {t_stat:.2f}, p = {p_val:.3f}"

    ax.text(0.5, y_bar * 1.01, sig_label, ha="center", va="bottom", fontsize=12)

    # --- Formatting ---
    ax.set_xticks([0, 1])
    ax.set_xticklabels([group_labels[0], group_labels[1]], fontsize=12)
    ax.set_ylabel("Mean Estimation Error", fontsize=12)
    ax.set_xlim(-0.5, 1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Print t-test results
    print(f"Welch's two-sided t-test: t = {t_stat:.3f}, p = {p_val:.4f}")
    print(f"HC: n={len(hc_vals)}, mean={np.mean(hc_vals):.2f}, SEM={stats.sem(hc_vals):.2f}")
    print(f"AD: n={len(ad_vals)}, mean={np.mean(ad_vals):.2f}, SEM={stats.sem(ad_vals):.2f}")
