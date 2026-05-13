"""Simple plot of regression results."""

if __name__ == "__main__":

    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    from allinpy import latex_plt
    from rbmpy import parameter_summary

    import numpy as np
    import scipy.stats as stats
    import seaborn as sns
    from allinpy import cm2inch


    def parameter_summary_group(
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
                palette=["#4C72B0", "#DD8452"],
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
            plt.title(f"p = {pval}")
            sns.despine()

        plt.tight_layout()

    # Update matplotlib to use Latex and to change some defaults
    matplotlib = latex_plt(matplotlib)

    # Use preferred backend for Linux, or just take default
    try:
        matplotlib.use("Qt5Agg")
    except ImportError:
        pass

    # Load data
    model_2 = pd.read_pickle("for_data/regression_model_2_3_100_sp.pkl")

    behav_labels = [
        "beta_0",
        "beta_1",
        "beta_4",
        "beta_5",
        "beta_6",
        "omikron_0",
        "omikron_1"
    ]

    axis_labels = [
        "Intercept",
        "Fixed LR",
        "Adaptive LR",
        "Catch effect",
        "Noise condition",
        "Motor Noise",
        "Learning-Rate Noise",
    ]

    '''
    behav_labels = [
        "beta_0",
        "beta_1",
        "beta_4",
        "beta_5",
        "beta_6",
        "omikron_0",
        "omikron_1",
    ]

    axis_labels = [
        "Intercept",
        "Fixed LR",
        "Adaptive LR",
        "Catch effect",
        "Noise condition",
        "Motor Noise",
        "Learning-Rate Noise",
    ]
    '''

    grid_size = (3, 3)
    parameter_summary(model_2, behav_labels, grid_size, axis_labels=axis_labels)
    plt.savefig("C:/Users/te65luf/Documents/Nextcloud/1_laufend_FOR_Statistical Learning (HH)/Auswertung/Confetti_beh/data/RU0226/winning_regression.png", dpi=400)

    # Show plot
    plt.ioff()
    plt.show()

    # plot

    behav_labels = [
        "beta_1",
        "beta_4"
    ]

    axis_labels = [
        "Fixed LR",
        "Adaptive LR"
    ]

    grid_size = (1, 2)
    parameter_summary_group(model_2, behav_labels, grid_size, axis_labels=axis_labels)
    plt.savefig("C:/Users/te65luf/Documents/Nextcloud/1_laufend_FOR_Statistical Learning (HH)/Auswertung/Confetti_beh/data/RU0226/group_test.png", dpi=400)

    # Show plot
    plt.ioff()
    plt.show()

