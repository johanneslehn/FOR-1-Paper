"""Simple plot of regression results."""

if __name__ == "__main__":

    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    from allinpy import latex_plt
    from rbmpy import parameter_summary
    from plotting_utilities import parameter_summary_group, est_err_group

    import numpy as np
    import scipy.stats as stats
    import seaborn as sns
    from allinpy import cm2inch

    # Update matplotlib to use Latex and to change some defaults
    matplotlib = latex_plt(matplotlib)

    # Use preferred backend for Linux, or just take default
    try:
        matplotlib.use("Qt5Agg")
    except ImportError:
        pass

    ## estimation error ##

    # Load data
    df_beh = pd.read_pickle("for_data/data_prepr.pkl")

    est_err_group(df_beh)
    plt.savefig("C:/Users/te65luf/Documents/Nextcloud/1_laufend_FOR_Statistical Learning (HH)/Auswertung/Confetti_beh/data/RU0526/est_err_group.png", dpi=400)

    # Show plot
    plt.ioff()
    plt.show()

    ## regression betas ##

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
    plt.savefig("C:/Users/te65luf/Documents/Nextcloud/1_laufend_FOR_Statistical Learning (HH)/Auswertung/Confetti_beh/data/RU0526/winning_regression.png", dpi=400)

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
    plt.savefig("C:/Users/te65luf/Documents/Nextcloud/1_laufend_FOR_Statistical Learning (HH)/Auswertung/Confetti_beh/data/RU0526/group_test.png", dpi=400)

    # Show plot
    plt.ioff()
    plt.show()

