# ========================== #
# author:      Martin KUKRÁL #
# last change: 2025-12-03    #
# -------------------------- #
# Python version    3.11.4   #
# itertools         built-in #
# matplotlib        3.7.2    #
# numpy             1.25.2   #
# pandas            2.1.0    #
# scikit-learn      1.3.0    #
# scipy             1.11.2   #
# ========================== #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit
from scipy.stats import ttest_ind

# set random seed:
np.random.seed(0)





def prediction_performance(rmse_path:str):
    """
    Function to generate Figure 3.
    """
    # load data:
    df = pd.read_csv(rmse_path)

    # prepare the plot:
    _, axs = plt.subplot_mosaic([["orig", "norm"]])

    # 15-MIN SCATTERPLOTS -----------------------------------------------------
    t15 = df[df["PREDHORIZON"] == 15]
    t15_nl = t15[t15["TYPE"] == "NL"]
    t15_ml = t15[t15["TYPE"] == "ML"]
    # RMSE scatter:
    axs["orig"].scatter(np.repeat(1, len(t15_nl)) + np.random.uniform(-0.15, 0.15, len(t15_nl)), t15_nl["RMSE"], color="lightsteelblue", marker="o", s=60, linewidth=0.25, edgecolor="white", label="non-learning")
    axs["orig"].scatter(np.repeat(1, len(t15_ml)) + np.random.uniform(-0.15, 0.15, len(t15_ml)), t15_ml["RMSE"], color="lightsteelblue", marker="^", s=75, linewidth=0.25, edgecolor="white", label="learning")
    # RMSE per minute scatter:
    axs["norm"].scatter(np.repeat(1, len(t15_nl))+np.random.uniform(-0.15, 0.15, len(t15_nl)), t15_nl["RMSE"]/15, color="thistle", marker="o", s=60, linewidth=0.25, edgecolor="white", label="non-learning")
    axs["norm"].scatter(np.repeat(1, len(t15_ml))+np.random.uniform(-0.15, 0.15, len(t15_ml)), t15_ml["RMSE"]/15, color="thistle", marker="^", s=75, linewidth=0.25, edgecolor="white", label="learning")

    # 30-MIN SCATTERPLOTS -----------------------------------------------------
    t30 = df[df["PREDHORIZON"] == 30]
    t30_nl = t30[t30["TYPE"] == "NL"]
    t30_ml = t30[t30["TYPE"] == "ML"]
    # RMSE scatter:
    axs["orig"].scatter(np.repeat(2, len(t30_nl))+np.random.uniform(-0.15, 0.15, len(t30_nl)), t30_nl["RMSE"], color="lightsteelblue", marker="o", s=60, linewidth=0.25, edgecolor="white")
    axs["orig"].scatter(np.repeat(2, len(t30_ml))+np.random.uniform(-0.15, 0.15, len(t30_ml)), t30_ml["RMSE"], color="lightsteelblue", marker="^", s=75, linewidth=0.25, edgecolor="white")
    # RMSE per minute scatter:
    axs["norm"].scatter(np.repeat(2, len(t30_nl))+np.random.uniform(-0.15, 0.15, len(t30_nl)), t30_nl["RMSE"]/30, color="thistle", marker="o", s=60, linewidth=0.25, edgecolor="white")
    axs["norm"].scatter(np.repeat(2, len(t30_ml))+np.random.uniform(-0.15, 0.15, len(t30_ml)), t30_ml["RMSE"]/30, color="thistle", marker="^", s=75, linewidth=0.25, edgecolor="white")

    # 45-MIN SCATTERPLOTS -----------------------------------------------------
    t45 = df[df["PREDHORIZON"] == 45]
    t45_nl = t45[t45["TYPE"] == "NL"]
    t45_ml = t45[t45["TYPE"] == "ML"]
    # RMSE scatter:
    axs["orig"].scatter(np.repeat(3, len(t45_nl))+np.random.uniform(-0.15, 0.15, len(t45_nl)), t45_nl["RMSE"], color="lightsteelblue", marker="o", s=60, linewidth=0.25, edgecolor="white")
    axs["orig"].scatter(np.repeat(3, len(t45_ml))+np.random.uniform(-0.15, 0.15, len(t45_ml)), t45_ml["RMSE"], color="lightsteelblue", marker="^", s=75, linewidth=0.25, edgecolor="white")
    # RMSE per minute scatter:
    axs["norm"].scatter(np.repeat(3, len(t45_nl))+np.random.uniform(-0.15, 0.15, len(t45_nl)), t45_nl["RMSE"]/45, color="thistle", marker="o", s=60, linewidth=0.25, edgecolor="white")
    axs["norm"].scatter(np.repeat(3, len(t45_ml))+np.random.uniform(-0.15, 0.15, len(t45_ml)), t45_ml["RMSE"]/45, color="thistle", marker="^", s=75, linewidth=0.25, edgecolor="white")
    
    # 60-MIN SCATTERPLOTS -----------------------------------------------------
    t60 = df[df["PREDHORIZON"] == 60]
    t60_nl = t60[t60["TYPE"] == "NL"]
    t60_ml = t60[t60["TYPE"] == "ML"]
    # RMSE scatter:
    axs["orig"].scatter(np.repeat(4, len(t60_nl))+np.random.uniform(-0.15, 0.15, len(t60_nl)), t60_nl["RMSE"], color="lightsteelblue", marker="o", s=60, linewidth=0.25, edgecolor="white")
    axs["orig"].scatter(np.repeat(4, len(t60_ml))+np.random.uniform(-0.15, 0.15, len(t60_ml)), t60_ml["RMSE"], color="lightsteelblue", marker="^", s=75, linewidth=0.25, edgecolor="white")
    # RMSE per minute scatter:
    axs["norm"].scatter(np.repeat(4, len(t60_nl))+np.random.uniform(-0.15, 0.15, len(t60_nl)), t60_nl["RMSE"]/60, color="thistle", marker="o", s=60, linewidth=0.25, edgecolor="white")
    axs["norm"].scatter(np.repeat(4, len(t60_ml))+np.random.uniform(-0.15, 0.15, len(t60_ml)), t60_ml["RMSE"]/60, color="thistle", marker="^", s=75, linewidth=0.25, edgecolor="white")

    # 120-MIN SCATTERPLOTS ----------------------------------------------------
    t120 = df[df["PREDHORIZON"] == 120]
    t120_nl = t120[t120["TYPE"] == "NL"]
    t120_ml = t120[t120["TYPE"] == "ML"]
    # RMSE scatter:
    axs["orig"].scatter(np.repeat(5, len(t120_nl))+np.random.uniform(-0.15, 0.15, len(t120_nl)), t120_nl["RMSE"], color="lightsteelblue", marker="o", s=60, linewidth=0.25, edgecolor="white")
    axs["orig"].scatter(np.repeat(5, len(t120_ml))+np.random.uniform(-0.15, 0.15, len(t120_ml)), t120_ml["RMSE"], color="lightsteelblue", marker="^", s=75, linewidth=0.25, edgecolor="white")
    # RMSE per minute scatter:
    axs["norm"].scatter(np.repeat(5, len(t120_nl))+np.random.uniform(-0.15, 0.15, len(t120_nl)), t120_nl["RMSE"]/120, color="thistle", marker="o", s=60, linewidth=0.25, edgecolor="white")
    axs["norm"].scatter(np.repeat(5, len(t120_ml))+np.random.uniform(-0.15, 0.15, len(t120_ml)), t120_ml["RMSE"]/120, color="thistle", marker="^", s=75, linewidth=0.25, edgecolor="white")

    # RMSE MEANS ±95% CIs -----------------------------------------------------
    # RMSE means:
    t15_mean = np.mean(t15["RMSE"])
    t30_mean = np.mean(t30["RMSE"])
    t45_mean = np.mean(t45["RMSE"])
    t60_mean = np.mean(t60["RMSE"])
    t120_mean = np.mean(t120["RMSE"])
    axs["orig"].plot([0.75, 1.25], [t15_mean, t15_mean], linewidth=3, color="royalblue")
    axs["orig"].plot([1.75, 2.25], [t30_mean, t30_mean], linewidth=3, color="royalblue")
    axs["orig"].plot([2.75, 3.25], [t45_mean, t45_mean], linewidth=3, color="royalblue")
    axs["orig"].plot([3.75, 4.25], [t60_mean, t60_mean], linewidth=3, color="royalblue")
    axs["orig"].plot([4.75, 5.25], [t120_mean, t120_mean], linewidth=3, color="royalblue")
    # RMSE 95% confidence intervals:
    t15_ci = 1.96*(np.std(t15["RMSE"])/np.sqrt(len(t15["RMSE"])))
    t30_ci = 1.96*(np.std(t30["RMSE"])/np.sqrt(len(t30["RMSE"])))
    t45_ci = 1.96*(np.std(t45["RMSE"])/np.sqrt(len(t45["RMSE"])))
    t60_ci = 1.96*(np.std(t60["RMSE"])/np.sqrt(len(t60["RMSE"])))
    t120_ci = 1.96*(np.std(t120["RMSE"])/np.sqrt(len(t120["RMSE"])))
    # RMSE error bars:
    means = [t15_mean, t30_mean, t45_mean, t60_mean, t120_mean]
    cis = [t15_ci, t30_ci, t45_ci, t60_ci, t120_ci]
    axs["orig"].errorbar(range(1, 6), means, cis, fmt=":o", color="royalblue", capsize=10, capthick=1.5, label="means with 95% CIs")

    # RMSE PER MINUTE MEANS ±95% CIs ------------------------------------------
    t15_ = t15["RMSE"]/15
    t30_ = t30["RMSE"]/30
    t45_ = t45["RMSE"]/45
    t60_ = t60["RMSE"]/60
    t120_ = t120["RMSE"]/120
    # RMSE per minute means:
    t15_mean_ = np.mean(t15_)
    t30_mean_ = np.mean(t30_)
    t45_mean_ = np.mean(t45_)
    t60_mean_ = np.mean(t60_)
    t120_mean_ = np.mean(t120_)
    axs["norm"].plot([0.75, 1.25], [t15_mean_, t15_mean_], linewidth=3, color="mediumpurple")
    axs["norm"].plot([1.75, 2.25], [t30_mean_, t30_mean_], linewidth=3, color="mediumpurple")
    axs["norm"].plot([2.75, 3.25], [t45_mean_, t45_mean_], linewidth=3, color="mediumpurple")
    axs["norm"].plot([3.75, 4.25], [t60_mean_, t60_mean_], linewidth=3, color="mediumpurple")
    axs["norm"].plot([4.75, 5.25], [t120_mean_, t120_mean_], linewidth=3, color="mediumpurple")
    # RMSE per minute 95% confidence intervals:
    t15_ci_ = 1.96*(np.std(t15_)/np.sqrt(len(t15_)))
    t30_ci_ = 1.96*(np.std(t30_)/np.sqrt(len(t30_)))
    t45_ci_ = 1.96*(np.std(t45_)/np.sqrt(len(t45_)))
    t60_ci_ = 1.96*(np.std(t60_)/np.sqrt(len(t60_)))
    t120_ci_ = 1.96*(np.std(t120_)/np.sqrt(len(t120_)))
    # RMSE error bars:
    means_ = [t15_mean_, t30_mean_, t45_mean_, t60_mean_, t120_mean_]
    cis_ = [t15_ci_, t30_ci_, t45_ci_, t60_ci_, t120_ci_]
    axs["norm"].errorbar(range(1, 6), means_, cis_, fmt=":o", color="mediumpurple", capsize=10, capthick=1.5, label="means with 95% CIs")

    # PRINT RESULTS -----------------------------------------------------------
    predhorizons = [15, 30, 45, 60, 120]
    # RMSE:
    print("\n=== RMSE ================================================")
    for i, ph in enumerate(predhorizons):
        print(f"PH{ph}: {means[i]} (± {cis[i]})")
    # RMSE per minute:
    print("\n=== RMSE PER MINUTE =====================================")
    for i, ph in enumerate(predhorizons):
        print(f"PH{ph}: {means_[i]} (± {cis_[i]})")

    # FINALIZE THE PLOT -------------------------------------------------------
    axs["orig"].set_ylim(bottom=0, top=50)
    axs["orig"].set_xticks([1, 2, 3, 4, 5], ["15", "30", "45", "60", "120"])
    axs["orig"].set_xlabel("PREDICTION HORIZON [min]")
    axs["orig"].set_ylabel("RMSE [mg/dl]")
    axs["orig"].set_title("PREDICTION ERROR")
    axs["orig"].legend()
    axs["norm"].set_ylim(bottom=0, top=0.95)
    axs["norm"].set_xticks([1, 2, 3, 4, 5], ["15", "30", "45", "60", "120"])
    axs["norm"].set_xlabel("PREDICTION HORIZON [min]")
    axs["norm"].set_ylabel("RMSE PER MINUTE [mg⋅min/dl]")
    axs["norm"].set_title("PREDICTION ERROR PER MINUTE")
    axs["norm"].legend()
    axs["norm"].yaxis.set_label_position("right")
    axs["norm"].yaxis.tick_right()
    plt.subplots_adjust(wspace=0.035)
    plt.show()



def regressions(rmse_path:str):
    """
    Function to generate Figure 5.
    """
    # load and prepare data:
    df = pd.read_csv(rmse_path)
    t15 = df[df["PREDHORIZON"] == 15]["RMSE"]/15
    t30 = df[df["PREDHORIZON"] == 30]["RMSE"]/30
    t45 = df[df["PREDHORIZON"] == 45]["RMSE"]/45
    t60 = df[df["PREDHORIZON"] == 60]["RMSE"]/60
    t120 = df[df["PREDHORIZON"] == 120]["RMSE"]/120

    # prepare data points to fit:
    x = np.concatenate((np.repeat(15, len(t15)), np.repeat(30, len(t30)), np.repeat(45, len(t45)), np.repeat(60, len(t60)), np.repeat(120, len(t120)))).reshape(-1, 1)
    y = np.concatenate((t15, t30, t45, t60, t120))

    # prepare polynomial x:
    poly_features = PolynomialFeatures(degree=2)
    x_poly = poly_features.fit_transform(x)

    # linear regression:
    lin = LinearRegression()
    lin_fit = lin.fit(x, y)

    # quadratic regression:
    qdr = LinearRegression()
    qdr_fit = qdr.fit(x_poly, y)

    # REGRESSION CURVES PLOT --------------------------------------------------
    # means:
    x_means = np.array([15, 30, 45, 60, 120]).reshape(-1, 1)
    means = [np.mean(t15), np.mean(t30), np.mean(t45), np.mean(t60), np.mean(t120)]
    # 95% confidence intervals:
    t15_ci = 1.96*(np.std(t15)/np.sqrt(len(t15)))
    t30_ci = 1.96*(np.std(t30)/np.sqrt(len(t30)))
    t45_ci = 1.96*(np.std(t45)/np.sqrt(len(t45)))
    t60_ci = 1.96*(np.std(t60)/np.sqrt(len(t60)))
    t120_ci = 1.96*(np.std(t120)/np.sqrt(len(t120)))
    cis = [t15_ci, t30_ci, t45_ci, t60_ci, t120_ci]
    # compute the full regression curves:
    x_ = np.linspace(0, 130, 1000)
    y_ = lin_fit.predict(np.linspace(0, 130, 1000).reshape(-1, 1))
    y_poly = qdr_fit.predict(poly_features.transform(np.linspace(0, 130, 1000).reshape(-1, 1)))
    plt.plot(x_, y_, color="darkorange", label="linear regression")
    plt.plot(x_, y_poly, color="deeppink", label="quadratic regression")
    plt.errorbar(x_means, means, cis, fmt="o:", color="mediumpurple", capsize=10, capthick=1.5, alpha=0.25, label="means with 95% CIs")
    plt.xlabel("time [min]")
    plt.ylabel("RMSE PER MINUTE [mg⋅min/dl]")
    plt.legend()
    plt.show()



def compare_algorithms(algs_path:str):
    """
    Function to generate Figure 6.
    """
    # load and prepare data:
    df = pd.read_csv(algs_path)
    subjects = [df[df["SUBJECT"] == subj] for subj in np.unique(df["SUBJECT"])]
    time = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

    # prepare plot:
    _, axs = plt.subplot_mosaic([["shift", "pattern", "bglp"]], sharey=True, sharex=True)
    
    # plot individual patients
    for subj in subjects:
        axs["shift"].plot(time, subj["RMSE_SHIFT"], "o-", color="indianred", alpha=0.1, label="patients")
        axs["pattern"].plot(time, subj["RMSE_PATTERN"], "o-", color="dodgerblue", alpha=0.1, label="patients")
        axs["bglp"].plot(time, subj["RMSE_BGLP"], "o-", color="green", alpha=0.1, label="patients")

    # plot averages:
    phs = [df[df["PREDHORIZON"] == pd] for pd in np.unique(df["PREDHORIZON"])]
    axs["shift"].plot(time, [np.mean(ph["RMSE_SHIFT"]) for ph in phs], color="indianred", linewidth=2, label="average")
    axs["pattern"].plot(time, [np.mean(ph["RMSE_PATTERN"]) for ph in phs], color="dodgerblue", linewidth=2, label="average")
    axs["bglp"].plot(time, [np.mean(ph["RMSE_BGLP"]) for ph in phs], color="green", linewidth=2, label="average")

    # FINDING OPTIMAL f(t) ----------------------------------------------------
    # define f(t):
    ft = lambda x, e_max, e_inhib: (e_max*x)/(e_inhib + x)
    # find the optimal parameters:
    x = np.array([[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] * 12]).flatten()
    params_shift, _ = curve_fit(ft, x, df["RMSE_SHIFT"])
    params_pattern, _ = curve_fit(ft, x, df["RMSE_PATTERN"])
    params_bglp, _ = curve_fit(ft, x, df["RMSE_BGLP"])
    # print the optimal parameters:
    print("\nshift: ", params_shift)
    print("pattern: ", params_pattern)
    print("bglp: ", params_bglp)
    # compute optimal f(t) curves:
    x_ = np.linspace(0, 60, 1000)
    y_shift = ft(x_, *params_shift)
    y_pattern = ft(x_, *params_pattern)
    y_bglp = ft(x_, *params_bglp)

    # plot the f(t) curves:
    axs["shift"].plot(x_, y_shift, color="black", linewidth=3, linestyle=":", label="$f(t)$ fit")
    axs["pattern"].plot(x_, y_pattern, color="black", linewidth=3, linestyle=":", label="$f(t)$ fit")
    axs["bglp"].plot(x_, y_bglp, color="black", linewidth=3, linestyle=":", label="$f(t)$ fit")

    # FINALIZE PLOT -----------------------------------------------------------
    # labels:
    axs["shift"].set_ylabel("RMSE from relative errors")
    axs["shift"].set_title("Time-Shift")
    axs["pattern"].set_title("Pattern Prediction")
    axs["bglp"].set_title("BGLP 2020 winner")
    axs["shift"].set_xlabel("time [min]")
    axs["pattern"].set_xlabel("time [min]")
    axs["bglp"].set_xlabel("time [min]")
    # legends:
    axs["shift"].legend()
    handles, labels = axs["shift"].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axs["shift"].legend(unique.values(), unique.keys())
    axs["pattern"].legend()
    handles, labels = axs["pattern"].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axs["pattern"].legend(unique.values(), unique.keys(), loc="upper left")
    axs["bglp"].legend()
    handles, labels = axs["bglp"].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axs["bglp"].legend(unique.values(), unique.keys())
    # final adjustments:
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()



def meal_pa_detection(meal_path:str, pa_path:str):
    """
    Function to generate Figure 7.
    """
    # load data:
    meals = pd.read_csv(meal_path)
    pa = pd.read_csv(pa_path)

    # means and 95% CIs:
    meals_means = [np.mean(meals["SENSITIVITY"]), np.mean(meals["FPR*"])]
    meals_ci_sen = 1.96*(np.std(meals["SENSITIVITY"]) / np.sqrt(len(meals)))
    meals_ci_fpr = 1.96*(np.std(meals["FPR*"]) / np.sqrt(len(meals)))
    pa_means = [np.mean(pa["SENSITIVITY"]), np.mean(pa["FPR"])]
    pa_ci_sen = 1.96*(np.std(pa["SENSITIVITY"])/np.sqrt(len(pa)))
    pa_ci_fpr = 1.96*(np.std(pa["FPR"])/np.sqrt(len(pa)))

    # print results:
    print("\n=== MEALS ===============================================")
    print(f"mean sensitivity: {meals_means[0]} (± {meals_ci_sen})")
    print(f"mean FPR*:        {meals_means[1]} (± {meals_ci_fpr})")
    print("=== PA ==================================================")
    print(f"mean sensitivity: {pa_means[0]} (± {pa_ci_sen})")
    print(f"mean FPR:         {pa_means[1]} (± {pa_ci_fpr})")

    # PLOT THE VALUES ---------------------------------------------------------
    # prepare plot:
    fig, axs = plt.subplot_mosaic([["meals", "pa"]])
    # plot confidence intervals:
    axs["meals"].fill_between([50, meals_means[0]-meals_ci_sen], y1=2*[meals_means[1]-meals_ci_fpr], y2=2*[meals_means[1]+meals_ci_fpr], color="blanchedalmond")
    axs["meals"].fill_between([meals_means[0]+meals_ci_sen, 100], y1=2*[meals_means[1]-meals_ci_fpr], y2=2*[meals_means[1]+meals_ci_fpr], color="blanchedalmond")
    axs["meals"].fill_between([meals_means[0]-meals_ci_sen, meals_means[0]+meals_ci_sen], y1=[0, 0], y2=[50, 50], color="blanchedalmond")
    axs["pa"].fill_between([50, pa_means[0]-pa_ci_sen], y1=2*[pa_means[1]-pa_ci_fpr], y2=2*[pa_means[1]+pa_ci_fpr], color="lightgoldenrodyellow")
    axs["pa"].fill_between([pa_means[0]+pa_ci_sen, 100], y1=2*[pa_means[1]-pa_ci_fpr], y2=2*[pa_means[1]+pa_ci_fpr], color="lightgoldenrodyellow")
    axs["pa"].fill_between([pa_means[0]-pa_ci_sen, pa_means[0]+pa_ci_sen], y1=[0, 0], y2=[50, 50], color="lightgoldenrodyellow")
    # plot scatter:
    meals_nl = meals[meals["TYPE"] == "NL"]
    meals_ml = meals[meals["TYPE"] == "ML"]
    pa_nl = pa[pa["TYPE"] == "NL"]
    pa_ml = pa[pa["TYPE"] == "ML"]
    axs["meals"].scatter(meals_nl["SENSITIVITY"], meals_nl["FPR*"], color="indianred", marker="o", s=60, linewidth=0.25, edgecolor="white", label="non-learning")
    axs["meals"].scatter(meals_ml["SENSITIVITY"], meals_ml["FPR*"], color="indianred", marker="^", s=75, linewidth=0.25, edgecolor="white", label="learning")
    axs["pa"].scatter(pa_nl["SENSITIVITY"], pa_nl["FPR"], color="seagreen", marker="o", s=60, linewidth=0.25, edgecolor="white", label="non-learning")
    axs["pa"].scatter(pa_ml["SENSITIVITY"], pa_ml["FPR"], color="seagreen", marker="^", s=75, linewidth=0.25, edgecolor="white", label="learning")
    # means:
    axs["meals"].axvline(meals_means[0], color="indianred", linestyle="--", linewidth=1.25, label="means with 95% CIs")
    axs["meals"].axhline(meals_means[1], color="indianred", linestyle="--", linewidth=1.25)
    axs["pa"].axvline(pa_means[0], color="seagreen", linestyle="--", linewidth=1.25, label="means with 95% CIs")
    axs["pa"].axhline(pa_means[1], color="seagreen", linestyle="--", linewidth=1.25)
    # labels, final settings:
    axs["meals"].set_title("MEAL DETECTION")
    axs["meals"].set_xlabel("SENSITIVITY [%]")
    axs["meals"].set_ylabel("FPR* [%]")
    axs["meals"].set_xlim(50, 100)
    axs["meals"].set_ylim(0, 50)
    axs["meals"].legend()
    axs["pa"].set_title("PHYSICAL ACTIVITY DETECTION")
    axs["pa"].set_xlabel("SENSITIVITY [%]")
    axs["pa"].set_ylabel("FPR [%]")
    axs["pa"].set_xlim(50, 100)
    axs["pa"].set_ylim(0, 50)
    axs["pa"].yaxis.set_label_position("right")
    axs["pa"].yaxis.tick_right()
    axs["pa"].legend()
    plt.subplots_adjust(wspace=0.035)
    plt.show()  



def statistical_tests(rmse_path:str):
    """
    Function to compute pairwise Welch's t-tests (values for Figure 4).
    """
    # load data:
    df = pd.read_csv(rmse_path)

    # get all combinations of prediction horizons:
    pairs = list(combinations([15, 30, 45, 60, 120], 2))

    # COMPUTE THE P-VALUES ----------------------------------------------------
    pvals = []
    pvals_ = []
    for pair in pairs:
        # RMSE:
        ph1 = df[df["PREDHORIZON"] == pair[0]]["RMSE"]
        ph2 = df[df["PREDHORIZON"] == pair[1]]["RMSE"]
        _, p = ttest_ind(ph1, ph2, equal_var=False)
        pvals.append(p)
        # RMSE per minute:
        ph1_ = ph1/pair[0]
        ph2_ = ph2/pair[1]
        _, p_ = ttest_ind(ph1_, ph2_, equal_var=False)
        pvals_.append(p_)

    # print the p-values:
    print("\n=== WELCH'S t-TEST (RMSE) ==================")
    for i, pair in enumerate(pairs):
        print(f"{pair}: {pvals[i]} ({0 if pvals[i] > 0.05 else 1})")
    print("\n=== WELCH'S t-TEST (RMSE PER MINUTE) =======")
    for i, pair in enumerate(pairs):
        print(f"{pair}: {pvals_[i]} ({0 if pvals_[i] > 0.05 else 1})")





if __name__ == "__main__":
    prediction_performance("prediction.csv")              # Figure 3
    regressions("prediction.csv")                         # Figure 5
    compare_algorithms("algorithms.csv")                  # Figure 6
    meal_pa_detection("meal.csv", "physicalactivity.csv") # Figure 7
    statistical_tests("prediction.csv")                   # values for Figure 4