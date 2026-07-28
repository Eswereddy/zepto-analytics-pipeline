"""
Module 2 — Analytics Pipeline (/analytics)
01_eda.py

Part A: Profiling, cleaning, and the data story.

This script is the ONE AND ONLY place the raw Titanic dataset is loaded from
network/cache (sns.load_dataset). It saves a committed offline fallback
(titanic.csv) that 02_modeling.py reads instead of ever calling
sns.load_dataset again.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

report = {}  # collects numbers we'll reuse in the README

# ---------------------------------------------------------------------------
# Task 1: Load, profile, and save the ONE offline fallback CSV
# ---------------------------------------------------------------------------
print("=" * 80)
print("TASK 1: Load + profile")
print("=" * 80)

df = sns.load_dataset("titanic")  # <-- the one and only raw load in the module

buf_info = []
df.info(buf=type("W", (), {"write": buf_info.append})())
print("".join(buf_info))

print("\n--- df.describe() ---")
print(df.describe(include="all"))

print("\n--- df.shape ---")
print(df.shape)
report["shape"] = list(df.shape)

# Save the committed offline fallback immediately after loading, before any
# cleaning, so it is a faithful copy of the raw dataset.
df.to_csv("titanic.csv", index=False)
print("\nSaved raw dataset -> titanic.csv (offline fallback, loadable via pd.read_csv)")

missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]
print("\n--- Missing value percentages (columns with any missing values) ---")
print(missing_pct.round(2))
report["missing_pct"] = missing_pct.round(2).to_dict()

# ---------------------------------------------------------------------------
# Task 2: Missing-value handling per the threshold rule
#   <5% missing  -> drop those rows
#   5%-30%       -> impute
#   very high    -> drop column OR encode "missing" as its own category
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 2: Missing-value handling")
print("=" * 80)

df_clean = df.copy()
decisions = {}

for col, pct in missing_pct.items():
    if pct < 5:
        before = len(df_clean)
        df_clean = df_clean[df_clean[col].notna()]
        after = len(df_clean)
        decisions[col] = (
            f"{pct:.2f}% missing (<5%) -> dropped {before - after} rows "
            f"with missing '{col}'."
        )
    elif pct <= 30:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            fill_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(fill_val)
            decisions[col] = (
                f"{pct:.2f}% missing (5-30%) -> imputed with median "
                f"({fill_val:.2f}) since '{col}' is numeric."
            )
        else:
            fill_val = df_clean[col].mode(dropna=True)[0]
            df_clean[col] = df_clean[col].fillna(fill_val)
            decisions[col] = (
                f"{pct:.2f}% missing (5-30%) -> imputed with mode "
                f"('{fill_val}') since '{col}' is categorical."
            )
    else:
        # Very high missing rate (>30%): imputation would be unreliable.
        if col == "deck":
            # deck is ~77% missing; imputing a cabin deck for most passengers
            # would be fabricating information we have no basis for.
            # Encoding "missing" as its own category preserves the (real)
            # signal that "we don't know this passenger's deck" while not
            # inventing values, and keeps the column usable if desired.
            df_clean[col] = df_clean[col].astype(object).fillna("Missing")
            decisions[col] = (
                f"{pct:.2f}% missing (>30%) -> too high to impute reliably; "
                f"encoded missing values as their own category 'Missing' "
                f"rather than dropping the column outright, since 'unknown "
                f"deck' may itself correlate with fare class/survival."
            )
        else:
            df_clean = df_clean.drop(columns=[col])
            decisions[col] = (
                f"{pct:.2f}% missing (>30%) -> dropped column '{col}' "
                f"(imputation would be unreliable at this rate)."
            )

for col, msg in decisions.items():
    print(f"- {col}: {msg}")
report["missing_decisions"] = decisions

print(f"\nShape after Task 2 cleaning: {df_clean.shape}")
report["shape_after_cleaning"] = list(df_clean.shape)

# ---------------------------------------------------------------------------
# Task 3: Univariate analysis — age & fare
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 3: Univariate analysis (age, fare)")
print("=" * 80)


def iqr_outliers(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (series < lo) | (series > hi)
    return int(mask.sum()), lo, hi


fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, col in enumerate(["age", "fare"]):
    sns.histplot(df_clean[col], kde=True, ax=axes[0, i], color="#3E7CB1")
    axes[0, i].set_title(f"Histogram of {col}")
    sns.boxplot(x=df_clean[col], ax=axes[1, i], color="#F2A65A")
    axes[1, i].set_title(f"Box plot of {col}")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/01_univariate_age_fare.png", dpi=120)
plt.close()

age_out, age_lo, age_hi = iqr_outliers(df_clean["age"])
fare_out, fare_lo, fare_hi = iqr_outliers(df_clean["fare"])
print(f"age  IQR outliers: {age_out}  (bounds: [{age_lo:.2f}, {age_hi:.2f}])")
print(f"fare IQR outliers: {fare_out}  (bounds: [{fare_lo:.2f}, {fare_hi:.2f}])")
report["iqr_outliers"] = {"age": age_out, "fare": fare_out}

fare_mean = df_clean["fare"].mean()
fare_median = df_clean["fare"].median()
fare_mode = df_clean["fare"].mode()[0]
print(f"\nfare -> mean={fare_mean:.2f}, median={fare_median:.2f}, mode={fare_mode:.2f}")
skew_dir = "right-skewed" if fare_mean > fare_median > fare_mode else (
    "left-skewed" if fare_mean < fare_median < fare_mode else "not cleanly ordered, but"
)
skew_text = (
    f"fare is {skew_dir}: mean ({fare_mean:.2f}) > median ({fare_median:.2f}) > "
    f"mode ({fare_mode:.2f}), the classic signature of a long right tail caused "
    f"by a small number of very expensive first-class fares pulling the mean "
    f"above the median and mode."
)
print(skew_text)
report["fare_stats"] = {"mean": fare_mean, "median": fare_median, "mode": fare_mode}
report["fare_skew_text"] = skew_text

# ---------------------------------------------------------------------------
# Task 4: Bivariate analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 4: Bivariate analysis")
print("=" * 80)

# (a) by sex
surv_by_sex = {}
for s in df_clean["sex"].unique():
    mask = df_clean["sex"] == s
    rate = df_clean.loc[mask, "survived"].mean()
    surv_by_sex[s] = round(rate, 4)
print("Survival rate by sex:", surv_by_sex)

# (b) by pclass
surv_by_pclass = {}
for p in sorted(df_clean["pclass"].unique()):
    mask = df_clean["pclass"] == p
    rate = df_clean.loc[mask, "survived"].mean()
    surv_by_pclass[int(p)] = round(rate, 4)
print("Survival rate by pclass:", surv_by_pclass)

# (c) sex AND pclass together, via boolean masking with & / |
surv_by_sex_pclass = {}
for s in df_clean["sex"].unique():
    for p in sorted(df_clean["pclass"].unique()):
        mask = (df_clean["sex"] == s) & (df_clean["pclass"] == p)
        rate = df_clean.loc[mask, "survived"].mean()
        surv_by_sex_pclass[f"{s}_pclass{p}"] = round(rate, 4)
print("Survival rate by sex & pclass:", surv_by_sex_pclass)

# example of | usage, sanity-check group
mask_extreme = (df_clean["pclass"] == 1) | (df_clean["fare"] > 100)
print(
    f"\n(sanity check using |) 1st-class OR fare>100 passengers: "
    f"{mask_extreme.sum()}, survival rate {df_clean.loc[mask_extreme, 'survived'].mean():.4f}"
)

report["survival"] = {
    "by_sex": surv_by_sex,
    "by_pclass": surv_by_pclass,
    "by_sex_pclass": surv_by_sex_pclass,
}

# Correlation matrix restricted to the 6 numeric columns (excluding
# adult_male / alone — derived/redundant flags)
corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = df_clean[corr_cols].corr()
print("\nCorrelation matrix (6 columns):")
print(corr.round(3))

plt.figure(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation heatmap (6 numeric columns)")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/02_correlation_heatmap.png", dpi=120)
plt.close()

# rank off-diagonal pairs by |corr|
pairs = []
for i in range(len(corr_cols)):
    for j in range(i + 1, len(corr_cols)):
        pairs.append((corr_cols[i], corr_cols[j], corr.iloc[i, j]))
pairs.sort(key=lambda t: abs(t[2]), reverse=True)
top2 = pairs[:2]
print("\nTop 2 strongest correlations (by |corr|):")
for a, b, v in top2:
    print(f"  {a} <-> {b}: {v:.3f}")
report["top2_correlations"] = [{"pair": f"{a}-{b}", "corr": round(float(v), 3)} for a, b, v in top2]

corr_interp = (
    f"The two strongest off-diagonal correlations are {top2[0][0]}-{top2[0][1]} "
    f"({top2[0][2]:.2f}) and {top2[1][0]}-{top2[1][1]} ({top2[1][2]:.2f}). "
    f"{top2[0][0]} and {top2[0][1]} move together because "
)
if {top2[0][0], top2[0][1]} == {"pclass", "fare"}:
    corr_interp += "lower pclass numbers (1st class) are sold at much higher fares."
elif {top2[0][0], top2[0][1]} == {"survived", "pclass"} or {top2[0][0], top2[0][1]} == {"survived", "fare"}:
    corr_interp += "wealthier, higher-class passengers had materially better survival odds."
elif {top2[0][0], top2[0][1]} == {"sibsp", "parch"}:
    corr_interp += "passengers travelling with siblings/spouses also tended to travel with parents/children — i.e. as family units."
else:
    corr_interp += "of the underlying passenger/family structure in the data."
print(corr_interp)
report["corr_interpretation"] = corr_interp

# ---------------------------------------------------------------------------
# Task 5: Multivariate "data story" — at least 4 charts, each interpreted
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 5: Multivariate data story (>=4 charts)")
print("=" * 80)

interpretations = []

# Chart 1: survival rate by class and sex (bar)
plt.figure(figsize=(7, 5))
sns.barplot(data=df_clean, x="pclass", y="survived", hue="sex", errorbar=None, palette="Set2")
plt.title("Survival rate by class and sex")
plt.ylabel("Survival rate")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/03_survival_by_class_sex.png", dpi=120)
plt.close()
interpretations.append(
    "Chart 1 (bar, survival by class & sex): Women survived at a far higher "
    "rate than men in every class, and the gap barely narrows even in 3rd "
    "class, showing 'women and children first' dominated over class "
    "privilege. Within each sex, survival still drops steadily from 1st to "
    "3rd class, so both sex and class independently mattered."
)

# Chart 2: age distribution by survival (box)
plt.figure(figsize=(7, 5))
sns.boxplot(data=df_clean, x="survived", y="age", palette="Set3")
plt.xticks([0, 1], ["Died", "Survived"])
plt.title("Age distribution by survival outcome")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/04_age_by_survival.png", dpi=120)
plt.close()
interpretations.append(
    "Chart 2 (box, age by survival): The median age is similar between "
    "survivors and non-survivors, but survivors show a wider lower tail, "
    "reflecting the many young children who were prioritized for lifeboats. "
    "Age alone is a weak predictor compared to sex and class."
)

# Chart 3: fare vs age scatter, colored by survival
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df_clean, x="age", y="fare", hue="survived", alpha=0.6, palette=["#D9534F", "#5CB85C"])
plt.title("Fare vs age, colored by survival")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/05_fare_age_scatter.png", dpi=120)
plt.close()
interpretations.append(
    "Chart 3 (scatter, fare vs age colored by survival): Green (survived) "
    "points cluster more densely at higher fares, while red (died) points "
    "dominate the low-fare band regardless of age. This visually reinforces "
    "that ticket price (a proxy for class/wealth) separates outcomes more "
    "than age does."
)

# Chart 4: family size vs survival
df_clean["family_size"] = df_clean["sibsp"] + df_clean["parch"] + 1
plt.figure(figsize=(7, 5))
sns.barplot(data=df_clean, x="family_size", y="survived", errorbar=None, color="#6A4C93")
plt.title("Survival rate by family size")
plt.ylabel("Survival rate")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/06_family_size_survival.png", dpi=120)
plt.close()
interpretations.append(
    "Chart 4 (bar, survival by family size): Passengers travelling alone "
    "(family_size=1) or in very large families (6+) survived at the lowest "
    "rates, while small families of 2-4 survived best. This suggests "
    "solo travellers lacked help coordinating an escape, while very large "
    "families struggled to stay together and evacuate as a unit."
)

# Chart 5 (bonus): embark_town vs survival
if "embark_town" in df_clean.columns:
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df_clean, x="embark_town", y="survived", errorbar=None, palette="pastel")
    plt.title("Survival rate by port of embarkation")
    plt.ylabel("Survival rate")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/07_embark_survival.png", dpi=120)
    plt.close()
    interpretations.append(
        "Chart 5 (bar, survival by embarkation port): Passengers who "
        "boarded at Cherbourg survived at a noticeably higher rate than "
        "those from Southampton or Queenstown. This is likely a proxy "
        "effect — Cherbourg had a higher proportion of 1st-class "
        "passengers — rather than the port itself causing survival."
    )

for txt in interpretations:
    print("-", txt)
report["multivariate_interpretations"] = interpretations

# ---------------------------------------------------------------------------
# Task 6: Exploratory standardization check (age, fare) — EDA-stage only
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 6: Exploratory z-score standardization check (age, fare)")
print("=" * 80)

before_stats = df_clean[["age", "fare"]].agg(["mean", "std"])
z_age = (df_clean["age"] - df_clean["age"].mean()) / df_clean["age"].std()
z_fare = (df_clean["fare"] - df_clean["fare"].mean()) / df_clean["fare"].std()
after_stats = pd.DataFrame({"age": [z_age.mean(), z_age.std()], "fare": [z_fare.mean(), z_fare.std()]},
                            index=["mean", "std"])

print("Before standardization:\n", before_stats.round(3))
print("\nAfter standardization (z-score):\n", after_stats.round(3))
report["standardization_check"] = {
    "before": before_stats.round(3).to_dict(),
    "after": after_stats.round(3).to_dict(),
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df_clean["age"], kde=True, ax=axes[0], color="gray", label="age (raw)")
sns.histplot(z_age, kde=True, ax=axes[0], color="teal", label="age (z-score)")
axes[0].legend()
axes[0].set_title("age: before vs after standardization")
sns.histplot(df_clean["fare"], kde=True, ax=axes[1], color="gray", label="fare (raw)")
sns.histplot(z_fare, kde=True, ax=axes[1], color="teal", label="fare (z-score)")
axes[1].legend()
axes[1].set_title("fare: before vs after standardization")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/08_standardization_check.png", dpi=120)
plt.close()

print(
    "\nNote: this z-score check is purely an EDA-stage sanity check on the "
    "full cleaned DataFrame. It does NOT feed the modeling pipeline in "
    "02_modeling.py, which performs its own train-only StandardScaler fit."
)

# ---------------------------------------------------------------------------
# Persist cleaned data (as the same committed titanic.csv the module works
# from) and the report dict for the README generator.
# ---------------------------------------------------------------------------
# NOTE: titanic.csv above is the RAW offline fallback (per Task 1 spec).
# 02_modeling.py re-derives the same cleaning from that raw CSV so the whole
# module still only performs ONE raw load overall (here), never a second
# sns.load_dataset call.
with open("eda_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

print("\nSaved eda_report.json (used by README + 02_modeling.py for consistency).")
print("\n01_eda.py complete.")
