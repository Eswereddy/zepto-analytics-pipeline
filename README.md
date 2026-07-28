# Module 2 — Analytics Pipeline (`/analytics`)

Titanic dataset, one cohesive pipeline: `01_eda.py` (profiling → cleaning →
data story) feeds `02_modeling.py` (classification + regression) through a
single committed `titanic.csv`. The raw dataset is loaded from
`sns.load_dataset('titanic')` **exactly once**, inside `01_eda.py`;
`02_modeling.py` reads `titanic.csv` and never re-fetches it.

Run order:
```
python3 01_eda.py       # profiles, cleans, saves titanic.csv, produces charts/eda_report.json
python3 02_modeling.py  # reads titanic.csv, trains/evaluates models, saves the pipeline
```

## Files

| File | Purpose |
|---|---|
| `01_eda.py` | Part A: profiling, cleaning, univariate/bivariate/multivariate EDA, standardization check |
| `02_modeling.py` | Part B: split, preprocessing pipeline, 3 classifiers, imbalance comparison, tuning, regression, save/reload pipeline |
| `titanic.csv` | The one committed offline fallback of the **raw** dataset (`pd.read_csv("titanic.csv")` works with no network) |
| `titanic_full_pipeline.joblib` | Full fitted pipeline (ColumnTransformer + tuned RandomForestClassifier) — usable end-to-end on raw new data |
| `eda_report.json` / `modeling_report.json` | Machine-readable copies of every number reported below |
| `charts/` | All 12 saved chart images |

---

## Part A — Profiling, cleaning, and the data story

### Task 1 — Profiling
Raw shape: **891 rows × 15 columns**. `df.info()` / `df.describe()` output is printed in full when `01_eda.py` runs.

Missing-value percentages (columns with any missing values):

| Column | % missing |
|---|---|
| deck | 77.22% |
| age | 19.87% |
| embarked | 0.22% |
| embark_town | 0.22% |

### Task 2 — Missing-value handling (threshold rule)

| Column | % missing | Rule | Decision |
|---|---|---|---|
| deck | 77.22% | >30% → drop or "missing" category | **Encoded as own category `"Missing"`** rather than dropped — "unknown deck" may itself correlate with class/survival, and dropping would discard 891 rows' worth of a whole feature for no benefit. |
| age | 19.87% | 5–30% → impute | **Median imputed** (28.00) — age is numeric and right-tailed enough that median is more robust than mean. |
| embarked | 0.22% | <5% → drop rows | **Dropped** (2 rows). |
| embark_town | 0.22% | <5% → drop rows | **Dropped** (0 additional rows — same 2 passengers). |

Shape after cleaning: **889 rows** (15 columns retained, `deck` recoded not dropped).

### Task 3 — Univariate analysis (age, fare)
![Univariate age/fare](charts/01_univariate_age_fare.png)

- **age**: 65 IQR outliers (bounds [2.50, 54.50]).
- **fare**: 114 IQR outliers (bounds [-26.76, 65.66]).
- **fare**: mean = 32.10, median = 14.45, mode = 8.05. Since mean > median > mode,
  **fare is right-skewed** — a long tail of expensive 1st-class fares pulls
  the mean well above the typical (median/mode) ticket price.

### Task 4 — Bivariate analysis

Survival rate by **sex**: female 0.740, male 0.189.
Survival rate by **pclass**: 1st 0.626, 2nd 0.473, 3rd 0.242.
Survival rate by **sex & pclass**:

| | 1st | 2nd | 3rd |
|---|---|---|---|
| female | 0.967 | 0.921 | 0.500 |
| male | 0.369 | 0.157 | 0.135 |

**Correlation heatmap** (6 numeric columns only: `survived, pclass, age, sibsp, parch, fare`; `adult_male`/`alone` excluded as derived/redundant flags):

![Correlation heatmap](charts/02_correlation_heatmap.png)

The two strongest off-diagonal correlations are **pclass–fare (−0.55)** and
**sibsp–parch (0.41)**. Pclass and fare move together because lower pclass
numbers (1st class) were sold at much higher fares — essentially the same
underlying variable measured two ways. Sibsp and parch correlate because
passengers travelling with siblings/spouses also tended to travel with
parents/children, i.e. they moved as family units.

### Task 5 — Multivariate data story (5 charts, each interpreted)

**Chart 1 — Survival by class and sex**
![Survival by class and sex](charts/03_survival_by_class_sex.png)
Women survived at a far higher rate than men in every class, and the gap
barely narrows even in 3rd class, showing "women and children first"
dominated over class privilege. Within each sex, survival still drops
steadily from 1st to 3rd class, so both sex and class independently
mattered.

**Chart 2 — Age by survival**
![Age by survival](charts/04_age_by_survival.png)
The median age is similar between survivors and non-survivors, but
survivors show a wider lower tail, reflecting the young children
prioritized for lifeboats. Age alone is a weak predictor compared to sex
and class.

**Chart 3 — Fare vs age, colored by survival**
![Fare vs age scatter](charts/05_fare_age_scatter.png)
Survivors cluster more densely at higher fares, while non-survivors
dominate the low-fare band regardless of age. This visually reinforces that
ticket price (a proxy for class/wealth) separates outcomes more than age
does.

**Chart 4 — Survival by family size**
![Family size survival](charts/06_family_size_survival.png)
Passengers travelling alone or in very large families (6+) survived at the
lowest rates, while small families of 2–4 survived best. Solo travellers
likely lacked help coordinating an escape, while very large families
struggled to stay together and evacuate as a unit.

**Chart 5 — Survival by embarkation port**
![Embark survival](charts/07_embark_survival.png)
Passengers who boarded at Cherbourg survived at a noticeably higher rate
than those from Southampton or Queenstown. This is likely a proxy effect —
Cherbourg had a higher proportion of 1st-class passengers — rather than the
port itself causing survival.

### Task 6 — Exploratory standardization check (EDA-stage only)

![Standardization check](charts/08_standardization_check.png)

| | age (before) | age (after) | fare (before) | fare (after) |
|---|---|---|---|---|
| mean | 29.315 | 0.0 | 32.097 | 0.0 |
| std | 12.985 | 1.0 | 49.698 | 1.0 |

This confirms the z-scored columns have (approximately) mean 0, std 1. This
check is purely exploratory — the modeling pipeline in `02_modeling.py`
performs its own **train-only** `StandardScaler` fit, independent of this
check.

---

## Part B — Predictive modeling

### Task 7 — Stratified split
Class balance of `survived`: **61.75% died / 38.25% survived** — imbalanced
enough that a plain random split risks train/test folds with meaningfully
different survival ratios by chance. `stratify=y` preserves the ~62/38 ratio
in both folds (train survival rate 0.3826, test 0.3820 — near-identical),
making test metrics a fair, low-variance estimate of real performance.

### Task 8 — Preprocessing (fit on train only)
A `ColumnTransformer` — numeric features (`pclass, age, sibsp, parch, fare`)
→ median-impute + `StandardScaler`; categorical features (`sex, embarked`)
→ most-frequent-impute + `OneHotEncoder` — wrapped in an sklearn `Pipeline`
with each classifier. `.fit()` is only ever called on `X_train`/`y_train`;
`X_test` only ever sees `.transform()`/`.predict()`.

### Task 9–10 — Three classifiers, full evaluation

**Decision tree visualization** (depth-limited to 3 levels for readability):
![Decision tree](charts/09_decision_tree.png)

**Confusion matrices:**
![Confusion matrices](charts/10_confusion_matrices.png)

**ROC curves:**
![ROC curves](charts/11_roc_curves.png)

**Classifier comparison table:**

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.809 | 0.783 | 0.691 | 0.734 | 0.861 |
| Decision Tree | 0.764 | 0.760 | 0.559 | 0.644 | 0.837 |
| Random Forest | 0.809 | 0.766 | 0.721 | 0.742 | 0.820 |

### Task 11 — Imbalance handling comparison (Logistic Regression)

| Strategy | Precision | Recall | F1 |
|---|---|---|---|
| Baseline (no handling) | 0.783 | 0.691 | 0.734 |
| `class_weight='balanced'` | 0.718 | 0.750 | 0.734 |
| SMOTE (train fold only) | 0.735 | 0.735 | 0.735 |

**Conclusion:** both `class_weight='balanced'` and SMOTE trade some
precision for higher recall on the minority ("survived") class versus the
baseline, since they force the model to pay more attention to survivors.
`class_weight='balanced'` gave the highest recall in this run, but all
three strategies land within ~0.001 F1 of each other — for this dataset the
imbalance (62/38) isn't severe enough to make one strategy a clear winner,
so the choice would come down to whether the deployment context values
recall (catching survivors) or precision more.

### Task 12 — GridSearchCV tuning (Random Forest)
Best params: `max_depth=None, max_features='sqrt', n_estimators=300`.
**OOB score: 0.807**, close to held-out test accuracy (0.803), indicating
the tuned model generalizes well rather than overfitting the training fold.

### Task 13 — Regression side-task (predict fare)
![Residual plot](charts/12_residual_plot.png)

| MAE | RMSE | R² | Adjusted R² |
|---|---|---|---|
| 21.099 | 41.702 | 0.348 | 0.321 |

**Heteroscedasticity:** yes — the residual plot shows a clear fan/funnel
shape, with residual spread roughly 4x larger for high-predicted-fare
passengers (std ≈ 57.2) than low-predicted-fare passengers (std ≈ 13.2).
This is consistent with fare's right-skewed, high-variance nature at the
top end (a handful of very expensive tickets are much harder to predict
precisely than the many cheap ones).

### Task 14 — Final model comparison & recommendation

**Classification metrics** (one scale) vs. **regression metrics** (a
different, non-comparable scale) — kept as separate column groups:

| Model | Accuracy | Precision | Recall | F1 | AUC | — | MAE | RMSE | R² | Adj. R² |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.809 | 0.783 | 0.691 | 0.734 | 0.861 | | | | | |
| Decision Tree | 0.764 | 0.760 | 0.559 | 0.644 | 0.837 | | | | | |
| Random Forest | 0.809 | 0.766 | 0.721 | 0.742 | 0.820 | | | | | |
| Fare Regression (Linear) | | | | | | | 21.099 | 41.702 | 0.348 | 0.321 |

**Recommendation:** We recommend deploying the **Random Forest** classifier.
It achieves the best balance of precision (0.766) and recall (0.721),
giving the top F1 score (0.742) among the three classifiers, with an AUC of
0.820 indicating strong ranking ability between survivors and
non-survivors. It's also more robust to the non-linear interactions seen in
the data story (e.g. the sex-by-class effect) than plain Logistic
Regression, without the overfitting risk of a single unpruned Decision
Tree. GridSearchCV tuning further confirmed a strong OOB score (0.807)
close to test performance, indicating the tuned model generalizes well.

### Task 15 — Saved pipeline
`titanic_full_pipeline.joblib` contains the **complete fitted pipeline**
(ColumnTransformer preprocessing + tuned `RandomForestClassifier` from
GridSearchCV) as a single object. `02_modeling.py` reloads it with
`joblib.load` and confirms its prediction on a raw, unpreprocessed test row
matches the original in-memory pipeline's prediction — verified passing.
