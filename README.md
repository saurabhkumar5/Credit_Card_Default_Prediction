# Credit Card Default Prediction

A machine learning project that predicts whether a credit card client will miss their next payment, built as part of my MSc Artificial Intelligence coursework at the Berlin School of Business and Innovation.

The interesting part of this project isn't really the models — it's a hidden trap in the dataset that gave me a "perfect" 100% accuracy at first. Chasing down *why* that was happening turned out to be the most useful thing I learned here. More on that below.

---

## What this project is about

Banks lose a lot of money when cardholders default. If they can flag risky clients early, they can step in before things go wrong — adjust the credit limit, reach out, restructure, whatever. So the practical question is simple: **given a client's payment history and profile, how likely are they to default next month?**

I worked with a dataset of roughly 35,000 credit card clients covering six months (April to September 2005). Each row has demographic info, credit limit, billing amounts, and the repayment status for each month. The target column is `default.payment.next.month` (1 = defaulted, 0 = paid).

I built the whole thing as an end-to-end pipeline in a single Colab notebook, going from raw CSV all the way to model interpretation.

---

## The data leakage story (the main thing I'd want you to read)

When I first trained my models, every single one of them came back with ~100% accuracy. My initial reaction was to be happy about it — and then suspicious, because nothing in real credit risk is that easy.

I went back and looked at the correlation between each feature and the target. One feature, `risk_leak`, had a correlation of around **0.94** with the target when I only looked at the properly-formatted values. That is way too high. No single legitimate feature in credit scoring behaves like that. It basically meant `risk_leak` already "knew" the answer — it was derived from the target itself, which is a classic case of **data leakage**.

This matters because a model that scores 100% in the notebook but secretly relies on a leaked column is useless in production. The day the bank tries to use it on a *new* client (who obviously doesn't have a future-knowing `risk_leak` value), it falls apart.

So I decided to train everything **twice**:

1. **With `risk_leak`** — to show the inflated, fake-perfect numbers.
2. **Without `risk_leak`** — to get an honest picture of how the models actually perform.

The gap between those two runs is, in my opinion, the most important result in the whole report.

---

## Methodology

I structured the notebook around the five tasks in the assignment brief.

### 1. Exploratory Data Analysis
- Looked at the target distribution and found a clear class imbalance (far more non-defaulters than defaulters), which I needed to deal with later.
- Generated descriptive stats for the key numerical columns: `LIMIT_BAL`, `AGE`, `BILL_AMT_SUM`, `LIMIT_BAL_LOG`, `risk_leak`.
- Plotted histograms, bar charts and box plots to spot skew and outliers.
- Built a correlation heatmap — this is where the `risk_leak` anomaly first jumped out.

### 2. Data Preparation
- Handled missing values (several columns had them).
- Parsed the European number format — `risk_leak` and `LIMIT_BAL_LOG` used commas as decimal separators, which Python reads as text, so those columns needed cleaning before anything else worked.
- Standardised the numerical features with `StandardScaler`.
- Applied **SMOTE** on the training set to balance the classes so the models wouldn't just learn to always predict "no default".

### 3. Model Training
I trained six classifiers so I could actually compare different families of algorithms:
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- K-Nearest Neighbours (KNN)
- Gradient Boosting

I tuned the main models with `GridSearchCV` and `RandomizedSearchCV`.

### 4. Evaluation & Visualisation
- Confusion matrices and classification reports for each model.
- ROC curves with AUC scores.
- **SHAP** analysis on the best model to actually understand *why* it predicts what it predicts, not just how accurate it is.

### 5. Conclusion, Limitations & Future Work
Summarised in the report and below.

---

## Results

With `risk_leak` removed (the honest run), **Random Forest** came out on top at roughly **86% accuracy**, with the other tree-based and boosting models close behind.

The SHAP analysis lined up nicely with what credit risk literature says:

- **`PAY_0` (most recent repayment status)** was by far the strongest *legitimate* predictor. Clients who are two or more months behind right now are the ones most likely to default next month.
- The payment-history features showed a clean decay: `PAY_0` mattered more than `PAY_2`, which mattered more than `PAY_3`, and so on. Recent behaviour beats old behaviour.
- `LIMIT_BAL` had a weak *negative* relationship with default — higher limits usually go to lower-risk clients, so it acts as a rough proxy for creditworthiness.
- `BILL_AMT_SUM` was surprisingly weak. How much someone owes matters far less than whether they actually pay on time.

---

## Problems I ran into (and how I fixed them)

I'm including this because honestly the debugging was half the work.

- **`ParserError` on load.** The CSV had a few malformed lines. Fixed it with `on_bad_lines='skip'` and `encoding='utf-8-sig'` (there was a BOM at the start of the file messing up the first column name).
- **European decimals.** As mentioned, comma decimals had to be converted before the numeric columns would behave.
- **Hyperparameter tuning ran forever.** My first GridSearch was going to take over an hour. I cut the parameter grids down and reduced the number of CV folds to keep it sane while still searching a meaningful space.
- **SHAP kept timing out.** Computing SHAP values on the full set was too slow, so I sampled a smaller subset of rows for the explanation step. The interpretation barely changed but it actually finished.

---

## Repository structure

```
.
├── credit_card_default_prediction.ipynb   # the full notebook (all 5 tasks)
├── data/                                   # dataset (not committed if too large)
├── report/                                 # the written report (PDF/DOCX)
├── images/                                 # exported plots and SHAP figures
└── README.md
```

*(Adjust the folder names above to match what you actually committed.)*

---

## How to run it

The simplest way is to open the notebook in Google Colab, since that's where I built it.

1. Open `credit_card_default_prediction.ipynb` in Colab.
2. Upload the dataset CSV when prompted (or mount your Drive).
3. Run the cells top to bottom — they're ordered to match the five tasks.

If you'd rather run it locally:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn shap
```

Then open the notebook in Jupyter and run it the same way.

---

## What I'd do differently / future work

- The `RISK_RATING` column is a bit suspicious too — if it was assigned with any knowledge of the eventual default, it could be a softer form of leakage. I'd want to confirm how it was created before fully trusting it.
- Try a proper cost-sensitive approach. In real lending, a missed default (false negative) costs the bank far more than a false alarm, so optimising for plain accuracy isn't ideal. Tuning the decision threshold around recall would be more realistic.
- Bring in a gradient boosting library like XGBoost or LightGBM and compare properly.

---

## Tech used

Python, pandas, NumPy, scikit-learn, imbalanced-learn (SMOTE), SHAP, Matplotlib, Seaborn — all inside a Google Colab notebook.

---

*Built for the MSc Artificial Intelligence programme, Berlin School of Business and Innovation.*
