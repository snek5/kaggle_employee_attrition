# Employee Attrition Analysis through Survival Modeling and Principal Component Analysis

> An interpretable statistical approach to employee attrition using Survival Analysis, Principal Component Analysis (PCA), and parametric/non-parametric survival models.

---

## Overview

Employee attrition is commonly approached as a binary classification problem:

> **Will an employee leave?**

While useful, this ignores one of the most important questions for workforce planning:

> **When is an employee likely to leave?**

This project reformulates employee attrition as a **time-to-event** problem using **Survival Analysis**, allowing us to estimate employee retention over time while identifying the variables that accelerate or delay employee turnover.

Unlike many machine learning approaches that prioritize prediction accuracy, this notebook emphasizes:

- Statistical interpretability
- Explainable models
- Time-to-event analysis
- Mathematical rigor

The project uses the IBM HR Analytics Employee Attrition dataset to demonstrate the complete workflow from exploratory analysis to statistical survival modeling.

---

# Mathematical Background

Before reading the notebook, it is useful to understand the statistical concepts used throughout the analysis.

---

## Survival Analysis

Survival Analysis studies the amount of time before an event occurs.

Instead of modeling

$$
Y \in \{0,1\}
$$

we model

$$
T
$$

where

- **T** = employee tenure until resignation.

The objective is estimating

$$
P(T>t)
$$

which represents the probability an employee remains employed after time $t$.

---

## Survival Function

The Survival Function is defined as

$$
S(t)=P(T>t)
$$

Properties:

- $S(0)=1$
- decreases monotonically
- approaches zero as time increases

This represents the probability that an employee has **not yet left** by tenure $t$.

---

## Kaplan-Meier Estimator

The Kaplan-Meier estimator estimates the survival curve without assuming any probability distribution.

$$
\hat S(t)
=
\prod_{t_i\le t}
\left(
1-\frac{d_i}{n_i}
\right)
$$

where

- $d_i$ = number of resignations
- $n_i$ = employees still employed before $t_i$

This produces empirical survival curves that compare employee groups across different characteristics.

---

## Hazard Function

While the survival function measures the probability of remaining employed, the hazard function measures the instantaneous resignation risk.

$$
h(t)
=
\frac{f(t)}{S(t)}
$$

The hazard answers:

> Given an employee has remained until today, what is the likelihood they leave immediately afterwards?

---

## Cox Proportional Hazards Model

The Cox model estimates

$$
h(t|X)
=
 h_0(t)e^{\beta^TX}
$$

where

- $h_0(t)$ = baseline hazard
- $X$ = employee characteristics
- $\beta$ = estimated coefficients

Exponentiating the coefficients gives **Hazard Ratios**

$$
HR=e^\beta
$$

Interpretation:

- HR > 1 increases resignation risk
- HR < 1 decreases resignation risk

This provides an interpretable explanation of which employee characteristics contribute most strongly to attrition.

---

## Accelerated Failure Time (AFT) Model

Rather than modeling hazard, AFT models the survival time directly.

The Log-Normal AFT model assumes

$$
\log(T)
=
\beta^TX+\sigma\epsilon
$$

Positive coefficients increase expected employment duration, while negative coefficients shorten expected tenure.

Compared to Cox regression, AFT models often provide a more intuitive interpretation because they directly estimate how variables affect employee lifespan within the company.

---

## Principal Component Analysis (PCA)

Many HR variables are highly correlated.

Examples include

- Monthly Income
- Job Level
- Years at Company
- Total Working Years

PCA reduces these correlated variables into orthogonal principal components.

Mathematically,

$$
\Sigma
=
\frac1nX^TX
$$

is decomposed into

$$
\Sigma
=
Q\Lambda Q^T
$$

where

- eigenvectors define principal directions
- eigenvalues quantify explained variance

The transformed observations become

$$
Z=XQ
$$

reducing dimensionality while preserving most of the information.

---

## Akaike Information Criterion (AIC)

Model comparison is performed using

$$
AIC
=
2k-2\log(L)
$$

where

- $k$ = number of model parameters
- $L$ = maximum likelihood

Lower AIC indicates a better trade-off between model complexity and goodness of fit.

---

# Methodology

The notebook follows the following analytical workflow.

```text
Raw Employee Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Principal Component Analysis
        │
        ▼
Kaplan-Meier Survival Curves
        │
        ▼
Log-Normal Accelerated Failure Time Model
        │
        ▼
Cox Proportional Hazards Model
        │
        ▼
Model Comparison (AIC)
        │
        ▼
Interpretation & Business Recommendations
```

---

# Statistical Techniques Used

- Exploratory Data Analysis (EDA)
- Correlation Analysis
- Principal Component Analysis (PCA)
- Kaplan-Meier Survival Estimation
- Log-Rank Tests
- Log-Normal Accelerated Failure Time Model
- Cox Proportional Hazards Regression
- Akaike Information Criterion (AIC)

---

# Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- lifelines

---

# Repository Structure

```
employee_attrition_analysis/
│
├── employee_attrition_kaggle.ipynb
├── README.md
├── requirements.txt
└── data/
```

---

# Running the Notebook

Clone the repository

```bash
git clone https://github.com/<username>/employee-attrition-survival-analysis.git

cd employee-attrition-survival-analysis
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Linux / macOS

```bash
source .venv/bin/activate
```

Windows

```powershell
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch Jupyter

```bash
jupyter notebook
```

Open

```
employee_attrition_kaggle.ipynb
```

---

# Key Takeaways

This project demonstrates how traditional statistical methods remain powerful tools for explainable workforce analytics.

Rather than treating attrition as a simple classification problem, Survival Analysis provides richer insights by estimating employee retention over time and quantifying the effect of employee characteristics on expected tenure.

The notebook also illustrates how dimensionality reduction through PCA can simplify correlated HR variables before downstream statistical modeling, producing interpretable models suitable for real-world decision making.

---

# Future Work

Potential extensions include:

- Random Survival Forests
- DeepSurv neural survival models
- XGBoost Survival
- Time-varying covariates
- SHAP explanations for survival models
- Concordance Index (C-index) evaluation
- Cross-validation for survival models
- Bayesian Survival Analysis

---

# References

- Cox, D. R. (1972). Regression Models and Life-Tables.
- Kaplan, E. L., & Meier, P. (1958). Nonparametric Estimation from Incomplete Observations.
- Jolliffe, I. T. Principal Component Analysis.
- Hosmer, D., Lemeshow, S., & May, S. Applied Survival Analysis.
- Kleinbaum, D. G., & Klein, M. Survival Analysis: A Self-Learning Text.

---

## Author

This project was developed as part of my work as an Innovation officer to showcase statistical techniques. My interests lie at the intersection of mathematics, statistics, and data engineering, with an emphasis on building reproducible and explainable analytical workflows.
