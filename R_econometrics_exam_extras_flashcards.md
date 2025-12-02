# Applied Econometrics in R — Extra Exam-Oriented Flashcards

## Distribution functions for manual test calculations

What does this code/function do: `stats::pf()` for F-test p-values
Purpose:
- Convert manually computed F statistics into p-values for joint significance tests.
- Used when building F-tests by hand (e.g., White test or custom restriction tests) instead of relying on built-in helpers.
Interpretation:
- Returns the upper-tail probability under the F distribution; small values imply rejection of the null in joint linear restriction tests.
Typical usage in this repo:
```r
F_stat <- ((rrss - urss)/2)/((urss)/(df))
p_value <- pf(F_stat, 2, df, lower.tail = FALSE)
```
Related names / alternatives:
- `anova()` or `car::linearHypothesis()` compute the same test automatically; `qf()` gives critical values when you prefer rejection regions.

What does this code/function do: `stats::pchisq()` and `stats::qchisq()` for LM/LR chi-square tests
Purpose:
- Map LM or LR statistics to p-values (`pchisq`) or retrieve critical values (`qchisq`) for chi-square based diagnostics.
- Applied to overidentification checks and specification tests derived from auxiliary regressions.
Interpretation:
- Large test statistics with small `pchisq(..., lower.tail = FALSE)` indicate rejection of the null (e.g., excluded instruments are relevant or errors are homoskedastic).
Typical usage in this repo:
```r
LM <- summary(reg2)$r.squared * 526
1 - pchisq(LM, 2)
qchisq(0.95, 1)
```
Related names / alternatives:
- `lmtest::lrtest()` automates LR tests; `car::linearHypothesis(..., test = "Chisq")` performs Wald-style chi-square tests.

## Multicollinearity and correlation diagnostics

What does this code/function do: `stats::cor(..., use = "na.or.complete")`
Purpose:
- Compute pairwise correlations to flag multicollinearity between regressors before running OLS or IV models.
- Handles missing values safely in problem-set datasets.
Interpretation:
- Values near ±1 imply potential variance inflation and unstable coefficient estimates; low absolute correlations suggest multicollinearity is minor.
Typical usage in this repo:
```r
cor(discrim$lincome, discrim$prppov, use = "na.or.complete")
```
Related names / alternatives:
- `car::vif()` provides a regression-based multicollinearity measure; `cov()` yields covariance rather than standardized correlation.

## Weighted least squares and heteroskedasticity modeling

What does this code/function do: `lm(..., weight = w)` for feasible GLS/WLS
Purpose:
- Re-estimate linear models using estimated inverse-variance weights to correct heteroskedasticity.
- Implements the second step of FGLS where weights come from an auxiliary log-variance regression.
Interpretation:
- Coefficients are still linear but now efficient under the modeled heteroskedasticity; standard errors correspond to the weighted likelihood.
Typical usage in this repo:
```r
w <- 1 / exp(fitted(reg2))
WLS <- lm(cigs ~ lincome + lcigpric + educ + age + agesq + restaurn, weight = w)
```
Related names / alternatives:
- `nlme::gls()` allows richer correlation structures; `estimatr::lm_robust()` keeps OLS coefficients but adjusts SEs instead of reweighting.

## Temporary data attachment for quick model specification

What does this code/function do: `attach()` / `detach()` for datasets
Purpose:
- Place data frame columns on the search path to shorten formula typing in exploratory problem-set work.
- Useful in quick regression experiments when repeatedly referencing the same dataset.
Interpretation:
- Simplifies variable access but risks naming conflicts; ensures formulas like `lm(cigs ~ lincome + lcigpric + ...)` resolve variables inside the attached data.
Typical usage in this repo:
```r
attach(smoke)
summary(lm(cigs ~ lincome + lcigpric + educ + age + agesq + restaurn))
...
detach(smoke)
```
Related names / alternatives:
- Prefer `with()` or `dplyr::mutate()` pipelines for safer scoping; always `detach()` after use to avoid masking issues.

## Binary response prediction and evaluation

What does this code/function do: thresholding fitted probabilities with `as.numeric(p_hat >= 0.5)`
Purpose:
- Convert predicted probabilities from linear or probit models into hard class predictions for accuracy checks.
- Supports policy evaluation metrics such as percent correctly predicted.
Interpretation:
- Classifies observations into 0/1 based on a chosen cutoff; ties model outputs to decision rules (e.g., predict participation if probability ≥ 50%).
Typical usage in this repo:
```r
Data$te401k <- as.numeric(Data$fit >= 0.5)
```
Related names / alternatives:
- Adjust the threshold for imbalanced data; `ifelse()` performs the same logic; ROC/AUC tools offer cutoff-independent performance views.

What does this code/function do: `table()` / `as.data.frame()` for confusion matrices
Purpose:
- Summarize predicted vs. actual binary outcomes to compute accuracy rates and subgroup performance.
- Core step after thresholding probabilities in discrete-choice models.
Interpretation:
- Counts along the diagonal represent correct predictions; proportions highlight model fit beyond coefficient significance.
Typical usage in this repo:
```r
as.data.frame(table(Data$e401k, Data$te401k, dnn = list("e401k", "te401k")))
(4607 + 1429) / nrow(Data)  # percent correctly predicted
```
Related names / alternatives:
- `caret::confusionMatrix()` adds precision/recall; `prop.table(table(...))` yields shares directly.

## Time-series setup utilities

What does this code/function do: `ts()` to create time-series objects
Purpose:
- Convert raw data frames into R time-series objects with start dates for use in dynamic regressions.
- Ensures lag operators in `dynlm()` work with properly indexed series.
Interpretation:
- Produces an ordered series so that `L()` in formulas references actual temporal lags; crucial for autocorrelation tests like AR(1) checks.
Typical usage in this repo:
```r
tsdata <- ts(phillips, start = 1948)
```
Related names / alternatives:
- `zoo::zoo()` or `xts::xts()` for irregular time stamps; `tsibble` for tidy time-series workflows.
