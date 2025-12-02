# Applied Econometrics in R — Function Flashcards

## Data import and preparation

What does this code/function do: `haven::read_dta()`
Purpose:
- Import Stata `.dta` files for regression exercises without manual conversion.
- Ensures variables (factors, numeric) are read consistently across examples.
Interpretation:
- Produces a data frame/tibble used as the design matrix for econometric models; no estimation itself but foundational for OLS/IV/logit setups.
Typical usage in this repo:
```r
wage1 <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2024/Lectures/A1 OLS/wage1.dta")
```
Related names / alternatives:
- `foreign::read.dta()` for older Stata versions; `data.table::fread()` for CSV inputs.

What does this code/function do: `dplyr::mutate()` / base transformations for interactions & dummies
Purpose:
- Create centered interactions or polynomial terms that encode treatment heterogeneity or nonlinear effects.
- Generate dummy variables required for LPM/logit/probit or DID designs.
Interpretation:
- Alters regressors to reflect econometric design (e.g., centered covariates so treatment coefficient = ATE at means; lag terms for AR tests).
Typical usage in this repo:
```r
df <- jtrain98 %>%
  mutate(c_earn = earn96 - mean(earn96, na.rm = TRUE),
         train:c_earn = train * c_earn)
```
Related names / alternatives:
- Base R `transform()`; `within()`; inline transformations inside formulas like `I(x^2)`.

What does this code/function do: `subset()` for sample splits
Purpose:
- Restrict estimation to specific periods/groups (e.g., pre/post policy year) to define treatment/control samples.
Interpretation:
- Ensures comparability and correct DID contrasts by isolating relevant observations.
Typical usage in this repo:
```r
df_1978 <- subset(df, y81 == 0)
```
Related names / alternatives:
- `dplyr::filter()` for pipe workflows; logical indexing (`df[df$y81==0, ]`).

What does this code/function do: `model.matrix()` for design matrices
Purpose:
- Extract the regressor matrix (without response) for penalized regressions or manual prediction.
Interpretation:
- Supplies \(X\) matrix for LASSO/CV routines where formulas are not used directly; columns map to regressors in OLS/GLM contexts.
Typical usage in this repo:
```r
X <- model.matrix(~ sat + tothrs + athlete + hsize + hsrank + hsperc +
                    female + white + black + fem_ath + bl_ath + wh_ath,
                  data = GPA)[, -1]
```
Related names / alternatives:
- `model.frame()` to include response; `model.matrix.lm()` on fitted models.

## OLS and linear models

What does this code/function do: `lm()` with formula `y ~ x1 + x2`
Purpose:
- Estimate classical and multiple linear regression models (OLS) across cross-section, DID, and treatment-adjusted setups.
- Baseline for counterfactuals (ATE via regression adjustment) and elasticity interpretation in log specifications.
Interpretation:
- Returns coefficient estimates assuming exogeneity; paired with robust SE to relax homoskedasticity assumption.
Typical usage in this repo:
```r
m0 <- lm(lbwght ~ npvis + cigs + male + mwhte + mblck + fwhte + fblck + lmage + lfage, data = BWGHT)
```
Related names / alternatives:
- `dynlm()` for time-series regressions; `glm()` for non-linear links; `estimatr::lm_robust()` for built-in robust SE.

What does this code/function do: `dynlm()` with lag/difference operators `L()`/`d()`
Purpose:
- Run OLS on time-series models with automatically aligned lags or differences to test AR terms or dynamic relationships.
Interpretation:
- Estimates linear models where regressors include lags/first differences; useful for testing serial correlation (e.g., AR(1) in residuals) without manual shifting.
Typical usage in this repo:
```r
reg1 <- dynlm(inf ~ unem, data = tsdata, end = 1996)
coeftest(dynlm(resid(reg1) ~ L(resid(reg1)) + 0))
```
Related names / alternatives:
- `stats::tslm()`; manual lag creation via `dplyr::lag()` then `lm()`; HAC inference via `sandwich::NeweyWest()`.

What does this code/function do: `predict()` / `fitted()` / `resid()` from lm/dynlm objects
Purpose:
- Extract fitted values or residuals for diagnostics (BP/White, AR tests) or to build auxiliary regressions and treatment predictions.
Interpretation:
- Fitted values approximate \(\hat{y}\); residuals approximate \(\hat{u}\). Used to detect heteroskedasticity, serial correlation, or compute treatment-effect contrasts.
Typical usage in this repo:
```r
df_1978$uhat2 <- resid(m0)^2
reg2 <- lm(resid(reg1)^2 ~ lincome + lcigpric + educ + age + agesq + restaurn, data = smoke)
```
Related names / alternatives:
- `augment()` from broom for tidy residuals/fits; `rstandard()` for standardized residuals.

## Robust SE and hypothesis testing

What does this code/function do: `lmtest::coeftest()` with `sandwich::vcovHC()` or `car::hccm()`
Purpose:
- Compute heteroskedasticity-robust (HC0/HC1) or cluster/HAC-adjusted standard errors for OLS/IV estimators.
- Replace classical t-tests when MLR.5 (homoskedasticity) fails.
Interpretation:
- Coefficient estimates unchanged; SEs and t-stats adjust for heteroskedasticity (White) or serial correlation (when paired with `NeweyWest`).
Typical usage in this repo:
```r
coeftest(reg1, vcov = hccm(reg1, type = "hc0"))
coeftest(l_iv, vcov = vcovHC(l_iv, type = "HC0"))
```
Related names / alternatives:
- `estimatr::lm_robust()`; `sandwich::vcovCL()` for clustering; `clubSandwich::coef_test()` for CRVE.

What does this code/function do: `sandwich::NeweyWest()`
Purpose:
- Provide HAC (heteroskedasticity and autocorrelation consistent) variance-covariance matrices for time-series OLS.
- Addresses serial correlation up to chosen lag in dynamic models.
Interpretation:
- Expands inference validity when errors are autocorrelated; test statistics based on HAC SE remain consistent.
Typical usage in this repo:
```r
coeftest(reg1, vcov = NeweyWest(reg1, lag = 3))
```
Related names / alternatives:
- `vcovHAC()`; `nlme::gls()` for explicit correlation structures; manual BG tests to detect serial correlation.

What does this code/function do: `lmtest::bptest()` (Breusch–Pagan)
Purpose:
- Test for heteroskedasticity by regressing squared residuals on regressors.
Interpretation:
- Rejecting H0 implies heteroskedastic errors; motivates robust SE or FGLS.
Typical usage in this repo:
```r
bptest(reg1)
```
Related names / alternatives:
- Manual LM calculation via auxiliary regression \(R^2 \times n\); `white_lm` variants using fitted values/squares.

What does this code/function do: `lmtest::bgtest()` / manual AR regression
Purpose:
- Breusch–Godfrey test for higher-order autocorrelation using residual lags; manual AR(k) regression yields LM or F statistics.
Interpretation:
- Significant statistics indicate serial correlation → use HAC SE, differencing, or AR terms.
Typical usage in this repo:
```r
bgtest(inf ~ unem, order = 3, type = "Chisq", data = phillips.limited)
# manual AR(3)
ar3 <- lm(res ~ res_1 + res_2 + res_3 - 1, data = phillips.limited)
```
Related names / alternatives:
- `durbinWatsonTest()` for AR(1); `car::linearHypothesis()` on lagged residuals; `dwtest()` from lmtest.

What does this code/function do: `lmtest::resettest()` / manual fitted-value powers
Purpose:
- Ramsey RESET specification test using higher-order fitted values to detect omitted nonlinearities.
Interpretation:
- Rejecting H0 suggests mis-specification; consider adding polynomials/interactions.
Typical usage in this repo:
```r
r_test <- resettest(aux, 2:3, c("fitted"), data = Data)
```
Related names / alternatives:
- Manual inclusion of `yhatsq` and `yhatcub` then `linearHypothesis()`; `spec.lm()`.

What does this code/function do: `car::linearHypothesis()`
Purpose:
- Wald tests for linear restrictions (single or joint) on model coefficients, using chosen vcov (classical or robust).
Interpretation:
- Tests exclusion restrictions, equality of coefficients, interaction significance; supports robust vcov for heteroskedasticity/serial correlation.
Typical usage in this repo:
```r
linearHypothesis(m1, c("y81 = 0", "ldist:y81 = 0"))
linearHypothesis(reg3, c("IQ=KWW", "KWW=0"))
```
Related names / alternatives:
- `anova()` for nested models; `waldtest()` from lmtest; `linearHypothesis(..., vcov.=vcovHC(...))` for robust Wald.

What does this code/function do: `car::vif()`
Purpose:
- Compute variance inflation factors to diagnose multicollinearity in multiple regression.
Interpretation:
- High VIF indicates inflated SEs due to collinear regressors, impacting inference reliability.
Typical usage in this repo:
```r
vif(reg1)
```
Related names / alternatives:
- `performance::check_collinearity()`; condition numbers via `kappa()`.

What does this code/function do: Likelihood ratio via `lmtest::lrtest()`
Purpose:
- Compare nested models (e.g., probit with extra quadratic terms) using LR statistics.
Interpretation:
- Rejecting H0 favors the richer model; aligns with maximum-likelihood inference for GLMs.
Typical usage in this repo:
```r
lrtest(probit, probit2)
```
Related names / alternatives:
- `anova()` for GLMs with `test="Chisq"`; Wald tests via `linearHypothesis()`.

## FGLS, WLS, and heteroskedasticity handling

What does this code/function do: Weighted least squares via `lm(..., weights = w)`
Purpose:
- Implement Feasible GLS when heteroskedasticity structure is estimated from residuals (e.g., log-variance regression).
Interpretation:
- Reweights observations to stabilize variance; improves efficiency relative to OLS under heteroskedasticity assumptions.
Typical usage in this repo:
```r
varreg <- lm(log(resid(reg1)^2) ~ lincome + lcigpric + educ + age + agesq + restaurn, data = smoke)
w <- 1 / exp(fitted(varreg))
WLS <- lm(cigs ~ lincome + lcigpric + educ + age + agesq + restaurn, weight = w, data = smoke)
```
Related names / alternatives:
- `gls()` in nlme; `feols(..., weights =)`; HC SEs as a robustness alternative.

What does this code/function do: Manual White/BP auxiliary regression of `resid^2`
Purpose:
- Diagnose heteroskedasticity and compute LM/F statistics from regression of squared residuals on regressors/polynomials.
Interpretation:
- \(n R^2\) yields LM statistic; high values imply heteroskedasticity → use robust SE or re-specify variance model.
Typical usage in this repo:
```r
reg2 <- lm(resid(reg1)^2 ~ lotsize + sqrft + bdrms, data = hprice1)
LM <- summary(reg2)$r.squared * length(reg2$residuals)
```
Related names / alternatives:
- `bptest()`; White test including fitted values and squares.

## IV and 2SLS

What does this code/function do: `AER::ivreg()`
Purpose:
- Estimate instrumental variables / 2SLS models for endogenous regressors (education, price, etc.).
Interpretation:
- Provides consistent estimates when regressors correlate with errors, assuming valid instruments; inference often paired with robust SE.
Typical usage in this repo:
```r
l_iv <- ivreg(lwage ~ educ + exper + expersq + black + smsa + south + smsa66 + reg662 + reg663 + reg664 + reg665 + reg666 + reg667 + reg668 + reg669 |
               nearc4 + exper + expersq + black + smsa + south + smsa66 + reg662 + reg663 + reg664 + reg665 + reg666 + reg667 + reg668 + reg669, data = Data)
```
Related names / alternatives:
- `ivreg::ivreg()` from ivreg pkg (same interface); `fixest::feols(..., iv = )`; `iv_robust()` from estimatr for robust SE.

What does this code/function do: Robust inference on IV via `coeftest(..., vcovHC())`
Purpose:
- Apply heteroskedasticity-robust or cluster-robust SE to IV/2SLS estimates.
Interpretation:
- Adjusts IV standard errors when homoskedasticity fails, ensuring valid Wald tests on instrumented coefficients.
Typical usage in this repo:
```r
coeftest(l_iv, vcov = vcovHC(l_iv, type = "HC0"))
```
Related names / alternatives:
- `sandwich::vcovCL()` for clustering; `iv_robust()` (estimatr) computes HC SE directly.

What does this code/function do: First-stage relevance and over-identification checks (manual LM/Sargan)
Purpose:
- Test instrument strength and validity via first-stage F (weak-ID) and Sargan/over-ID LM using residual regression on instruments.
Interpretation:
- High first-stage F indicates relevant instruments; insignificant Sargan LM implies instruments satisfy exclusion.
Typical usage in this repo:
```r
aux <- lm(u_hat ~ nearc4 + fatheduc + exper + expersq + black + smsa + south + smsa66 + reg662 + reg663 + reg664 + reg665 + reg666 + reg667 + reg668 + reg669, data = Data)
LM_statistic <- summary(aux)$r.squared * nrow(Data)
```
Related names / alternatives:
- `AER::ivreg()` diagnostics; `overid()` in ivreg package; `sargan()` in plm; Hansen J in GMM contexts.

What does this code/function do: Custom `ivreg2()` Durbin–Wu–Hausman wrapper
Purpose:
- User-defined function to compute IV coefficients, first-stage F, and Wu–Hausman endogeneity tests when packaged tools are unavailable.
Interpretation:
- Compares OLS vs IV estimates; significant DWH implies endogenous regressors → prefer IV estimates.
Typical usage in this repo:
```r
end <- ivreg2(lwage ~ educ + exper + expersq, endog = "exper", iv = c("motheduc", "fatheduc"), data = Data_omit, digits = 3)
F <- end$endogeneity[1]
```
Related names / alternatives:
- `ivreg::hausman()`; `AER::ivdiag`; manual Wu–Hausman via residual inclusion (`u_hat`) regression.

## Policy evaluation and treatment effects

What does this code/function do: DID via interaction term `y81*nearinc` in `lm()`
Purpose:
- Estimate difference-in-differences by interacting post-treatment indicator with treatment group.
Interpretation:
- Interaction coefficient = treatment effect; controls for baseline differences and common trends between groups.
Typical usage in this repo:
```r
summary(lm(rprice ~ y81 + nearinc + y81 * nearinc, data = kielmc))
```
Related names / alternatives:
- Two-group mean difference `(gamma78 - gamma81)`; `fixest::feols(..., fe =)` for panel DID.

What does this code/function do: Regression adjustment / URA with centered interactions
Purpose:
- Compute ATE by regressing outcome on treatment, covariates, and treatment×demeaned covariates so the treatment coefficient equals ATE at mean covariate values.
Interpretation:
- Treatment main effect after centering represents average partial effect; interactions test heterogeneity.
Typical usage in this repo:
```r
ura <- lm(earn98 ~ train + earn96 + educ + age + married +
            train.demeaned_earn96 + train.demeaned_educ +
            train.demeaned_age + train.demeaned_married, data = jtrain)
```
Related names / alternatives:
- Simple difference-in-means `lm(outcome ~ train)`; propensity-score weighting (not shown); manual URA prediction `tau_hat_ura` via two-group models.

What does this code/function do: Auxiliary two-regression ATE/ATT prediction using `model.matrix` and group-specific OLS
Purpose:
- Estimate potential outcomes for all units using separate treated/control regressions, then average differences (URA formula).
Interpretation:
- Provides ATE or ATT consistent with regression adjustment; connects to treatment effect theory without explicit matching.
Typical usage in this repo:
```r
g1 <- lm(unem98 ~ earn96 + educ + age + married, data = jt_treated)
g0 <- lm(unem98 ~ earn96 + educ + age + married, data = jt_control)
mu1 <- as.numeric(model.matrix(~ earn96 + educ + age + married, data = df) %*% coef(g1))
mu0 <- as.numeric(model.matrix(~ earn96 + educ + age + married, data = df) %*% coef(g0))
tau_hat_ura <- mean(mu1 - mu0)
```
Related names / alternatives:
- `difference_in_means()` (estimatr); `marginaleffects` predictions on interaction models.

## Binary choice / probit models

What does this code/function do: `glm(..., family = binomial(link = probit))`
Purpose:
- Estimate probit models for binary outcomes (e.g., arrest, employment) via MLE.
Interpretation:
- Coefficients scale latent index; significance assessed via z-stats; fitted values give predicted probabilities.
Typical usage in this repo:
```r
probit <- glm(arr86 ~ pcnv + avgsen + tottime + ptime86 + inc86 + black + hispan + born60,
              family = binomial(link = probit), data = Data)
```
Related names / alternatives:
- `glm(..., link = logit)`; linear probability model `lm()` with robust SE; `AER::ivprobit` (not used here).

What does this code/function do: `margins::margins()` and `predict(..., type = "response")`
Purpose:
- Compute marginal effects at specified covariate values and predicted probabilities from GLM/probit models.
Interpretation:
- Translates probit coefficients into changes in probability; contrasts scenarios (e.g., pcnv=0.25 vs 0.75).
Typical usage in this repo:
```r
lev1 <- data.frame(0.75, mean(Data$avgsen), mean(Data$tottime), mean(Data$ptime86),
                   mean(Data$inc86), 1, 0, 1)
names(lev1) <- c("pcnv", "avgsen", "tottime", "ptime86", "inc86", "black", "hispan", "born60")
margins(probit, at = lev1)
p1 <- predict(probit, lev1, type = "response")
```
Related names / alternatives:
- `marginaleffects::marginaleffects()`; `effects` package; analytical marginal effects for logit via `plm::phtest`? (not here).

What does this code/function do: Classification metrics from fitted probabilities
Purpose:
- Convert predicted probabilities to binary classifications and compute percent correctly predicted for model evaluation.
Interpretation:
- Measures in-sample predictive performance; not causal but informs model fit.
Typical usage in this repo:
```r
Data$tarr86 <- as.numeric(Data$fit >= 0.5)
table(Data$arr86, Data$tarr86)
(1903 + 78) / 2725
```
Related names / alternatives:
- `yardstick` metrics (accuracy, ROC); confusion matrices via `caret`.

## Time-series correlation and diagnostics

What does this code/function do: Lag creation via `dplyr::lag()` and AR auxiliary regressions
Purpose:
- Build lagged residuals or series for AR tests (e.g., AR(3) in Phillips curve), enabling LM or F statistics.
Interpretation:
- Helps detect serial correlation structure; informs HAC bandwidth or differencing choices.
Typical usage in this repo:
```r
phillips.limited <- phillips.limited %>%
  mutate(res_1 = dplyr::lag(res, n = 1, default = NA),
         res_2 = dplyr::lag(res, n = 2, default = NA),
         res_3 = dplyr::lag(res, n = 3, default = NA))
ar3 <- lm(res ~ res_1 + res_2 + res_3 - 1, data = phillips.limited)
```
Related names / alternatives:
- `stats::lag()`; `dynlm` lag operator `L()` inside formulas.

## Shrinkage and high-dimensional regression

What does this code/function do: `glmnet::glmnet()` and `glmnet::cv.glmnet()` (LASSO/Ridge)
Purpose:
- Estimate penalized regressions (LASSO alpha=1, Ridge alpha=0) and select penalty via cross-validation.
Interpretation:
- Shrinks/sets coefficients to zero to manage multicollinearity and prediction accuracy; selected lambda balances bias–variance.
Typical usage in this repo:
```r
cv_fit <- cv.glmnet(X, y, alpha = 1, nfolds = 10, family = "gaussian", standardize = TRUE)
best_lambda <- cv_fit$lambda.min
coef(cv_fit, s = "lambda.min")
```
Related names / alternatives:
- `gglasso` for grouped penalties; `glmnet` with `alpha=0` for Ridge; `caret`/`tidymodels` wrappers.

