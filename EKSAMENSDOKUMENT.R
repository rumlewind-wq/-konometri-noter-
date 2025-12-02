# ============================== Libraries ----
# Scope: grouped by task. Each comment lists core, exam-relevant functions.

# --- Data & datasets ----------------------------------------------------------- -
library(wooldridge)     # datasets for econometrics: data("<name>")
library(data.table)     # fast I/O + table ops: fread(), fwrite(), :=, setDT(), rbindlist()
library(readr)          # tidy I/O: read_csv(), read_tsv(), write_csv(), cols(), locale()
library(tibble)         # modern frames: tibble(), as_tibble(), enframe(), deframe()
library(dplyr)          # verbs: filter(), select(), mutate(), summarise(), group_by(), arrange()
library(tidyr)          # reshape: pivot_longer(), pivot_wider(), separate(), unite(), drop_na()

# --- Plotting ------------------------------------------------------------------ -
library(ggplot2)        # grammar of graphics: ggplot(), aes(), geom_*, facet_wrap(), theme()

# --- Core OLS, inference, robust SEs -------------------------------------- -- --
library(lmtest)         # tests: coeftest(), waldtest(), lrtest(), bptest(), resettest()
library(sandwich)       # robust VCs: vcovHC(), vcovHAC(), vcovCL(), bread(), meat()
library(car)            # hypothesis & diagnostics: linearHypothesis(), vif(), durbinWatsonTest()
library(broom)          # model tidiers: tidy(), glance(), augment()
library(modelsummary)   # tables: modelsummary(), msummary(), datasummary_skim()

# --- Robust and design-based estimators ----------------------------------- ---- -
library(estimatr)       # robust estimators: lm_robust(), iv_robust(), difference_in_means()

# --- IV / 2SLS / econometric tools ------------------------------------------ --
library(AER)            # IV utilities: ivreg() (alt), ivdiagnostics, tobit(), hurdle(), pscl links
library(ivreg)          # dedicated IV: ivreg(), diagnostics(), vcov() methods
library(fixest)         # high-perf FE/IV: feols(), fepois(), feglm(), iv(), etable(), i(), cluster

# --- Panels & clustered inference -------------------------------------------- -
library(plm)            # panels: plm(), pdata.frame(), phtest(), purtest(), pooltest(), vcovHC()
library(clubSandwich)   # cluster-robust tests: vcovCR(), coef_test(), Wald_test()
library(multiwayvcov)   # multi-way clustering: cluster.vcov(), cluster.se()

# --- Marginal effects & postestimation --------------------------------------- -
library(marginaleffects) # effects/predictions: predictions(), marginaleffects(), avg_comparisons()
library(margins)         # classic margins: margins(), dydx(), cplot()

# --- Model quality & distributional tests ------------------------------------ -
library(performance)    # diagnostics: check_model(), check_collinearity(), r2(), icc()
library(moments)        # moments: skewness(), kurtosis(), moment()
library(nortest)        # normality tests: ad.test(), lillie.test(), cvm.test(), pearson.test()

# --- Time series, unit roots, VAR, forecasting ------------------------------- -
library(zoo)            # ordered obs: zoo(), rollapply(), na.locf(), index()
library(xts)            # extensible TS: xts(), merge.xts(), period.apply(), endpoints()
library(urca)           # unit root/cointegration: ur.df(), ur.pp(), ur.kpss(), ca.jo(), ca.po()
library(dynlm)          # dynamic LM: dynlm(), L(), d() for lags/diffs in formulas
library(vars)           # VAR/SVAR: VAR(), predict(), irf(), fevd(), causality()
library(forecast)       # forecasting: auto.arima(), Arima(), ets(), forecast(), accuracy()


# Datainmport ----
# ==== TEKST/TABULÆRE FILER ===================================================== -
# CSV / TSV / vilkårlig delimiter
library(data.table); dt <- fread("data.csv")                               # hurtig CSV/TSV
library(readr);    df <- read_csv("data.csv")                               # CSV
library(readr);    df <- read_tsv("data.tsv")                               # TSV
library(readr);    df <- read_delim("data.txt", delim = ";")                # vilkårlig delimiter
library(readr);    df <- read_fwf("data.fwf", fwf_cols(id=c(1,5), v=c(6,15))) # fixed width
library(vroom);    df <- vroom("data.csv")                                  # multi-threaded

# Komprimerede filer og URLs
library(readr);    df <- read_csv("https://host/path/data.csv.gz")          # URL + .gz
library(utils);    df <- read.csv(unz("archive.zip","file.csv"))            # ZIP-indre fil

# ==== REGNEARK ================================================================= -
library(readxl);   df <- read_excel("data.xlsx", sheet = 1)                 # .xlsx/.xls/.xlsm
library(readxlsb); df <- read_xlsb("data.xlsb")                             # .xlsb (binary)
library(readODS);  df <- read_ods("data.ods")                               # LibreOffice/ODS
library(googlesheets4); df <- read_sheet("https://docs.google.com/...")     # Google Sheets

# ==== STATISTIKPAKKER ========================================================== -
library(haven);    df <- read_dta("file.dta")                               # Stata
library(haven);    df <- read_sav("file.sav")                               # SPSS
library(haven);    df <- read_por("file.por")                               # SPSS portable
library(haven);    df <- read_sas("file.sas7bdat", catalog_file = NULL)     # SAS
library(haven);    df <- read_xpt("file.xpt")                               # SAS transport
# Alternativ SAS7BDAT-loader:
library(sas7bdat); df <- read.sas7bdat("file.sas7bdat")

# ==== KOLONNEORIENTEREDE / HURTIGE FORMATTER =================================== -
library(arrow);    df <- read_parquet("data.parquet")                       # Parquet
library(arrow);    df <- read_feather("data.feather")                       # Feather/Arrow IPC
library(arrow);    ds <- open_dataset("s3://bucket/path/")                  # lazy dataset (lokal/cloud)

# ==== R-NATIVE ================================================================= -
obj <- readRDS("object.rds")                                                # .rds
load("workspace.RData")                                                     # .RData (indlæser objekter)

# ==== JSON / NDJSON / XML / HTML ============================================== -
library(jsonlite); df <- fromJSON("data.json", flatten = TRUE)              # JSON → tibble
library(jsonlite); df <- stream_in(file("data.ndjson"))                     # NDJSON (line-delimited)
library(xml2);     doc <- read_xml("data.xml")                              # XML-doc
library(rvest);    df <- read_html("https://example.com") |> html_table(fill=TRUE) |> `[[`(1)  # HTML-tabel #1

# ==== DATABASER ================================================================= -
library(DBI); library(RSQLite); con <- dbConnect(SQLite(),"db.sqlite"); df <- dbGetQuery(con,"SELECT * FROM t"); dbDisconnect(con) # SQLite
library(DBI); library(RPostgres); con <- dbConnect(RPostgres::Postgres(), dbname="db", host="host", user="u", password="p"); df <- dbGetQuery(con,"SELECT 1"); dbDisconnect(con) # Postgres
library(DBI); library(RMariaDB); con <- dbConnect(RMariaDB::MariaDB(), dbname="db", host="host", user="u", password="p"); df <- dbGetQuery(con,"SELECT 1"); dbDisconnect(con) # MySQL/MariaDB
library(DBI); library(odbc); con <- dbConnect(odbc::odbc(), Driver="SQL Server", Server="srv", Database="db", UID="u", PWD="p"); df <- dbGetQuery(con,"SELECT 1"); dbDisconnect(con) # SQL Server
library(duckdb);   con <- dbConnect(duckdb(), dbdir=":memory:"); duckdb_read_csv(con, "t", "data.csv"); df <- dbGetQuery(con,"SELECT * FROM t"); dbDisconnect(con) # DuckDB
library(bigrquery); con <- dbConnect(bigquery(), project="proj", dataset="ds"); df <- dbGetQuery(con,"SELECT 1") # BigQuery

# ==== CLOUD-STORAGE DIREKTE ===================================================== -
library(arrow);    df <- read_parquet("s3://bucket/path/file.parquet")      # AWS S3 (cred i env)
library(arrow);    df <- read_parquet("gs://bucket/path/file.parquet")      # Google Cloud Storage
library(AzureStor); cont <- blob_container("https://acct.blob.core.windows.net/c"); storage_download(cont,"file.csv","/tmp/file.csv"); df <- readr::read_csv("/tmp/file.csv") # Azure Blob

# ==== GEO / RUMDATA ============================================================= -
library(sf);       g  <- st_read("data.gpkg")                               # GeoPackage
library(sf);       g  <- st_read("data.geojson")                            # GeoJSON
library(sf);       g  <- st_read("shape_dir")                                # Shapefile-mappe
library(terra);    r  <- rast("raster.tif")                                  # GeoTIFF raster
library(terra);    v  <- vect("data.gpkg")                                   # vektor via terra

# ==== VIDENSKABSDATA (MULTIDIMENSIONELLE) ====================================== -
library(ncdf4);    nc <- nc_open("file.nc"); x <- ncvar_get(nc,"var"); nc_close(nc) # netCDF
library(rhdf5);    y  <- h5read("file.h5","/group/dataset")                 # HDF5

# ==== MATLAB, EXCEL-MAKROER, ANDRE ============================================= -
library(R.matlab); mat <- readMat("file.mat")                                # MATLAB .mat
library(readxl);   df  <- read_excel("file.xlsm")                            # Excel makro-arbejdsbog
library(readtext); txt <- readtext("dir/*.txt")                              # Korpora (txt/pdf/docx/html)

# ==== PDF-TABELLER / OCR ======================================================== -
library(tabulizer); tabs <- extract_tables("file.pdf")                       # PDF-tabeller
library(tesseract); library(magick); img <- image_read("scan.png"); txt <- ocr(img)  # OCR til tekst

# ==== SPARK / DISTRIBUERET ====================================================== -
library(sparklyr); sc <- spark_connect(master="local"); tbl <- spark_read_csv(sc,"t","data.csv", header=TRUE); collect(tbl) # Spark


# Midterm -----
## Midterm 1 ----
# A) Data
library(haven)
library(dplyr)
library(lmtest)
library(sandwich)
library(car)

BWGHT <- read_dta("~/Desktop/CBS/Eksamensforberedelse/Økonomitri/Tidligere eksamener/midt 1/mid_term.dta") %>% as.data.frame()
summary(BWGHT); str(BWGHT)

# B) Basismodel
m0 <- lm(lbwght ~ npvis + cigs + male + mwhte + mblck + fwhte + fblck + lmage + lfage, data = BWGHT)
summ_rob <- coeftest(m0, vcov. = vcovHC(m0, type = "HC1"))
nobs(m0); summary(m0)$r.squared; summary(m0)$adj.r.squared

# E) Multikollinearitet
car::vif(m0)
cor(BWGHT[, c("npvis","cigs","male","mwhte","mblck","fwhte","fblck","lmage","lfage")], use="pairwise.complete.obs")

# F) H0: β_npvis = −β_cigs  (kan testes med linearHypothesis)
car::linearHypothesis(m0, "npvis + cigs = 0", vcov.=vcovHC(m0, type="HC1"))

# G) dif-model
BWGHT$dif <- BWGHT$npvis - BWGHT$cigs
m1 <- lm(lbwght ~ dif + cigs + male + mwhte + mblck + fwhte + fblck + lmage + lfage, data = BWGHT)
coeftest(m1, vcov.=vcovHC(m1, type="HC1"))  # koef(dif)=β_npvis; koef(cigs)=β_cigs−β_npvis

# J) Breusch–Pagan
bptest(m0)  # H0: homoskedasticitet


## Midterm 2 -----

## ============================================================================ -
##  Midterm 2 — Econometrics (CBS)
##  Classical OLS version (no robust SEs)
## ============================================================================ -

## --- Load packages and data -------------------------------------------------- -
library(haven)
library(moments)
library(car)

# Import dataset
df <- read_dta("~/Desktop/CBS/Eksamensforberedelse/Økonomitri/Tidligere eksamener/midt 2/mid term resit data.dta")

# Inspect structure and summary
summary(df)
str(df)

## --- B) Base model for 1978 -------------------------------------------------- -
# Subset for year 1978 (y81 == 0)
df_1978 <- subset(df, y81 == 0)

# Simple regression: Price on log(distance)
m0 <- lm(price ~ ldist, data = df_1978)
summary(m0)

# Interpretation:
# Level–log model → a 1% increase in distance increases price by 0.01*β_ldist dollars.

## --- E) White test for heteroskedasticity ----------------------------------- -
# Step 1: Squared residuals
df_1978$uhat2 <- resid(m0)^2

# Step 2: Fitted values and their square
df_1978$y_fit <- fitted(m0)
df_1978$y_fitsq <- df_1978$y_fit^2

# Step 3: Auxiliary regression (White special case)
white_aux <- lm(uhat2 ~ y_fit + y_fitsq, data = df_1978)
summary(white_aux)

# Step 4: Compute LM statistic
R2u <- summary(white_aux)$r.squared
n <- nobs(white_aux)
LM_stat <- n * R2u

# Degrees of freedom = number of regressors excluding intercept
df_lm <- 2
pval <- 1 - pchisq(LM_stat, df = df_lm)

# Display results
list(
  n = n,
  R2_aux = R2u,
  LM_stat = LM_stat,
  df = df_lm,
  p_value = pval
)

# Interpretation:
# If p-value < 0.05 → reject homoskedasticity (MLR.5).
# With df=2, p≈0.37 → fail to reject homoskedasticity.

## --- F) Skewness and kurtosis ---------------------------------------------- -
# Check normality of price
skewness(df$price, na.rm = TRUE)
kurtosis(df$price, na.rm = TRUE)

# Log-transform for better normality
df$lprice <- log(df$price)
skewness(df$lprice, na.rm = TRUE)
kurtosis(df$lprice, na.rm = TRUE)

# Interpretation:
# log(price) is approximately normal, making OLS assumptions more plausible.

## --- G) Interaction model --------------------------------------------------- -
# Full model with year dummy and interaction
m1 <- lm(price ~ ldist * y81, data = df)
summary(m1)

# Interpretation:
# ldist → slope for 1978
# y81 → intercept shift (level difference)
# ldist:y81 → slope change in 1981
# Positive interaction → higher prices farther away after incinerator construction.

## --- H) Classical inference and hypothesis tests --------------------------- -
# Test if slope difference (interaction term) = 0
car::linearHypothesis(m1, "ldist:y81 = 0")

# Joint test: equal intercept and slope across years (Chow-type test)
car::linearHypothesis(m1, c("y81 = 0", "ldist:y81 = 0"))

# One-sided p-value for positive slope difference
t_value <- summary(m1)$coefficients["ldist:y81", "t value"]
df_resid <- df.residual(m1)
p_one_sided <- pt(t_value, df = df_resid, lower.tail = FALSE)
p_one_sided

# Interpretation:
# Interaction term is weakly significant (p≈0.067 one-sided ≈ 0.033).
# Suggests houses farther away became relatively more expensive in 1981.


## Mock midterm 3 ------
# ============================================================================== - 
# Midterm 2 — GPA & Athlete (mock_mid_term.dta) 
# Purpose: Answer parts (a)–(j) with clear, exam-ready R code and comments
# ============================================================================== -

# --- Setup ------------------------------------------------------------------ --
# Core: data I/O, OLS, robust SEs, joint tests, LASSO, tidy printing
suppressPackageStartupMessages({
  library(haven)        # read_dta()
  library(dplyr)        # pipes, mutate, select
  library(broom)        # tidy(), glance()
  library(lmtest)       # coeftest(), waldtest()
  library(sandwich)     # vcovHC()
  library(car)          # linearHypothesis()
  library(modelsummary) # modelsummary() tables
  library(glmnet)       # LASSO
})

# Reproducibility for CV
set.seed(1234)

# --- (a) Load data & show summary ------------------------------------------- -
# Question: Load as GPA and show summary stats. :contentReference[oaicite:0]{index=0}
GPA <- read_dta("~/Desktop/CBS/Eksamensforberedelse/Økonomitri/Tidligere eksamener/midt mock/mock_mid_term.dta")

# Quick structure + five-number summaries
str(GPA)
summary(GPA)   # matches Table 1A in the exam handout. :contentReference[oaicite:1]{index=1}

# --- (b) Simple OLS: colgpa on athlete -------------------------------------- -
# Model: level–level. Intercept = mean GPA for non-athletes; athlete adds level shift.
m_b <- lm(colgpa ~ athlete, data = GPA)

# Report classical and HC1-robust SEs
msummary(
  list("B: OLS colgpa~athlete" = m_b),
  vcov = list("Classical" = vcov(m_b), "HC1" = vcovHC(m_b, type = "HC1")),
  estimate = "{estimate} ({std.error})",
  stars = TRUE
)
# Interpretation guide (printed in comments only):
#  - coef[athlete] < 0 means athletes have lower mean GPA than non-athletes, c.p.
#  - Outline solution says about -0.28 and significant for (b). :contentReference[oaicite:2]{index=2}

# --- (c) Interpret athlete coefficient from (b) ------------------------------ -
# Read as: "Average difference in GPA between athletes and non-athletes,
# holding nothing else fixed." Level units on 0–4 GPA scale. (Printed via comments)

# --- (d) Assumptions for validity in (b) ------------------------------------- -
# SLR/MLR assumptions, tailored to this context (comments only):
#  A1 Linear in parameters: E[colgpa | athlete] = β0 + β1*athlete + u
#  A2 Random sampling: the 4,137 students form a random sample (stated in exam). :contentReference[oaicite:3]{index=3}
#  A3 No perfect collinearity: athlete not constant. OK.
#  A4 Zero conditional mean: E[u | athlete] = 0. Risky here since many omitted factors
#     (study time, ability, subject match) likely correlate with athlete → OVB risk. :contentReference[oaicite:4]{index=4}
#  A5 Homoskedasticity (for classical SEs). If doubtful, use HC1 as above.
# Consequences: Violation of A4 → biased/inconsistent β1; violation of A5 → SEs wrong.

# --- (e) Full model: add all other regressors -------------------------------- -
# Exam text: regress GPA on all other variables. Variables listed on page 2. :contentReference[oaicite:5]{index=5}
m_e <- lm(colgpa ~ sat + tothrs + athlete + hsize + hsrank + hsperc +
            female + white + black + fem_ath + bl_ath + wh_ath,
          data = GPA)

msummary(
  list("E: Full model" = m_e),
  vcov = list("Classical" = vcov(m_e), "HC1" = vcovHC(m_e, type = "HC1")),
  estimate = "{estimate} ({std.error})",
  stars = TRUE
)
# Outline solution: athlete flips sign (~ +0.29) and becomes insignificant once controls enter,
# consistent with omitted-variable bias in (b). Do not hard-code numbers; read from output. :contentReference[oaicite:6]{index=6}

# --- (f) Interpret athlete in (e) vs (b) ------------------------------------ --
# Comment:
#  - (b) is unconditional mean difference.
#  - (e) is ceteris paribus effect, controlling for SAT, hours, HS record, demographics, interactions.
#  - Sign change indicates OVB in (b) due to correlation between athlete and included controls. :contentReference[oaicite:7]{index=7}

# --- (g) State joint-significance hypothesis for interactions ---------------- -
# H0: fem_ath = 0, bl_ath = 0, wh_ath = 0  (jointly). Use an F-test and standard F rejection rule. :contentReference[oaicite:8]{index=8}
# Mechanics: Compare unrestricted model (m_e) to restricted model without these three terms.
# car::linearHypothesis computes the F with chosen vcov (classical or robust).

# --- (h) Conduct the F-test ------------------------------------------------- --
# Classical F-test:
lh_classic <- car::linearHypothesis(m_e,
                                    c("fem_ath = 0", "bl_ath = 0", "wh_ath = 0"),
                                    test = "F"
)
print(lh_classic)

# Robust (HC1) F-test (Wald test with HC1):
lh_hc1 <- car::linearHypothesis(m_e,
                                c("fem_ath = 0", "bl_ath = 0", "wh_ath = 0"),
                                test = "F",
                                vcov. = vcovHC(m_e, type = "HC1")
)
print(lh_hc1)
# Outline solution: p-value is very small → reject H0. Jointly significant even if
# individual t's look small, due to correlation among the three regressors. :contentReference[oaicite:9]{index=9}

# --- (i) Bias–variance trade-off (comments) ---------------------------------- -
# Adding irrelevant variables ↑ variance of estimates; omitting relevant variables (corr. with included)
# → bias (OVB). Model selection aims to minimize error via a good trade-off and, ideally, oracle properties. :contentReference[oaicite:10]{index=10}

# --- (j) LASSO selection + post-LASSO OLS ---------------------------------- -- 
# Task: Use LASSO on the (e) specification, report selection, explain methodology. :contentReference[oaicite:11]{index=11}
# Build X matrix (no response), letting glmnet handle standardization. Keep all regressors from (e).
X <- model.matrix(
  ~ sat + tothrs + athlete + hsize + hsrank + hsperc +
    female + white + black + fem_ath + bl_ath + wh_ath,
  data = GPA
)[, -1]  # drop intercept column

y <- GPA$colgpa

# K-fold CV to choose lambda that minimizes MSE (lambda.min) and 1-SE rule (lambda.1se)
cv_fit <- cv.glmnet(X, y, alpha = 1, nfolds = 10, family = "gaussian", standardize = TRUE)
plot(cv_fit)  # visual check: paths and chosen lambdas

best_lambda  <- cv_fit$lambda.min
sparser_lambda <- cv_fit$lambda.1se

# Coefficients at chosen lambdas
coef_min <- coef(cv_fit, s = "lambda.min")
coef_1se <- coef(cv_fit, s = "lambda.1se")

print(coef_min)
print(coef_1se)

# Extract selected variables at lambda.min (non-zero coefficients excluding intercept)
sel_idx <- which(as.numeric(coef_min) != 0)
sel_vars <- rownames(coef_min)[sel_idx]
sel_vars <- setdiff(sel_vars, "(Intercept)")
sel_vars

# Post-LASSO OLS using selected variables
if (length(sel_vars) > 0) {
  fmla_post <- reformulate(sel_vars, response = "colgpa")
  m_post <- lm(fmla_post, data = GPA)
  msummary(
    list("Post-LASSO OLS" = m_post),
    vcov = list("Classical" = vcov(m_post), "HC1" = vcovHC(m_post, type = "HC1")),
    estimate = "{estimate} ({std.error})",
    stars = TRUE
  )
}
# Outline solution note: CV LASSO drops ~3 variables and athlete’s coef ~ 0.124 in post-lasso OLS,
# with athlete significant. Your actual numbers depend on the data draw and seed. :contentReference[oaicite:12]{index=12}

# --- Optional: covariance matrix of regressors (clean) ----------------------- -
# The exam’s “J” focuses on LASSO, but if you need a covariance matrix, do it on numeric regressors.
num_vars <- c("sat","tothrs","athlete","hsize","hsrank","hsperc","female","white","black","fem_ath","bl_ath","wh_ath")
cov_mat <- GPA %>%
  dplyr::select(dplyr::all_of(num_vars)) %>%
  as.data.frame() %>%
  cov(use = "pairwise.complete.obs")
cov_mat
# Avoid cov(x,y,...) misuse: cov() takes two vectors or a numeric matrix/data.frame.

# --- References for reasoning in comments ------------------------------------ -
# • Exam question text and variable list. :contentReference[oaicite:13]{index=13}
# • Outline solutions guidance for (b),(f),(g),(h),(j). :contentReference[oaicite:14]{index=14}
# • Wooldridge on interaction F-tests and group differences (Ch.7). 


# PROBLEMS SETS ----

## P1 ----
# Packages
library(haven)
library(car)        # linearHypothesis, vif
library(lmtest)     # coeftest
library(sandwich)   # vcovHC

# Data import (keep your paths)
dfdis <- read_dta("Opgaver datasæt/P1/ps1 data (1)/discrim.dta")
dfkie <- read_dta("Opgaver datasæt/P1/ps1 data (1)/KIELMC.DTA")
dfvot <- read_dta("Opgaver datasæt/P1/ps1 data (1)/VOTE1.DTA")

# ---------------------------------------------------------------------------- --
# 1) DISCRIM.dta
df <- dfdis

# (a) Baseline: log(price) on shares and income/poverty (log–level)
m0 <- lm(lpsoda ~ prpblck + lincome + prppov, data = df)
summary(m0)                              # classical SEs
coeftest(m0, vcov. = vcovHC(m0, "HC1"))  # robust SEs
nobs(m0); AIC(m0); c(R2=summary(m0)$r.squared, aR2=summary(m0)$adj.r.squared)
# Log–level note: a 0.10 increase in prpblck (10 pp) ≈ 0.10*beta percent change in price.

# (b) Correlation between income and poverty share
ct <- cor.test(df$lincome, df$prppov, method = "pearson")
ct$estimate; ct$conf.int; ct$p.value

# (c) Add housing value elasticity (log–log via lhseval)
m1 <- lm(lpsoda ~ prpblck + lincome + prppov + lhseval, data = df)
summary(m1)
coeftest(m1, vcov. = vcovHC(m1, "HC1"))
linearHypothesis(m1, "lhseval = 0")      # H0: elasticity on lhseval = 0

# (d) Joint test for lincome and prppov; multicollinearity check
linearHypothesis(m1, c("lincome = 0", "prppov = 0"))
vif(m0); vif(m1)

# (e) Fit comparison
c(
  n_m0 = nobs(m0), n_m1 = nobs(m1),
  R2_m0 = summary(m0)$r.squared, R2_m1 = summary(m1)$r.squared,
  aR2_m0 = summary(m0)$adj.r.squared, aR2_m1 = summary(m1)$adj.r.squared,
  AIC_m0 = AIC(m0), AIC_m1 = AIC(m1)
)
# Prefer m1 if aR2 increases and AIC decreases materially.

# ---------------------------------------------------------------------------- --
# 2) KIELMC.DTA (1981 cross-section)
# intst/lintst = distance to interstate (not interest). ldist = log distance to incinerator.
df <- subset(dfkie, year == 1981)

# (a) Bivariate: price on log distance to incinerator
k0 <- lm(lprice ~ ldist, data = df)
summary(k0)
coeftest(k0, vcov. = vcovHC(k0, "HC1"))
nobs(k0); AIC(k0); c(R2=summary(k0)$r.squared, aR2=summary(k0)$adj.r.squared)

# (b) Add property characteristics + interstate distance
k1 <- lm(lprice ~ ldist + lintst + larea + lland + rooms + baths + age, data = df)
summary(k1)
coeftest(k1, vcov. = vcovHC(k1, "HC1"))
nobs(k1); AIC(k1); c(R2=summary(k1)$r.squared, aR2=summary(k1)$adj.r.squared)

# (c) Quadratic in interstate distance (use provided lintstsq)
k2 <- lm(lprice ~ ldist + lintst + lintstsq + larea + lland + rooms + baths + age, data = df)
summary(k2)
coeftest(k2, vcov. = vcovHC(k2, "HC1"))
linearHypothesis(k2, c("lintst = 0", "lintstsq = 0"))   # test the quadratic jointly
# Marginal effect of interstate at mean lintst:
Lbar <- mean(df$lintst, na.rm = TRUE)
me_lintst_at_mean <- coef(k2)["lintst"] + 2*coef(k2)["lintstsq"]*Lbar
me_lintst_at_mean

# (d) Optional: quadratic in ldist (avoid sqrt(ldist) mixing)
k3 <- lm(lprice ~ ldist + I(ldist^2) + lintst + lintstsq + larea + lland + rooms + baths + age, data = df)
summary(k3)
coeftest(k3, vcov. = vcovHC(k3, "HC1"))
linearHypothesis(k3, c("ldist = 0", "I(ldist^2) = 0"))

# --------------------------------------------------------------------------- ---
# 3) VOTE1.DTA
df <- dfvot

# (a) Baseline: vote share on log expenditures and party strength
v0 <- lm(voteA ~ lexpendA + lexpendB + prtystrA, data = df)
summary(v0)
coeftest(v0, vcov. = vcovHC(v0, "HC1"))
nobs(v0); AIC(v0); c(R2=summary(v0)$r.squared, aR2=summary(v0)$adj.r.squared)
# Semi-elasticities: 1% ↑ in A’s spend ⇒ 0.01*beta_A percentage-points in voteA; similarly for B.

# (b) Net effect of symmetric 1% increases for both sides
# H0: beta_lexpendA + beta_lexpendB = 0
linearHypothesis(v0, "lexpendA + lexpendB = 0")

# (c) Alternative parameterization (optional)
# df$net_log_spend <- df$lexpendA + df$lexpendB
# v1 <- lm(voteA ~ net_log_spend + prtystrA, data = df)
# summary(v1); coeftest(v1, vcov. = vcovHC(v1, "HC1"))


## ============================================================================= -
## P2 ------
# Fully commented solutions (robust OLS, White/BP tests, FGLS) --
## =============================================================================- 

## Housekeeping -------------------------------------------------------------- -
## Clear workspace (optional). Comment out if you prefer keeping objects.
rm(list = ls())

## Load packages used below
library(haven)     # read_dta()
library(car)       # hccm(), linearHypothesis()
library(lmtest)    # coeftest(), bptest()
library(sandwich)  # vcovHC() alternatives (HC1, HC3, ...)

## =============================== Problem 4 ================================== -
## Data: hprice1.dta
## --------------------------------------------------------------------------- -

## 4) Load data -------------------------------------------------------------- -
## Replace the path with your local path. Use forward slashes for portability.
hprice1 <- read_dta("path/to/hprice1.dta")

## 4a) OLS and heteroskedasticity-robust SEs -------------------------------- -
## Model: log(price) on log(lotsize), log(square feet), bedrooms
reg4_1 <- lm(lprice ~ llotsize + lsqrft + bdrms, data = hprice1)

## Classical OLS summary (assumes homoskedasticity)
summary(reg4_1)

## Robust coefficient test using White's HC0 variance (car::hccm)
## coeftest() prints estimates with robust SEs, t-stats, and p-values
coeftest(reg4_1, vcov = hccm(reg4_1, type = "hc0"))

## NOTE:
## - summary(reg4_1) keeps OLS coefficients identical to coeftest().
## - Only the standard errors and test statistics change under robust vcov.

## 4b) White test (general heteroskedasticity) via auxiliary regression ----- --
## White test regresses squared OLS residuals on regressors, their squares,
## and their pairwise interactions. H0: homoskedasticity (no explanatory power).

## Build residuals and polynomial/interaction terms explicitly (as professor did)
hprice1$resid            <- resid(reg4_1)
hprice1$residsq          <- hprice1$resid^2
hprice1$llotsizesq       <- hprice1$llotsize^2
hprice1$lsqrftsq         <- hprice1$lsqrft^2
hprice1$bdrmssq          <- hprice1$bdrms^2
hprice1$llotsizelsqrft   <- hprice1$llotsize * hprice1$lsqrft
hprice1$llotsizebdrms    <- hprice1$llotsize * hprice1$bdrms
hprice1$lsqrftbdrms      <- hprice1$lsqrft  * hprice1$bdrms

## Auxiliary White regression
reg4_white <- lm(
  residsq ~ llotsize + lsqrft + bdrms +
    llotsizesq + lsqrftsq + bdrmssq +
    llotsizelsqrft + llotsizebdrms + lsqrftbdrms,
  data = hprice1
)
sum_white <- summary(reg4_white)
sum_white

## White test F-statistic and p-value (joint significance of all aux terms)
F_stat_white <- sum_white$fstatistic[1]  # model F-statistic
## df1 = number of regressors excluding the intercept in the aux model
df1_white <- sum_white$fstatistic[2]
## df2 = residual degrees of freedom in the aux model
df2_white <- sum_white$fstatistic[3]

## Upper-tail p-value for the White test
p_value_white <- pf(F_stat_white, df1_white, df2_white, lower.tail = FALSE)
p_value_white
## Interpretation:
## - Small p-value -> reject H0 -> evidence of heteroskedasticity.
## - Large p-value -> fail to reject -> no evidence against homoskedasticity.

## (Alternative) Breusch–Pagan test as a compact check (same null)
bptest(reg4_1, ~ llotsize + lsqrft + bdrms + I(llotsize^2) + I(lsqrft^2) +
         I(bdrms^2) + I(llotsize*lsqrft) + I(llotsize*bdrms) +
         I(lsqrft*bdrms), data = hprice1)

## Reset (optional, only if you want a clean slate before Problem 5)
rm(list = ls())

## =============================== Problem 5 ================================== -
## Data: SMOKE.DTA
## --------------------------------------------------------------------------- -

library(haven)
library(car)
library(lmtest)
library(sandwich)

## 5) Load data -------------------------------------------------------------- -
smoke <- read_dta("path/to/SMOKE.DTA")

## 5a) Baseline OLS ---------------------------------------------------------- - 
## Model: cigarettes smoked on income, price, education, age, age^2, restaurant ban
reg5_1 <- lm(cigs ~ lincome + lcigpric + educ + age + agesq + restaurn,
             data = smoke)
summary(reg5_1)

## Heteroskedasticity test: Breusch–Pagan (H0: homoskedasticity)
## Small p-value => reject H0 => evidence of heteroskedasticity
bptest(reg5_1)

## 5b) Feasible GLS (variance function via log(u^2) model) ------------------- -
## Step 1: Obtain OLS residuals and model log(residual^2) on X
luhsq   <- log(resid(reg5_1)^2)
reg5_2  <- lm(luhsq ~ lincome + lcigpric + educ + age + agesq + restaurn,
              data = smoke)

## Step 2: Predict log-variance, exponentiate to get variance proxy h(x)
hat_g   <- fitted(reg5_2)         # predicted log variance
hat_h   <- exp(hat_g)             # variance proxy h(x)

## Step 3: WLS using weights = 1 / h(x)
w_vec   <- 1 / hat_h
reg5_w  <- lm(cigs ~ lincome + lcigpric + educ + age + agesq + restaurn,
              data = smoke, weights = w_vec)
summary(reg5_w)

## 5c) Check if heteroskedasticity remains after FGLS ------------------------ -
## Idea: in a correct FGLS, transformed residuals (u / sqrt(h)) should have
## constant variance. A simple White-style check:
u_hat      <- resid(reg5_w)              # WLS residuals
y_hat      <- fitted(reg5_w)             # WLS fitted values
u_tilde_sq <- (u_hat / sqrt(hat_h))^2    # transformed squared residuals
y_tilde    <-  y_hat / sqrt(hat_h)
y_tilde_sq <-  y_tilde^2

## Auxiliary regression: u_tilde^2 on y_tilde and y_tilde^2 (no weights)
reg5_3 <- lm(u_tilde_sq ~ y_tilde + y_tilde_sq)
summary(reg5_3)

## Joint test H0: coefficients on y_tilde and y_tilde_sq are zero
## If we fail to reject, remaining heteroskedasticity is not detected.
car::linearHypothesis(reg5_3, c("y_tilde = 0", "y_tilde_sq = 0"))

## Notes:
## - Do NOT use weights in this auxiliary regression.
## - Do NOT mix fitted values from another model or dataset.
## - If H0 is rejected, FGLS variance model may be misspecified.

## 5d) Reporting guidance ----------------------------------------------------- -
## - Report OLS with robust SEs as the default (coeftest + vcovHC).
## - Optionally report FGLS if it meaningfully reduces SEs and passes 5c.
## - State hypotheses, test names, statistics, and p-values clearly.



############################################################-
## P3 — Robust, reusable solutions with diagnostics ----
# Dependencies
############################################################-
# Core I/O + data handling
library(haven)      # read_dta
library(dplyr)      # mutate, across, pipes (optional but handy)

# Time-series / regression tooling
library(zoo)        # dynlm dependency
library(dynlm)      # time-series OLS with lags/differences
library(lmtest)     # bgtest, coeftest, dwtest, anova, etc.
library(sandwich)   # vcovHC, NeweyWest
library(car)        # linearHypothesis

# Helper: robust SE wrappers you can reuse in future assignments
se_hc <- function(mod, type = "HC1") lmtest::coeftest(mod, vcov = sandwich::vcovHC(mod, type = type))
se_nw <- function(mod, lag = NULL) {
  # If lag not given, use a common rule of thumb bandwidth (floor is important)
  if (is.null(lag)) lag <- max(1L, floor(4 * (nobs(mod) / 100)^(2/9)))
  lmtest::coeftest(mod, vcov = sandwich::NeweyWest(mod, lag = lag))
}

############################################################-
# Load data
############################################################-
INTDEF   <- read_dta("Opgaver datasæt/P3/ps3 data/INTDEF.DTA")
okun_raw <- read_dta("Opgaver datasæt/P3/ps3 data/okun.dta")
jtrain98 <- read_dta("Opgaver datasæt/P3/ps3 data/jtrain98.dta")

############################################################-
# 1) Okun: pcrgdp_t = β0 + β1 * Δu_t + u_t
# Key points:
# - Use Δu (change in unemployment) as regressor (not u in levels).
# - Test AR(1) with BG on the Δ-specification.
# - Breusch–Pagan in the simple regression: regress û^2 on Δu.
# - Compare OLS, HC, and Newey–West SEs. Test H0: β1 = −2.
############################################################-

# Convert to ts and build Δu directly with dynlm’s d() to avoid manual differencing bugs.
okun_ts <- ts(okun_raw)  # dynlm prefers ts/zoo

# Main regression: pcrgdp ~ d(u)
m_okun <- dynlm(pcrgdp ~ d(u), data = okun_ts)
summary(m_okun)

# Interpretation (concise):
# β1 ≈ −1.59 in typical solutions. Negative as Okun’s law predicts. Strong t-stat with usual OLS SE.

## (a) BG test for AR(1) on the DIFFERENCED model
bgtest(m_okun, order = 1, type = "Chisq")
# Interpretation:
# Expect no evidence of AR(1) here (p large). If p is large, keep OLS; HAC still fine as a safeguard.

## (b) Breusch–Pagan (simple-regression form): regress û^2 on Δu
u2 <- resid(m_okun)^2
bp_aux <- dynlm(u2 ~ d(u), data = okun_ts)
summary(bp_aux)
# Interpretation:
# Slope on d(u) typically insignificant (p ~ 0.11). Little evidence of heteroskedasticity.

## (c) Robust SE comparisons + test H0: β1 = −2
# Usual OLS table:
summary(m_okun)

# HC (White) SE:
se_hc(m_okun, type = "HC0")     # HC0 like professor used; HC1 is fine too

# Newey–West SE:
se_nw(m_okun, lag = 1)          # annual data → lag 1 is a common choice
se_nw(m_okun)                   # same but with automatic bandwidth

# 95% CI for β1 with NW(1)
nw_tab <- se_nw(m_okun, lag = 1)
b1_hat <- nw_tab["d(u)", "Estimate"]
se_b1  <- nw_tab["d(u)", "Std. Error"]
c(b1_hat - 1.96*se_b1, b1_hat + 1.96*se_b1)

# Test H0: β1 = −2 in one line (OLS framework; works with linearHypothesis on OLS vcov).
# If you want HAC-consistent Wald, use 'car::linearHypothesis' on the coefficient vector
# with a robust vcov (sandwich::NeweyWest) via 'vcov.' argument.
linearHypothesis(m_okun, "d(u) = -2")  # OLS test
# HAC Wald for H0: β1=-2
car::linearHypothesis(m_okun, "d(u) = -2", vcov. = sandwich::NeweyWest(m_okun, lag = 1))

# Interpretation:
# HC and NW SEs are larger than OLS SE. With NW, the 95% CI typically contains −2.
# Conclusion: cannot reject β1 = −2 at 5% with NW SEs.

# Alternative coding (more general):
# If you already created Δu by hand: okun_raw <- mutate(okun_raw, du = c(NA, diff(u)));
# and then run lm(pcrgdp[-1] ~ du[-1]) to align lengths. dynlm(d(u)) avoids manual indexing.

############################################################-
# 2) Theory: AR(1) and OLS standard errors
############################################################-
# Notes you can paste in reports:
# Under exogeneity, OLS coefficients remain unbiased/consistent.
# With ρ>0 and persistent regressors, the usual OLS SE formula understates true variability.
# With ρ<0, the usual SE can overstate it. Use HAC (e.g., Newey–West) or model the AR(1) explicitly.

############################################################-
# 3) Given ρ̂ = 0.841, se(ρ̂) = 0.053
############################################################-
rho_hat <- 0.841; se_rho <- 0.053
t_rho <- rho_hat / se_rho
t_rho
# Interpretation: huge t-stat → strong serial correlation. Usual OLS SE invalid.

# DW approximation from ρ̂ (for quick intuition; not for testing):
dw_approx <- 2 * (1 - rho_hat)
dw_approx
# Remedy when keeping OLS: use HAC/Newey–West SE or difference the model if near unit root.

############################################################-
# 4) INTDEF: i3_t = β0 + β1 * inf_t + u_t
# (a) Estimate in levels, OMIT first observation (as per prompt) for later comparison.
# (b) First differences: Δi3_t on Δinf_t.
############################################################-

intdef <- INTDEF

# (a) Levels, drop first obs to match professor’s sample
m1 <- lm(i3[-1] ~ inf[-1], data = intdef)
summary(m1)
# Interpretation:
# Expect β1 ≈ 0.692, se ≈ 0.091 when omitting first obs.
# Strong positive relation in levels.

# (b) First differences
m3 <- lm(ci3 ~ cinf, data = intdef)
summary(m3)
# Interpretation:
# β1 in differences ≈ 0.211 with larger SE relative to level’s slope.
# Often preferred if exogeneity in levels is doubtful; differencing can mitigate endogeneity from persistent shocks.

# Alternative patterns:
# - HAC SE on levels: se_nw(m1, lag = 1)
# - HAC SE on FD:    se_nw(m3, lag = 1)

############################################################-
# 5) JTRAIN98: Training effect on unemployment in 1998
# (a) Report overall means for unem98 and unem96 (not only treated).
# (b) Simple LPM unem98 ~ train.
# (c) Add controls.
# (d) Full regression adjustment with centered interactions; train coeff is ATE at mean covariates.
# (e) Joint test of interactions.
# (f) Verify ATE by two-group regressions and universal regression adjustment (URA) formula.
############################################################-

# (a) Overall means
mean(jtrain98$unem98)
mean(jtrain98$unem96)
# Interpretation:
# ~17.2% unemployed in 1998 vs ~31.2% in 1996 in the full sample.
# Do NOT attribute the drop solely to training; many aggregate factors differ across years.

# (b) Simple LPM
m_b <- lm(unem98 ~ train, data = jtrain98)
summary(m_b)
# Interpretation:
# Coefficient on train positive but insignificant → selection likely drives naive sign.

# Robust SE for LPM are recommended (heteroskedasticity is inherent with binary y)
se_hc(m_b, type = "HC1")

# (c) With controls
m_c <- lm(unem98 ~ train + earn96 + educ + age + married, data = jtrain98)
summary(m_c)
se_hc(m_c, type = "HC1")
# Interpretation:
# train ≈ −0.121, highly significant. Controls address selection on observables.

# (d) Full regression adjustment with centered interactions
df <- jtrain98 %>%
  mutate(
    c_earn = earn96 - mean(earn96, na.rm = TRUE),
    c_educ = educ   - mean(educ,   na.rm = TRUE),
    c_age  = age    - mean(age,    na.rm = TRUE),
    c_mar  = married- mean(married,na.rm = TRUE)
  )

m_d <- lm(unem98 ~ train + earn96 + educ + age + married +
            train:c_earn + train:c_educ + train:c_age + train:c_mar, data = df)
summary(m_d)
se_hc(m_d, type = "HC1")
# Interpretation:
# Coef(train) now equals the ATE at the sample means of covariates (because we centered).
# Expect ≈ −0.123 with se ≈ 0.030. Small change from (c).

# (e) Joint significance of interactions
m_c_restricted <- lm(unem98 ~ train + earn96 + educ + age + married, data = df)
anova(m_c_restricted, m_d)                      # same as linearHypothesis on the block
car::linearHypothesis(m_d, c("train:c_earn=0", "train:c_educ=0", "train:c_age=0", "train:c_mar=0"))
# Interpretation:
# p ≈ 0.79 → fail to reject. No strong evidence of heterogeneous effects along these covariates.

# (f) URA check: two separate regressions, predict for ALL, then average difference
jt_treated <- subset(df, train == 1)
jt_control <- subset(df, train == 0)

# IMPORTANT: within-group regressions should NOT include the train dummy
g1 <- lm(unem98 ~ earn96 + educ + age + married, data = jt_treated)
g0 <- lm(unem98 ~ earn96 + educ + age + married, data = jt_control)

# Predict potential outcomes for everyone
X_all <- model.matrix(~ earn96 + educ + age + married, data = df)
mu1 <- as.numeric(X_all %*% coef(g1))
mu0 <- as.numeric(X_all %*% coef(g0))

tau_hat_ura <- mean(mu1 - mu0)
tau_hat_ura
# Interpretation:
# This equals coef(train) from m_d (up to rounding). m_d is more convenient for SEs.

# Alternatives you can reuse:
# - To compute ATT instead of ATE with the same URA machinery:
att_hat_ura <- mean((mu1 - mu0)[df$train == 1])
att_hat_ura

# - To get robust SE for tau via bootstrap (general pattern for future work):
#   1) Write a function that re-fits g1, g0 on bootstrap resamples, recomputes ATE.
#   2) Use replicate(B, ...) over sample indices and take the sd of the resulting taus.

############################################################-
# END
############################################################-


# Øvelser i bogen ----
### Kapital 4 ----
#### C1 – Campaign Expenditures and Election Outcomes ------
# Question summary:
# (i) Interpret β1.
# (ii) State H0: a 1% increase in A’s expenditures is offset by a 1% increase in B’s.
# (iii) Estimate model and discuss effects.
# (iv) Estimate a model or perform a test that directly tests H0.

# --- Load packages ---
library(wooldridge)
library(dplyr)
library(lmtest)
library(sandwich)
library(car)
library(broom)
library(modelsummary)

# --- Data ---
library(wooldridge)
data("vote1")
head(vote1)

###### (i) Interpretation of β1 ---
# The model: voteA = β0 + β1*log(expendA) + β2*log(expendB) + β3*prtystrA + u
# It is a level–log model. Interpretation:
# β1 represents the *absolute percentage-point change* in voteA
# from a 1% increase in A’s expenditures. 
# Example: if β1 = 0.6, a 1% increase in A’s spending increases vote share by 0.006 points.

###### (ii) Null hypothesis ---
# H0: β1 - β2 = 0
# (a 1% increase in A’s spending has the same absolute effect as a 1% increase in B’s spending, but opposite in sign)

###### (iii) Estimate the given model ---
m <- lm(voteA ~ lexpendA + lexpendB + prtystrA, data = vote1)
summary(m)

# --- Interpretation of coefficients ---
# β1 (lexpendA): effect of A’s spending on A’s vote share.
# β2 (lexpendB): effect of B’s spending on A’s vote share.
# β3 (prtystrA): effect of party strength on A’s vote share.

# --- Hypothesis test H0: β1 - β2 = 0 ---
# Using the linearHypothesis() function:
linearHypothesis(m, "lexpendA - lexpendB = 0")

# This F-test (and associated t-test) directly tests if β1 = β2.
# If p-value < 0.05, reject H0 → spending effects differ significantly.

# --- Manual computation of t-statistic for β1 - β2 ---
b1 <- coef(m)["lexpendA"]
b2 <- coef(m)["lexpendB"]
V <- vcov(m)
se_diff <- sqrt(V["lexpendA","lexpendA"] + V["lexpendB","lexpendB"] - 2*V["lexpendA","lexpendB"])
t_test <- (b1 - b2) / se_diff
p_value <- 2 * pt(abs(t_test), df = m$df.residual, lower.tail = FALSE)
t_test; p_value

# --- Interpretation ---
# If |t| > 1.96 (at 5% level), reject H0.
# This means A’s and B’s spending have significantly different marginal effects.
# If not, their effects cannot be statistically distinguished.

###### (iv) Alternative equivalent model for direct test ---
# Construct a new variable capturing log(expendA) - log(expendB)
vote1 <- vote1 %>%
  mutate(new_var1 = lexpendA - lexpendB)

m1 <- lm(voteA ~ new_var1 + prtystrA, data = vote1)
summary(m1)

# In this specification, the coefficient on new_var1 equals (β1 - β2).
# The t-statistic in this regression directly tests H0: β1 - β2 = 0.

# --- Extracting standard error for new_var1 ---
se_new_var1 <- summary(m1)$coefficients["new_var1", "Std. Error"]
se_new_var1

# --- Interpretation of results (typical empirical pattern) ---
# β1 (A’s spending): positive and significant → A’s expenditures increase A’s vote share.
# β2 (B’s spending): negative and significant → B’s expenditures decrease A’s vote share.
# Test (H0: β1 - β2 = 0): usually rejected → spending asymmetry significant.
# Party strength (prtystrA): positive → stronger base support increases votes.

# --- Summary ---
# (i) β1: absolute change in voteA for 1% change in A’s expenditures.
# (ii) H0: β1 - β2 = 0.
# (iii) Estimate via OLS; interpret signs and significance.
# (iv) Test H0 using linearHypothesis() or reparameterized model (new_var1).
# Conclude based on t-statistic or p-value.

###### C2 – LAW SCHOOL SALARY MODEL ----
# Objective:
# Examine determinants of median starting salary of law school graduates (lsalary)
# using the LAWSCH85 dataset.

# (i) Test if school rank has a ceteris paribus effect on salary
# (ii) Test if LSAT and GPA are individually or jointly significant
# (iii) Test if clsize and faculty should be added
# (iv) Discuss omitted factors possibly affecting rank

# --- Load packages ---
library(wooldridge)
library(dplyr)
library(lmtest)
library(sandwich)
library(car)

# --- Load data ---
data("lawsch85")
head(lawsch85)

# ========================================================== - 
# (i) Does school rank affect starting salary?
# Model specification (from Problem 4, Ch.3):
# lsalary = β0 + β1*LSAT + β2*GPA + β3*llibvol + β4*lcost + β5*rank + u
# Null hypothesis: H0: β_rank = 0
# ==========================================================-

m1 <- lm(lsalary ~ LSAT + GPA + llibvol + lcost + rank, data = lawsch85)
summary(m1)

# --- Interpretation:
# rank measures the school's prestige (lower = better). 
# A negative β_rank implies higher-ranked (lower-numbered) schools have higher salaries.
# If p-value on rank < 0.05, reject H0 → rank has a statistically significant effect.

# ==========================================================-
# (ii) Are LSAT and GPA jointly significant for explaining salary?
# Null: H0: β_LSAT = 0 and β_GPA = 0
# Restricted model excludes LSAT and GPA
# ==========================================================-

m2 <- lm(lsalary ~ llibvol + lcost + rank, data = lawsch85)

# Use linearHypothesis for joint F-test:
car::linearHypothesis(m1, c("LSAT = 0", "GPA = 0"))

# --- Manual F-statistic computation (for reference):
SSR_ur <- sum(resid(m1)^2)
SSR_r  <- sum(resid(m2)^2)
n <- nobs(m1)
q <- length(coef(m1)) - length(coef(m2))
df_ur <- n - length(coef(m1))
F_stat <- ((SSR_r - SSR_ur) / q) / (SSR_ur / df_ur)
p_val <- 1 - pf(F_stat, q, df_ur)
F_stat; p_val

# --- Interpretation:
# If p-value < 0.05 → reject H0 → LSAT and GPA are jointly significant.
# Both variables capture student quality and are expected to increase salary.

# ==========================================================-
# (iii) Should clsize or faculty be added?
# Test joint significance of class size and faculty size.
# ==========================================================-

# Augment model with clsize and faculty
m3 <- lm(lsalary ~ LSAT + GPA + llibvol + lcost + rank + clsize + faculty, data = lawsch85)
summary(m3)

# Test H0: β_clsize = 0 and β_faculty = 0
car::linearHypothesis(m3, c("clsize = 0", "faculty = 0"))

# --- Interpretation:
# If p-value > 0.05 → we fail to reject H0, meaning clsize and faculty
# do not add explanatory power beyond the existing model.
# A small increase in R² is expected when adding variables, 
# but significance depends on whether the joint test rejects H0.

# ==========================================================-
# (iv) Discussion: omitted factors that may affect rank
# ==========================================================-
# Possible omitted variables:
# - Historical reputation and alumni network
# - Research funding and endowments
# - Job placement rates and employer connections
# - Faculty quality and student–faculty ratio
# - Geographic location and proximity to legal markets
# - Tuition subsidies or scholarships

# --- Summary of findings:
# (i) Rank has a significant negative effect on salary (better rank → higher salary)
# (ii) LSAT and GPA are jointly significant indicators of student ability
# (iii) clsize and faculty do not materially improve explanatory power
# (iv) Rank may depend on unobserved prestige and quality indicators not in the dataset

#### C3 – Linear combination θ1 = 150*β1 + β2 in log-price model ----
# Model: lprice = β0 + β1*sqrft + β2*bdrms + u
# Goal:
# (i) Estimate θ1 (effect on log(price) of adding a 150-sqft bedroom: +150 sqft and +1 bedroom).
# (ii) Express β2 in terms of θ1 and β1, and show substitution.
# (iii) Compute SE(θ1) and a 95% CI; map to % change in price.

# --- Setup ---
library(wooldridge)
library(lmtest)
library(sandwich)

data("hprice1")

# --- Baseline model ---
m <- lm(lprice ~ sqrft + bdrms, data = hprice1)
summary(m)

###### (i) θ1 estimate as a linear combination of coefficients ---
# θ1 = 150*β_sqrft + 1*β_bdrms
cvec <- c(sqrft = 150, bdrms = 1)                          # weights for β1 and β2
b    <- coef(m)[names(cvec)]
V    <- vcov(m)[names(cvec), names(cvec)]

theta_hat <- as.numeric(c(cvec %*% b))                     # point estimate

###### (iii) Standard error and 95% CI (classical) ---
se_theta  <- sqrt( t(cvec) %*% V %*% cvec )
df        <- m$df.residual
crit      <- qt(0.975, df)
ci_lo     <- theta_hat - crit * se_theta
ci_hi     <- theta_hat + crit * se_theta

theta_hat; se_theta; c(ci_lo, ci_hi)

# Optional: HC1-robust SE and CI
V_HC1     <- sandwich::vcovHC(m, type = "HC1")[names(cvec), names(cvec)]
se_theta_HC1 <- sqrt( t(cvec) %*% V_HC1 %*% cvec )
ci_lo_HC1 <- theta_hat - crit * se_theta_HC1
ci_hi_HC1 <- theta_hat + crit * se_theta_HC1
se_theta_HC1; c(ci_lo_HC1, ci_hi_HC1)

# --- Percentage change in price implied by θ1 (interpretation) ---
# Approximate % change: 100 * θ1
pct_approx <- 100 * theta_hat
# Exact % change: 100 * (exp(θ1) - 1)
pct_exact  <- 100 * (exp(theta_hat) - 1)
pct_approx; pct_exact

###### (ii) Express β2 in terms of θ1 and β1 and plug back ---
# From θ1 = 150*β1 + β2  =>  β2 = θ1 - 150*β1
beta1 <- coef(m)[["sqrft"]]
beta2_from_theta <- theta_hat - 150 * beta1
beta2_from_theta

# Plug into the equation (algebraic form, not re-estimation):
# lprice = β0 + β1*sqrft + (θ1 - 150*β1)*bdrms + u
#        = β0 + β1*(sqrft - 150*bdrms) + θ1*bdrms + u
# This identity shows why θ1 measures the *combined* log-price effect
# of adding a 150-sqft bedroom (Δsqrft = +150, Δbdrms = +1).

# --- Sanity check via reparameterization (yields same θ1 and SE) ---
# Regress with z1 = (sqrft - 150*bdrms) and bdrms; coefficient on bdrms = θ1
hprice1$z1 <- with(hprice1, sqrft - 150*bdrms)
m_repar <- lm(lprice ~ z1 + bdrms, data = hprice1)
summary(m_repar)$coefficients["bdrms", ]                   # estimate and SE match theta_hat and se_theta

#### C5 – MLB1 salary model: specification checks and hypothesis tests ----
# Tasks:
# (i) Drop rbisyr from eq. (4.31) and assess hrunsyr’s significance and coefficient size.
# (ii) Add runsyr, fldperc, sbasesyr and identify which are individually significant.
# (iii) Test the joint significance of bavg, fldperc, sbasesyr.

# --- Setup ---
library(wooldridge)
library(lmtest)
library(sandwich)
library(car)
library(broom)

data("mlb1")

# Eq. (4.31) baseline
m_full <- lm(lsalary ~ years + gamesyr + bavg + hrunsyr + rbisyr, data = mlb1)
summary(m_full)

# (i) Drop rbisyr and compare hrunsyr
m_drop_rbi <- lm(lsalary ~ years + gamesyr + bavg + hrunsyr, data = mlb1)
summary(m_drop_rbi)

# Programmatic comparison of hrunsyr: coefficient and p-value
cmp <- function(m, term) {
  s <- summary(m)$coef
  c(beta = s[term,"Estimate"], se = s[term,"Std. Error"], p = s[term,"Pr(>|t|)"])
}
cbind(full = cmp(m_full,"hrunsyr"), drop_rbi = cmp(m_drop_rbi,"hrunsyr"))

# Optional: HC1-robust check
ct_full     <- coeftest(m_full, vcov. = vcovHC(m_full, type = "HC1"))
ct_drop_rbi <- coeftest(m_drop_rbi, vcov. = vcovHC(m_drop_rbi, type = "HC1"))
ct_full["hrunsyr", c("Estimate","Std. Error","Pr(>|t|)")]
ct_drop_rbi["hrunsyr", c("Estimate","Std. Error","Pr(>|t|)")]

# Interpretation guide:
# - If p-value on hrunsyr falls and |beta| rises after dropping rbisyr, multicollinearity with rbisyr was diluting hrunsyr.

# (ii) Add runsyr, fldperc, sbasesyr
m_aug <- lm(lsalary ~ years + gamesyr + bavg + hrunsyr + runsyr + fldperc + sbasesyr, data = mlb1)
summary(m_aug)

# Extract individual significance for the three added factors
tidy(m_aug) |>
  dplyr::filter(term %in% c("runsyr","fldperc","sbasesyr")) |>
  dplyr::select(term, estimate, std.error, p.value)

# Optional: HC1-robust p-values for those three
ct_aug <- coeftest(m_aug, vcov. = vcovHC(m_aug, type = "HC1"))
ct_aug[c("runsyr","fldperc","sbasesyr"), c("Estimate","Std. Error","Pr(>|t|)")]

# Interpretation guide:
# - A variable is individually significant at 5% if its p-value < 0.05.

# (iii) Joint significance of bavg, fldperc, sbasesyr in m_aug
car::linearHypothesis(m_aug, c("bavg = 0", "fldperc = 0", "sbasesyr = 0"))

# Interpretation guide:
# - Reject H0 if F-test p-value < 0.05 → the group adds explanatory power jointly.


#### C6 – WAGE2: H0 that exper effect equals tenure effect ----
# Model: lwage = β0 + β1*educ + β2*exper + β3*tenure + u
# H0: β2 = β3  ⇔  θ = β2 - β3 = 0

# --- Setup ---
library(wooldridge)
library(dplyr)
library(lmtest)
library(sandwich)
library(car)

data("wage2")

# --- Baseline model ---
m <- lm(lwage ~ educ + exper + tenure, data = wage2)
summary(m)

# --- Test H0 directly with an F/t test (recommended) ---
car::linearHypothesis(m, "exper - tenure = 0")

# --- CI-based null test for θ = βexper - βtenure (classic) ---
C <- c(exper = 1, tenure = -1)                # contrast weights
b <- coef(m)[names(C)]
V <- vcov(m)[names(C), names(C)]
theta_hat <- as.numeric(C %*% b)
se_theta  <- sqrt(t(C) %*% V %*% C)
df <- m$df.residual
crit <- qt(0.975, df)
ci_theta <- c(theta_hat - crit*se_theta, theta_hat + crit*se_theta)
theta_hat; se_theta; ci_theta
# Decision: reject H0 if 0 is outside ci_theta.

# --- HC1-robust CI for θ (optional robustness) ---
V_HC1 <- sandwich::vcovHC(m, type = "HC1")
V_HC1_theta <- V_HC1[names(C), names(C)]
se_theta_HC1 <- sqrt(t(C) %*% V_HC1_theta %*% C)
ci_theta_HC1 <- c(theta_hat - crit*se_theta_HC1, theta_hat + crit*se_theta_HC1)
se_theta_HC1; ci_theta_HC1

# --- Equivalent reparameterization check (θ appears as a coefficient) ---
wage2 <- wage2 %>% mutate(z = exper - tenure)
m_rep <- lm(lwage ~ educ + z + tenure, data = wage2)  # keep tenure so that coef(z) = βexper - βtenure
summary(m_rep)$coefficients["z", ]                    # estimate and SE match theta_hat and se_theta
confint(m_rep, "z", level = 0.95)                     # CI identical to ci_theta
# Same decision rule: reject H0 if CI for z excludes 0.

#### C7 – TWOYEAR ----
# Tasks:
# (i) Describe phsrank.
# (ii) Add phsrank to eq. (4.26): lwage ~ jc + univ + exper. Test significance and compute 10pp effect.
# (iii) Check if adding phsrank changes conclusions on returns to two- and four-year colleges.
# (iv) Explain and test why 'id' should be insignificant.

library(wooldridge)
library(dplyr)
library(broom)
library(lmtest)
library(sandwich)

data("twoyear")

#  (i) Descriptives 
summ_phs <- twoyear %>% summarise(
  min = min(phsrank, na.rm = TRUE),
  max = max(phsrank, na.rm = TRUE),
  mean = mean(phsrank, na.rm = TRUE)
)
summ_phs

# Use a COMMON SAMPLE for fair comparisons (handles any missing phsrank)
S <- complete.cases(twoyear[, c("lwage","jc","univ","exper","phsrank")])

# ---------- (ii) Base model and augmentation with phsrank -
m_base <- lm(lwage ~ jc + univ + exper, data = twoyear, subset = S)
m_phs  <- lm(lwage ~ jc + univ + exper + phsrank, data = twoyear, subset = S)

summary(m_phs)                                 # usual output
coeftest(m_phs, vcov. = vcovHC(m_phs,"HC1"))   # robust check

# Significance of phsrank and wage effect of +10 percentage points in rank
b_phs   <- coef(m_phs)["phsrank"]
ci_phs  <- confint(m_phs, "phsrank", level = 0.95)
p_phs   <- summary(m_phs)$coef["phsrank","Pr(>|t|)"]

# Log-level interpretation: Δ% wage ≈ 100 * (10*b_phs); exact = 100*(exp(10*b_phs)-1)
effect_10pp_approx <- 100 * (10 * b_phs)
effect_10pp_exact  <- 100 * (exp(10 * b_phs) - 1)
ci_effect_exact    <- 100 * (exp(10 * ci_phs) - 1)

b_phs; p_phs; effect_10pp_approx; effect_10pp_exact; ci_effect_exact

# ---------- (iii) Do returns to jc / univ change substantively? --
tab <- left_join(
  tidy(m_base) %>% filter(term %in% c("jc","univ")) %>%
    select(term, beta_base = estimate, se_base = std.error, p_base = p.value),
  tidy(m_phs) %>% filter(term %in% c("jc","univ")) %>%
    select(term, beta_phs = estimate, se_phs = std.error, p_phs = p.value),
  by = "term"
) %>% mutate(delta = beta_phs - beta_base)
tab
# Interpretation rule:
# - If signs and significance (p-values) for jc/univ are unchanged and deltas are small, conclusions are unchanged.

# ---------- (iv) Add 'id' (a record identifier). Expect insignificance. -
m_id <- lm(lwage ~ jc + univ + exper + phsrank + id, data = twoyear, subset = S)
p_id <- summary(m_id)$coef["id","Pr(>|t|)"]
p_id
# Rationale: 'id' is a non-economic label; with an intercept and no interactions it has no causal content, so it should be statistically insignificant.

#### C8 – 401KSUBS: single-person households, OLS, CI test, and OVB check ----
# Scope: use ONLY fsize == 1. Wealth (nettfa) and income (inc) are in $1,000s.

library(wooldridge)
library(dplyr)
library(lmtest)
library(sandwich)
library(broom)
library(car)

# --- Load and alias ---
if (!exists("k401ksubs")) { data("401ksubs"); if (!exists("k401ksubs")) data("k401ksubs") }
df0 <- k401ksubs

# ---------- (i) Count single-person households -
d_single_all <- subset(df0, fsize == 1)                       # no extra filters for (i)
n_single <- nrow(d_single_all); n_single

# ---------- Analysis sample used for (ii)–(v) --
d <- d_single_all %>% select(nettfa, inc, age) %>% filter(complete.cases(.))
nrow(d)

# ---------- (ii) OLS: nettfa = β0 + β1*inc + β2*age --
m <- lm(nettfa ~ inc + age, data = d)
summary(m)
coeftest(m, vcov. = vcovHC(m, "HC1"))                         # optional robust check

# Interpretation notes:
# - β1: change in net wealth (thousands $) per +$1,000 income.
# - β2: change in net wealth (thousands $) per +1 year of age.
# Surprise check: inspect sign and magnitude of β2 relative to β1.

# ---------- (iii) Intercept meaning --
range(d$inc); range(d$age)
# Intercept = predicted wealth at inc = 0 and age = 0 → outside support → not economically meaningful.

# ---------- (iv) One-sided CI/p-value for H0: β_age = 1 vs H1: β_age < 1 (α = 1%) --
b_age  <- coef(m)["age"]
se_age <- sqrt(vcov(m)["age","age"])
df     <- m$df.residual
t_val  <- (b_age - 1) / se_age
p_one  <- pt(t_val, df, lower.tail = TRUE)                     # one-sided p-value
t_val; p_one; p_one < 0.01                                     # decision at 1%

# (Option) robust one-sided p-value
se_age_hc1 <- sqrt(vcovHC(m, "HC1")["age","age"])
t_val_hc1  <- (b_age - 1) / se_age_hc1
p_one_hc1  <- pt(t_val_hc1, df, lower.tail = TRUE)
t_val_hc1; p_one_hc1

# ---------- (v) Simple regression nettfa ~ inc and OVB diagnostics --
m_simple <- lm(nettfa ~ inc, data = d)
summary(m_simple)

# Compare β_inc across models
beta_inc_simple <- coef(m_simple)["inc"]
beta_inc_multi  <- coef(m)["inc"]
c(beta_inc_simple = beta_inc_simple, beta_inc_multi = beta_inc_multi, delta = beta_inc_simple - beta_inc_multi)

# OVB decomposition: bias ≈ β_age * Cov(inc, age) / Var(inc)
cov_ia <- cov(d$inc, d$age)
var_i  <- var(d$inc)
ovb_pred <- coef(m)["age"] * cov_ia / var_i
c(ovb_pred = ovb_pred)

# Correlation between age and income (sign guides OVB direction)
cor(d$inc, d$age)

# Notes:
# - If Cov(inc, age) ≠ 0, the simple-regression β_inc differs from multiple-regression β_inc by ≈ ovb_pred.
# - Sign of difference matches sign(β_age) * sign(Cov(inc, age)).


#### C9 – DISCRIM: fast-food soda prices ----
# Tasks
# (i) OLS: log(psoda) ~ prpblck + log(income) + prppov. Test β_prpblck ≠ 0 at 5% and 1%.
# (ii) Correlation between log(income) and prppov with two-sided p-value.
# (iii) Add log(hseval). Interpret β_log(hseval) and test H0: β_log(hseval)=0.
# (iv) In (iii), check individual significance of log(income) and prppov; test joint H0.
# (v) Model selection rule of thumb.

# --- Setup ---
library(wooldridge)
library(dplyr)
library(lmtest)
library(sandwich)
library(car)
library(broom)

data("discrim")


# ---------------- (i) Baseline OL<- -
m1 <- lm(lpsoda ~ prpblck + lincome + prppov, data = df)
summary(m1)   
# Robust check (optional)
coeftest(m1, vcov. = vcovHC(m1, "HC1"))

# β_prpblck two-sided tests at 5% and 1%
p_prp <- summary(m1)$coef["prpblck","Pr(>|t|)"]
sig_5  <- p_prp < 0.05
sig_1  <- p_prp < 0.01
c(p_prp = p_prp, sig_5 = sig_5, sig_1 = sig_1)



# ---------------- (ii) Correlation -
ct <- cor.test(df$lincome, df$prppov, alternative = "two.sided", method = "pearson")
ct$estimate    # correlation coefficient
ct$p.value     # two-sided p-value

# ---------------- (iii) Add log(hseval) -


m2 <- lm(lpsoda ~ prpblck + lincome + prppov + lhseval, data = discrim)
summary(m2)
# Robust check (optional)
coeftest(m2, vcov. = vcovHC(m2, "HC1"))

# Two-sided p-value for H0: β_lhseval = 0
p_hse <- summary(m2)$coef["lhseval","Pr(>|t|)"]
p_hse

# Elasticity interpretation aid for lhseval (log-log w.r.t. hseval):
# A 1% increase in hseval changes psoda by approximately 100*β_lhseval percent.

# ---------------- (iv) Individual and joint tests ---
# Individual p-values for lincome and prppov in m2
p_lincome <- summary(m2)$coef["lincome","Pr(>|t|)"]
p_prppov  <- summary(m2)$coef["prppov","Pr(>|t|)"]
c(p_lincome = p_lincome, p_prppov = p_prppov)

# Joint significance: H0: β_lincome = 0 and β_prppov = 0
car::linearHypothesis(m2, c("lincome = 0", "prppov = 0"))  # F-test and p-value

# ---------------- (v) Reporting guidance -
# Prefer m2 if:
# - lhseval is a valid control for local housing market differences;
# - adjusted R² is higher without inflated multicollinearity (check VIF);
# - key coefficients (e.g., prpblck) are stable in sign and magnitude vs m1.
car::vif(m1); car::vif(m2)  # diagnostics

#### C10 – ELEM94_95: salary, benefits ratio, enrollment, staff, lunch ----
# Goal:
# (i) Simple OLS: lavgsal ~ bs. Test β_bs = 0 and β_bs = -1.
# (ii) Add lenrol and lstaff. Compare β_bs.
# (iii) Explain SE change on β_bs via residual variance vs multicollinearity.
# (iv) Interpret sign and magnitude of β_lstaff correctly (elasticity).
# (v) Add lunch and interpret whether teachers are compensated for disadvantaged students.
# (vi) Comment on pattern vs Table 4.1 (signs and movements).

library(wooldridge)
library(lmtest)
library(sandwich)
library(car)
library(dplyr)
data("elem94_95")
df <- elem94_95

# ---------------- (i) Simple regression --
m0 <- lm(lavgsal ~ bs, data = df)
summary(m0)

# H0: β_bs = 0  (two-sided)
car::linearHypothesis(m0, "bs = 0")

# H0: β_bs = -1  (two-sided: is slope different from −1?)
car::linearHypothesis(m0, "bs = -1")

# Interpretation:
# - β_bs < 0 and significant vs 0 → higher benefits/salary ratio associates with lower average salary.
# - Fail to reject equality with −1 at 5% if p ≈ 0.17 → not statistically different from −1.

# ---------------- (ii) Add lenrol and lstaff --
m1 <- lm(lavgsal ~ bs + lenrol + lstaff, data = df)
summary(m1)

coef(m0)["bs"]; coef(m1)["bs"]      # compare levels
# Interpretation:
# - |β_bs| shrinks when adding controls. Direction matches Table 4.1 patterns: controls absorb variation in bs.

# ---------------- (iii) Why SE(bs) falls when controls added? ---
# Decompose:
sigma_m0 <- sigma(m0)   # residual std. error
sigma_m1 <- sigma(m1)
c(RSE_simple = sigma_m0, RSE_control = sigma_m1)

# Multicollinearity diagnostic:
car::vif(m1)

# Interpretation:
# - Adding relevant controls reduces residual variance a lot (RSE falls), which lowers SE(bs).
# - Any multicollinearity increase from added regressors is dominated by the drop in error variance.
# - Robust vs classical SE: larger HC1 SEs indicate heteroskedasticity, not “bias from multicollinearity.”

# ---------------- (iv) Interpret lstaff ---
# Both lavgsal and lstaff are logs → β_lstaff is an elasticity.
b_lstaff <- coef(m1)["lstaff"]
b_lstaff
# Interpretation:
# - A 1% increase in staff is associated with approximately 100*b_lstaff percent change in average salary.
# - Negative sign means larger staffs correlate with lower average salaries, holding bs and enrollment fixed.
# - Magnitude: |b_lstaff| is economically material if several percent.

# ---------------- (v) Add lunch and interpret compensation ---
m2 <- lm(lavgsal ~ bs + lenrol + lstaff + lunch, data = df)
summary(m2)

# Effect of a 10 percentage point increase in lunch (if lunch is in percentage points):
b_lunch <- coef(m2)["lunch"]
effect_10pp_approx <- 100 * (10*b_lunch)               # % change approx in lavgsal
effect_10pp_exact  <- 100 * (exp(10*b_lunch) - 1)      # exact % change
c(effect_10pp_approx, effect_10pp_exact)

# Interpretation:
# - β_lunch < 0 and significant → no compensation; higher share of disadvantaged students associates with lower salaries, ceteris paribus.

# ---------------- (vi) Pattern vs Table 4.1 --
# Checklist of signs and movements:
signs <- c(bs = sign(coef(m2)["bs"]),
           lenrol = sign(coef(m2)["lenrol"]),
           lstaff = sign(coef(m2)["lstaff"]),
           lunch = sign(coef(m2)["lunch"]))
signs
# Interpretation:
# - bs negative and remains significant after controls.
# - Adding lenrol,lstaff reduces |β_bs| and SE(bs).
# - lunch enters negative and significant.
# These qualitative results align with the textbook’s Table 4.1 pattern even if exact numbers differ due to sample or software details.

# ---------------- Optional robustness --
coeftest(m0, vcov. = vcovHC(m0, "HC1"))
coeftest(m1, vcov. = vcovHC(m1, "HC1"))
coeftest(m2, vcov. = vcovHC(m2, "HC1"))

# C15 — APPLE: demand for eco-friendly apples
# Scope: (i) baseline OLS and single-term tests; (ii) equal-and-opposite price test;
# (iii) add demographics; (iv) joint test of demographics at 20%; (v) tabulate ecolbs.

library(wooldridge)
library(dplyr)
library(car)
library(lmtest)
library(sandwich)

data("apple")
df <- apple

# ---------------- (i) Baseline model --
m <- lm(ecolbs ~ ecoprc + regprc, data = df)
summary(m)

# Test each slope = 0 (two-sided)
car::linearHypothesis(m, "ecoprc = 0")
car::linearHypothesis(m, "regprc = 0")

# ---------------- (ii) Equal magnitude, opposite sign ---
# H0: beta_ecoprc + beta_regprc = 0
car::linearHypothesis(m, "ecoprc + regprc = 0")

# Optional robustness (HC1)
coeftest(m, vcov. = vcovHC(m, "HC1"))

# ---------------- (iii) Add demographics ---
m1 <- lm(ecolbs ~ ecoprc + regprc + faminc + educ + age + hhsize, data = df)
summary(m1)

# R^2 comparison
c(R2_base = summary(m)$r.squared, R2_ctrl = summary(m1)$r.squared)

# ---------------- (iv) Joint significance of four added covariates --
# H0: faminc = educ = age = hhsize = 0  (20% level check via p-value)
car::linearHypothesis(m1, c("faminc = 0","educ = 0","age = 0","hhsize = 0"))

# Optional robustness (HC1)
coeftest(m1, vcov. = vcovHC(m1, "HC1"))

# ---------------- (v) Tabulate ecolbs distribution --
tab <- sort(table(df$ecolbs), decreasing = TRUE)
tab[1:10]  # top frequencies
most_common    <- names(tab)[1]
second_common  <- names(tab)[2]
c(most_common = most_common, second_common = second_common)

# Diagnostics relevant to MLR.6 (non-normality likely due to discreteness and many zeros)
prop_zero <- mean(df$ecolbs == 0, na.rm = TRUE)
c(share_zero = prop_zero)




### Kapital 5 ----
##### C1 — WAGE1 residual diagnostics for level vs log-level models ----
# Objectives
# (i) OLS: wage ~ educ + exper + tenure. Save residuals and plot histogram.
# (ii) OLS: log(wage) ~ educ + exper + tenure. Save residuals and plot histogram.
# (iii) Assess MLR.6 (normality of errors) visually and numerically; compare models on the SAME sample.

# --- Setup ---
library(wooldridge)
library(dplyr)
library(lmtest)
library(sandwich)
library(moments)        # skewness, kurtosis
library(tseries)        # jarque.bera.test

data("wage1")

# Common sample for apples-to-apples comparison
S <- complete.cases(wage1[, c("wage","lwage","educ","exper","tenure")])

# --- Estimate models ---
m_lvl <- lm(wage  ~ educ + exper + tenure, data = wage1, subset = S)
m_log <- lm(lwage ~ educ + exper + tenure, data = wage1, subset = S)

# --- Residuals ---
e_lvl <- resid(m_lvl)
e_log <- resid(m_log)

# --- Histograms (side-by-side) ---
op <- par(mfrow = c(1,2))
hist(e_lvl, breaks = "FD", col = "lightgray",
     main = "Residuals: level model", xlab = "wage − fitted")
hist(e_log, breaks = "FD", col = "lightgray",
     main = "Residuals: log-level model", xlab = "log(wage) − fitted")
par(op)

# --- QQ-plots (visual normality check) ---
op <- par(mfrow = c(1,2))
qqnorm(e_lvl, main = "QQ: level model"); qqline(e_lvl)
qqnorm(e_log, main = "QQ: log-level model"); qqline(e_log)
par(op)

# --- Numeric diagnostics (skewness, excess kurtosis, JB, Shapiro) ---
diag_tbl <- tibble::tibble(
  model   = c("level","log-level"),
  n       = c(length(e_lvl), length(e_log)),
  RSE     = c(sigma(m_lvl), sigma(m_log)),                 # residual std. error
  adj_R2  = c(summary(m_lvl)$adj.r.squared, summary(m_log)$adj.r.squared),
  AIC     = c(AIC(m_lvl), AIC(m_log)),
  BIC     = c(BIC(m_lvl), BIC(m_log)),
  skew    = c(skewness(e_lvl), skewness(e_log)),
  exkurt  = c(kurtosis(e_lvl) - 3, kurtosis(e_log) - 3),
  JB_p    = c(jarque.bera.test(e_lvl)$p.value, jarque.bera.test(e_log)$p.value),
  SW_p    = c(shapiro.test(e_lvl)$p.value, shapiro.test(e_log)$p.value)  # n <= 5000 OK
)
diag_tbl

# --- Robust SE (optional) for later reference ---
coeftest(m_lvl, vcov. = vcovHC(m_lvl, "HC1"))
coeftest(m_log, vcov. = vcovHC(m_log, "HC1"))

# --- Interpretation guide (keep for review) ---
# - MLR.6 is “closer” when residuals are more symmetric, tails closer to normal in QQ,
#   skew ≈ 0, excess kurtosis ≈ 0, and normality test p-values (JB, Shapiro) are larger.
# - In most wage data, the log-level model improves symmetry and tail behavior relative to the level model.
# - Use the table above: pick the model with higher JB_p / SW_p and better visual QQ alignment as closer to MLR.6.


#### C2 — GPA2: full vs half-sample OLS and SE ratio for hsperc ----

library(wooldridge)
library(dplyr)
library(broom)

data("gpa2")

# Common variables and complete cases
df <- gpa2 %>% select(colgpa, hsperc, sat) %>% filter(complete.cases(.))
n_all <- nrow(df)

# ---------------- (i) All observations --
m_all <- lm(colgpa ~ hsperc + sat, data = df)
summary(m_all)

# ---------------- (ii) First 2,070 observations --
df_2070 <- df %>% slice(1:2070)
n_2070 <- nrow(df_2070)

m_2070 <- lm(colgpa ~ hsperc + sat, data = df_2070)
summary(m_2070)

# ---------------- (iii) SE ratio for hsperc --
se_all  <- summary(m_all)$coefficients["hsperc","Std. Error"]
se_2070 <- summary(m_2070)$coefficients["hsperc","Std. Error"]
ratio_empirical <- se_2070 / se_all

# Theoretical 1/sqrt(n) scaling (eq. 5.10 heuristic)
ratio_theoretical <- sqrt(n_all / n_2070)

c(n_all = n_all, n_2070 = n_2070,
  se_all = se_all, se_2070 = se_2070,
  ratio_empirical = ratio_empirical,
  ratio_theoretical = ratio_theoretical)

# Tidy comparison table (optional)
bind_rows(
  tidy(m_all)  %>% mutate(model = "all"),
  tidy(m_2070) %>% mutate(model = "first_2070")
) %>%
  filter(term %in% c("hsperc","sat")) %>%
  select(model, term, estimate, std.error, statistic, p.value)


#### C3 — BWGHT: LM test for joint significance of motheduc & fatheduc ----
library(wooldridge)

data("bwght")
df <- bwght

# (0) Common sample for the UNRESTRICTED model
vars_unres <- c("bwght","cigs","parity","faminc","motheduc","fatheduc")
S <- complete.cases(df[, vars_unres])

# (1) Restricted model: bwght ~ cigs + parity + faminc
m_r <- lm(bwght ~ cigs + parity + faminc, data = df, subset = S)

# (2) Residuals from restricted model
u_hat <- resid(m_r)

# Build working frame with identical rows and add residuals
wf <- df[S, ]
wf$u_hat <- u_hat

# (3) Auxiliary regression: regress residuals on ALL regressors of the unrestricted model
aux <- lm(u_hat ~ cigs + parity + faminc + motheduc + fatheduc, data = wf)

# (4) LM statistic and p-value (q = 2 restrictions)
LM   <- nobs(aux) * summary(aux)$r.squared
pval <- 1 - pchisq(LM, df = 2) # restriktioner i forhold til motheduc og fatheduc (de er eskkluderet)
list(n = nobs(aux), R2_aux = summary(aux)$r.squared, LM_stat = LM, df = 2, p_value = pval)

# Cross-check: classic F-test using the SAME sample
m_u <- lm(bwght ~ cigs + parity + faminc + motheduc + fatheduc, data = df, subset = S)
anova(m_r, m_u)
# --- LM test results interpretation -

# n = 1191
# → Number of observations used in both restricted and auxiliary regressions.

# R2_aux = 0.00242
# → The auxiliary regression explains only 0.24% of the residual variation.
#   Indicates that the excluded variables (motheduc, fatheduc) add almost no explanatory power.

# LM_stat = 2.88
# → Lagrange Multiplier statistic, computed as LM = n * R2_aux.
#   Used to test joint significance of the excluded variables.

# df = 2
# → Degrees of freedom, equal to the number of restrictions (two coefficients tested).

# p_value = 0.2367
# → Probability of observing an LM ≥ 2.88 under H0 (χ²(2) distribution).
#   Since p > 0.05, we fail to reject H0:
#   motheduc and fatheduc are jointly insignificant given cigs, parity, and faminc.


#### C4 — Skewness checks and log transform ----
# Tasks:
# (i)  401KSUBS with fsize==1: skewness for inc and log(inc)
# (ii) BWGHT2: skewness for bwght and log(bwght)
# (iii) Is "log always makes positive variables nearly normal" true?
# (iv) In regression, should we look at unconditional normality of y or log(y)?

library(wooldridge)
library(moments)  # for skewness() and kurtosis()

## ---------- (i) 401KSUBS: inc vs log(inc) --
data("k401ksubs")
df <- subset(k401ksubs, fsize == 1)

# Skewness and kurtosis (no type argument)
sk_inc  <- c(skew = skewness(df$inc, na.rm = TRUE),
             kurt = kurtosis(df$inc, na.rm = TRUE))
sk_linc <- c(skew = skewness(log(df$inc), na.rm = TRUE),
             kurt = kurtosis(log(df$inc), na.rm = TRUE))

sk_inc; sk_linc
# Interpretation:
# - Positive skew → right tail (non-normal).
# - If log reduces |skew| and kurtosis, log(inc) is closer to normal.

## ---------- (ii) BWGHT2: bwght vs log(bwght) ---
data("bwght2")
g <- bwght2

# lbwght is already the logged variable in the dataset
sk_bwght  <- c(skew = skewness(g$bwght, na.rm = TRUE),
               kurt = kurtosis(g$bwght, na.rm = TRUE))
sk_lbwght <- c(skew = skewness(g$lbwght, na.rm = TRUE),
               kurt = kurtosis(g$lbwght, na.rm = TRUE))

sk_bwght; sk_lbwght

# (iii) The claim "log always makes positive variables nearly normal" is false:
#       Log often reduces right skewness but not guaranteed; depends on data scale.

# (iv) In regression, normality is conditional:
#      We check normality of residuals u|X, not unconditional y or log(y).


#### C5 — HTV: distribution of educ ----

# --- Setup ---
library(wooldridge)   # data
library(dplyr)        # n_distinct
library(ggplot2)      # ggplots
library(moments)      # skewness, kurtosis
library(tseries)      # jarque.bera.test

data("htv")
df <- htv

# --- Baseline model from C11 (for context; not needed for C5(i)-(ii)) ---
m0 <- lm(educ ~ motheduc + fatheduc + abil + I(abil^2), data = df)

# --- C5(i): distinct values and discreteness ---
val_tab <- table(df$educ)             # frequency table
n_dist  <- dplyr::n_distinct(df$educ) # should be 15
rng     <- range(df$educ, na.rm = TRUE)

print(list(distinct_values = n_dist, minmax = rng, value_table_head = head(val_tab)))

# --- C5(ii): distribution visuals (two compact variants) ---

## Variant A: Base R barplot + normal overlay on a numeric histogram
# Bar plot is appropriate because educ is discrete integers
barplot(val_tab,
        main = "educ frequencies (discrete)",
        xlab = "years of education", ylab = "count")

# Numeric histogram with normal overlay (illustrative only)
hist(df$educ,
     breaks = seq(min(df$educ)-0.5, max(df$educ)+0.5, by = 1),
     col = "lightgray", border = "white",
     main = "educ: histogram with normal overlay",
     xlab = "years of education", probability = TRUE)
curve(dnorm(x, mean(df$educ, na.rm = TRUE), sd(df$educ, na.rm = TRUE)),
      add = TRUE, lwd = 2)

## Variant B: ggplot histogram + fitted normal density
ggplot(df, aes(x = educ)) +
  geom_histogram(binwidth = 1, boundary = 0.5) +
  stat_function(fun = dnorm,
                args = list(mean = mean(df$educ, na.rm = TRUE),
                            sd   = sd(df$educ, na.rm = TRUE)),
                linewidth = 1) +
  labs(title = "educ: histogram + normal curve", x = "years of education", y = "density")

# --- QQ-plot (normal reference) ---
qqnorm(df$educ, main = "QQ-plot of educ vs normal"); qqline(df$educ)

# --- Skewness and kurtosis diagnostics (no formulas) ---
sk  <- skewness(df$educ, na.rm = TRUE)
ku  <- kurtosis(df$educ, na.rm = TRUE)          # raw kurtosis (normal ≈ 3)
jb  <- jarque.bera.test(df$educ)                # omnibus normality test

print(list(skewness = sk, kurtosis = ku, JB_stat = unname(jb$statistic), JB_p = jb$p.value))

# --- C5(iii): implication for CLM/MLR.6 (programmatic note) ---
# educ is discrete and non-normal → normality assumption is not credible.
# Use robust (HC) standard errors or rely on large-sample inference if modeling educ.

#### C6 — ECONMATH: score bounds, normality, and robust inference ----

# --- Setup ---
library(wooldridge)   # data
library(ggplot2)      # plots
library(moments)      # skewness, kurtosis
library(lmtest)       # coeftest
library(sandwich)     # vcovHC

data("econmath")
df <- econmath

###### (i) Logical vs. sample bounds for score ---
logical_bounds <- c(0, 100)                 # score is a percent-type scale
sample_minmax  <- range(df$score, na.rm=TRUE)
print(list(logical_bounds = logical_bounds,
           sample_minmax   = sample_minmax))

###### (ii) Model and why MLR.6 may fail (bounded DV) ---
m <- lm(score ~ colgpa + actmth + acteng, data = df)  # OLS remains unbiased under MLR.1–MLR.5

# --- Distribution diagnostics (keep 2 concise variants) ---
## Variant A: Histogram with normal overlay
hist(df$score,
     breaks = 30, probability = TRUE,
     col = "lightgray", border = "white",
     main = "Score: histogram + normal overlay", xlab = "score")
curve(dnorm(x, mean(df$score, na.rm=TRUE), sd(df$score, na.rm=TRUE)),
      add = TRUE, lwd = 2)

## Variant B: QQ-plot versus normal
qqnorm(df$score, main = "QQ-plot: score vs normal")
qqline(df$score, lwd = 2)

# --- Skewness, kurtosis, and omnibus normality test ---
sk <- skewness(df$score, na.rm = TRUE)       # sign = direction; |.| = strength
ku <- kurtosis(df$score, na.rm = TRUE)       # normal ≈ 3
jb <- tseries::jarque.bera.test(df$score)    # large p → closer to normal

print(list(skewness = sk, kurtosis = ku,
           JB_stat = unname(jb$statistic), JB_p = jb$p.value))

###### (iii) Robust inference for H0: beta_acteng = 0 ---
vc_HC1 <- vcovHC(m, type = "HC1")            # heteroskedasticity-consistent covariance
ct      <- coeftest(m, vcov = vc_HC1)        # asymptotically valid t-tests
ct["acteng", , drop = FALSE]                 # report acteng row only

# Notes:
# - MLR.6 (normal u) can fail since score is bounded (0–100), which truncates tails.
# - Use robust SE (as above) or rely on large-sample asymptotics for valid inference.

#### C7 — APPLE: single coeffs, joint test, residual orthogonality, LM test ----

# --- Setup ---
library(wooldridge)
library(car)        # linearHypothesis
library(sandwich)   # vcovHC
library(lmtest)     # coeftest

data("apple")
df <- apple

###### (i) OLS baseline (use one common sample across all steps) ---
m_u <- lm(ecolbs ~ ecoprc + regprc + faminc + age + educ,
          data = df, na.action = na.omit)
summary(m_u)  # n = 660, R^2 ≈ 0.039
# ecoprc and regprc significant; faminc, age, educ not.

###### (ii) Joint test of {faminc, age, educ} = 0 (classical F) ---
# Same sample as m_u
lh_F <- car::linearHypothesis(m_u,
                              c("faminc = 0", "age = 0", "educ = 0"),
                              test = "F")
lh_F
# Target: F ≈ 0.575, p ≈ 0.632 (not jointly significant)

# --- Robust check (optional) ---
vc_HC1 <- sandwich::vcovHC(m_u, type = "HC1")
car::linearHypothesis(m_u,
                      c("faminc = 0", "age = 0", "educ = 0"),
                      vcov. = vc_HC1, test = "F")

###### (iii) Orthogonality demo: residuals from restricted vs included regressors ---
m_r <- lm(ecolbs ~ ecoprc + regprc, data = df, na.action = na.omit)  # restricted
u_r <- resid(m_r)                                                    # OLS residuals
m_ortho <- lm(u_r ~ ecoprc + regprc, data = model.frame(m_r))       # same rows
summary(m_ortho)$r.squared                                           # ≈ 0 by construction

###### (iv) LM test for adding {faminc, age, educ} (q = 3 restrictions) ---
# Auxiliary regression: residuals from the restricted model on ALL regressors
aux <- lm(u_r ~ ecoprc + regprc + faminc + age + educ,
          data = model.frame(m_u))  # ensures same rows as m_u/m_r
R2_aux <- summary(aux)$r.squared
n_aux  <- nobs(aux)
LM     <- n_aux * R2_aux
pLM    <- 1 - pchisq(LM, df = 3) # restriktioner i forhold til nul. faminc = 0, age = 0 og educ = 0
list(n = n_aux, R2_aux = R2_aux, LM_stat = LM, df = 3, p_value = pLM)
# Target: LM ≈ 1.737, p ≈ 0.629 → aligns with the F-test conclusion

# --- Cross-check: partial F using the SAME rows as above ---
anova(m_r, m_u)  # F ≈ 0.575, p ≈ 0.632

# Notes:
# - Step (iii) gives R^2 ≈ 0 because OLS residuals are orthogonal to included regressors.
# - LM uses n*R^2 from the auxiliary regression and df equal to #restrictions (3).
# - Discrepancies usually come from mixing samples (NA handling) or comparing wrong model pairs.

### Kapital 7 ----

#### C1 — GPA1: add parents' college, joint test, quadratic in hsGPA ----

# --- Setup ---
library(wooldridge)
library(car)          # linearHypothesis

data("gpa1")

# Use one common sample across all models to keep n identical
df <- na.omit(subset(gpa1, select = c(colGPA, PC, hsGPA, ACT, mothcoll, fathcoll)))

###### (i) Baseline (7.6) and with parents' college ---
m0 <- lm(colGPA ~ PC + hsGPA + ACT, data = df)                           # Example 7.6
m1 <- lm(colGPA ~ PC + hsGPA + ACT + mothcoll + fathcoll, data = df)     # add parents' college

summary(m0)  # PC should be statistically significant (t ≈ 2.7)
summary(m1)  # Check if PC stays significant after adding mothcoll/fathcoll

# Quick comparison of the PC estimate and p-value before/after
pc_cmp <- rbind(
  m0 = c(beta = coef(m0)["PC"],   p = coef(summary(m0))["PC","Pr(>|t|)"]),
  m1 = c(beta = coef(m1)["PC"],   p = coef(summary(m1))["PC","Pr(>|t|)"])
)
pc_cmp

###### (ii) Joint test: H0: mothcoll = fathcoll = 0 in m1 ---
# Classical partial F (same sample as m1)
lh <- linearHypothesis(m1, c("mothcoll = 0", "fathcoll = 0"), test = "F")
lh   # Expect F ≈ 0.58, p ≈ 0.63 → not jointly significant

###### (iii) Add quadratic in hsGPA and test necessity ---
# Uncentered quadratic can cause collinearity; center hsGPA for stability
df$c_hsGPA <- with(df, hsGPA - mean(hsGPA))

m2  <- lm(colGPA ~ PC + ACT + mothcoll + fathcoll + c_hsGPA + I(c_hsGPA^2), data = df)
summary(m2)

# Test H0: coefficient on c_hsGPA^2 = 0 (is the quadratic needed?)
linearHypothesis(m2, "I(c_hsGPA^2) = 0", test = "F")

# Compare fit metrics (adj. R^2 and AIC); small changes imply no material gain
c(adjR2_m1 = summary(m1)$adj.r.squared,
  adjR2_m2 = summary(m2)$adj.r.squared,
  AIC_m1   = AIC(m1),
  AIC_m2   = AIC(m2))

# Optional: turning point of the hsGPA profile (in original hsGPA units) if quadratic is kept
b1 <- coef(m2)["c_hsGPA"]
b2 <- coef(m2)["I(c_hsGPA^2)"]
turning_point_hsGPA <- mean(df$hsGPA) - b1/(2*b2)
turning_point_hsGPA


#### C2 — WAGE2: dummy-semielasticitet, kvadratikker, race*uddannelse, black*married ----
# Formål: Replikere/udvide log-løn modellen og rette fortolkninger:
# (i) Præcis %-effekt for black via 100·[exp(beta)−1]
# (ii) Test af exper^2 og tenure^2 (fælles og individuelt)
# (iii) Return to education afhænger af race: inkluder hovedleddet black sammen med black:educ
# (iv) Fire grupper via black*married; præcis %-differentiale for married blacks vs married nonblacks

# --- Setup ---
library(wooldridge)
library(car)            # linearHypothesis
data("wage2")

# Fælles sample og variable
df <- na.omit(subset(wage2,
                     select = c(lwage, educ, exper, tenure, married, black, south, urban)
))

###### (i) Baseline log-løn og præcis effekt af black ---
m0 <- lm(lwage ~ educ + exper + tenure + married + black + south + urban, data = df)
summary(m0)  # kontrol: black skal være stærkt signifikant

# Præcis semi-elasticitet for black
b_black <- coef(m0)["black"]
p_black <- coef(summary(m0))["black", "Pr(>|t|)"]
pct_black <- 100 * (exp(b_black) - 1)    # Δ% wage(black vs nonblack)
c(black_beta = b_black, black_p = p_black, black_pct = pct_black)  # ≈ -17.2%

###### (ii) Tilføj kvadratik i exper og tenure og test behovet ---
# Centering er ok for numerisk stabilitet, men ikke påkrævet
df$c_exper  <- with(df, exper  - mean(exper))
df$c_tenure <- with(df, tenure - mean(tenure))

m1 <- lm(lwage ~ educ + c_exper + I(c_exper^2) + c_tenure + I(c_tenure^2) +
           married + black + south + urban, data = df)
summary(m1)

# Fælles F-test: H0: koef. på c_exper^2 og c_tenure^2 = 0
lh_sq <- linearHypothesis(m1, c("I(c_exper^2) = 0", "I(c_tenure^2) = 0"), test = "F")
lh_sq    # forvent F≈1.49, p≈0.226 → ikke fælles signifikante

# Individuelle p-værdier
c(p_exper2 = coef(summary(m1))["I(c_exper^2)", "Pr(>|t|)"],
  p_tenure2 = coef(summary(m1))["I(c_tenure^2)", "Pr(>|t|)"])

###### (iii) Return to education afhænger af race (korrekt specifikation) ---
# Inkludér altid hovedleddet black sammen med interaktionen black:educ
m2 <- lm(lwage ~ educ + black + black:educ + exper + tenure + married + south + urban, data = df)
summary(m2)

# Fortolkning: nonblack return = 100*beta_educ; black return = 100*(beta_educ + beta_black:educ)
b_educ      <- coef(m2)["educ"]
b_bk_educ   <- coef(m2)["black:educ"]
ret_nonblk  <- 100 * b_educ
ret_blk     <- 100 * (b_educ + b_bk_educ)
diff_pp     <- ret_blk - ret_nonblk   # negativt tal = lavere afkast for black
c(ret_nonblack_pct = ret_nonblk, ret_black_pct = ret_blk, diff_pp = diff_pp)

# Test af differentialet: H0: black:educ = 0
linearHypothesis(m2, "black:educ = 0", test = "F")

###### (iv) Fire grupper via black*married og præcis %-differentiale ---
# black*married udvider til black + married + black:married
m3 <- lm(lwage ~ educ + exper + tenure + black*married + south + urban, data = df)
summary(m3)

# Lønkløft mellem married blacks og married nonblacks: δ = β_black + β_black:married
delta   <- coef(m3)["black"] + coef(m3)["black:married"]
pct_gap <- 100 * (exp(delta) - 1)  # præcis procent
c(delta = delta, pct_gap = pct_gap)  # forvent ≈ -16.4%

# --- Output i kort form (kun nøgletal) ---
list(
  i_black_pct = pct_black,
  ii_F_squares = lh_sq$F[2],        # F-stat fra fælles test
  ii_p_squares = lh_sq$`Pr(>F)`[2], # p-værdi
  iii_returns = c(nonblack = ret_nonblk, black = ret_blk, diff_pp = diff_pp),
  iv_married_black_gap_pct = pct_gap
)

# ekstra:

df$female <- as.integer(df$female == 1)   # 0 = male, 1 = female

m <- lm(cumgpa ~ female * (sat + hsperc + tothrs), data = df)
# Expands to: ~ female + sat + hsperc + tothrs +
#              female:sat + female:hsperc + female:tothrs

b  <- coef(m)
slopes_male   <- c(sat=b["sat"], hsperc=b["hsperc"], tothrs=b["tothrs"])
slopes_female <- c(sat=b["sat"]+b["female:sat"],
                   hsperc=b["hsperc"]+b["female:hsperc"],
                   tothrs=b["tothrs"]+b["female:tothrs"])

car::linearHypothesis(m, c("female:sat = 0", "female:hsperc = 0", "female:tothrs = 0"))


#### C3 — MLB1: positions (outfield base), catcher test, fælles test af positioner ----
# Formål: Korrekt test og rapportering i log-løn model med positions-dummies.
# (i) H0: β_catcher = 0  (catchers = outfielders i gennemsnit)
# (ii) H0: β_frstbase = β_scndbase = β_thrdbase = β_shrtstop = β_catcher = 0
# Fortolk altid dummy-effekter i log-level via 100·[exp(β)−1].

# --- Setup ---
library(wooldridge)
library(car)     # linearHypothesis
data("mlb1")

# Ét fælles sample og kun relevante variable
df <- na.omit(subset(mlb1, select = c(lsalary, years, gamesyr, bavg, hrunsyr, rbisyr,
                                      runsyr, fldperc, allstar,
                                      frstbase, scndbase, thrdbase, shrtstop, catcher)))

# --- Model med positioner (outfield er base) ---
m_pos <- lm(lsalary ~ years + gamesyr + bavg + hrunsyr + rbisyr + runsyr +
              fldperc + allstar +
              frstbase + scndbase + thrdbase + shrtstop + catcher,
            data = df)
summary(m_pos)

###### (i) Test: catchers vs outfielders og præcis %-præmie ---
lh_c <- linearHypothesis(m_pos, "catcher = 0", test = "F")
beta_c <- coef(m_pos)["catcher"]
prem_c <- 100 * (exp(beta_c) - 1)               # præcis semielasticitet
c(catcher_beta = beta_c, catcher_pct = prem_c)  # ≈ 28.9%
lh_c                                            # forvent t≈1.93, p≈0.054 → afvis H0 ved 10%, ikke ved 5%

###### (ii) Fælles test: ingen lønforskelle på tværs af positioner ---
lh_joint <- linearHypothesis(m_pos,
                             c("frstbase = 0", "scndbase = 0", "thrdbase = 0", "shrtstop = 0", "catcher = 0"),
                             test = "F"
)
lh_joint   # forvent F≈1.78, p≈0.117 → kan ikke afvise ved 10%/5%

###### (iii) Konsistens-tjek via RSS-fald ved at tilføje positions-dummies samlet ---
# Sammenlign model uden positioner med m_pos
m_base <- lm(lsalary ~ years + gamesyr + bavg + hrunsyr + rbisyr + runsyr + fldperc + allstar, data = df)
anova(m_base, m_pos)     # viser RSS-fald (≈ 174.99 → 170.52; ΔSSR ≈ 4.47) og samme F≈1.78

# --- Kort output til kontrol ---
list(
  i_catcher_F = lh_c$F[2], i_catcher_p = lh_c$`Pr(>F)`[2], i_catcher_pct = prem_c,
  ii_joint_F  = lh_joint$F[2], ii_joint_p = lh_joint$`Pr(>F)`[2]
)

#### C4 — GPA2: quadratic in hsize (vertex + joint test), OVB note, female*athlete, female*sat ----
# Purpose: Clean, testable templates for parts (i)–(v). Interpret log/level correctly.
# Data note: colgpa is in levels → dummy effects are level differences (no exp()).
# Sign expectations: hsperc = class percentile ⇒ higher is worse → expect negative.

# --- Setup ---
library(wooldridge)
library(car)            # deltaMethod, linearHypothesis
data("gpa2")

# One common sample across all specs
df <- na.omit(subset(gpa2, select = c(colgpa, hsize, hsizesq, hsperc, sat, female, athlete)))

###### (i) Baseline with SAT; quadratic in hsize: report vertex and joint relevance ---
m  <- lm(colgpa ~ hsize + hsizesq + hsperc + sat + female + athlete, data = df)
summary(m)

# Turning point of the hsize profile: -β_hsize/(2*β_hsizesq) with delta-method SE
tp <- deltaMethod(coef(m), "-hsize/(2*hsizesq)", vcov(m))  # value, SE, z, p
tp

# Joint test: H0: β_hsize = β_hsizesq = 0
car::linearHypothesis(m, c("hsize = 0", "hsizesq = 0"), test = "F")

###### (ii) OVB remark (code unchanged): dropping SAT can bias athlete/female if SAT correlates with them and with GPA ---
m_omitSAT <- lm(colgpa ~ hsize + hsizesq + hsperc + female + athlete, data = df)
summary(m_omitSAT)   # Compare athlete/female to m to illustrate OVB direction

###### (iii) Women athletes vs women nonathletes (correct model + null) ---
# Difference among women equals β_athlete + β_female:athlete
m3 <- lm(colgpa ~ hsize + hsizesq + hsperc + sat + female*athlete, data = df)
summary(m3)
car::linearHypothesis(m3, "athlete + female:athlete = 0", test = "F")  # H0: no athlete effect among women

###### (iv) Allow SAT effect to differ by gender; test necessity of interaction ---
# female*(athlete + sat) expands to female + athlete + sat + female:athlete + female:sat
m4 <- lm(colgpa ~ hsize + hsizesq + hsperc + female*(athlete + sat) + sat, data = df)
summary(m4)
car::linearHypothesis(m4, "female:sat = 0", test = "F")  # H0: same SAT slope for men and women

###### (v) Quick reporting helpers (exact, minimal) ---
ath_eff_lvl <- coef(m)["athlete"]                 # level difference in GPA (no exponentiation)
c(athlete_level_effect_m = ath_eff_lvl,
  hsperc_sign_check_m    = sign(coef(m)["hsperc"]),
  hsize_turning_point    = tp["Estimate"])



#### C6 — SLEEP75: split by sex, Chow test, joint interaction test, final model ----
# Goal: Reusable template for “group equality” with a binary group (male).
# Units: sleep is MINUTES per week.

# --- Setup ---
library(wooldridge)
library(car)                    # linearHypothesis, matchCoefs
data("sleep75")

# Use one common sample across all models (avoid n-mismatch in F-tests)
keep <- c("sleep","totwrk","educ","age","agesq","yngkid","male")
df <- na.omit(subset(sleep75, select = keep))

# Convenience object with baseline RHS (no group terms)
rhs <- sleep ~ totwrk + educ + age + agesq + yngkid

###### (i) Estimate separately by sex and compare (simple split) ---
m_f <- lm(rhs, data = subset(df, male == 0))     # women
m_m <- lm(rhs, data = subset(df, male == 1))     # men
summary(m_f); summary(m_m)

###### (ii) Chow test: equality of ALL parameters across sexes (intercept + slopes) ---
# Unrestricted pooled model with full set of shifts
mU <- lm(update(rhs, . ~ male*(totwrk + educ + age + agesq + yngkid)), data = df)
summary(mU)

# H0: all male-terms are zero (intercept shift + five slope shifts)
R_chow <- c("male = 0",
            "male:totwrk = 0", "male:educ = 0", "male:age = 0",
            "male:agesq = 0", "male:yngkid = 0")
chow <- linearHypothesis(mU, R_chow, test = "F")
chow        # df1 = 6, df2 = df.residual(mU). Reject at 5%?

###### (iii) Different intercept only; are slope shifts jointly needed? ---
# Keep male main effect; test ONLY interactions
m_int <- lm(update(rhs, . ~ male + . + male:(totwrk + educ + age + agesq + yngkid)), data = df)

# H0: all slope interactions = 0 (no slope differences by sex)
int_test <- linearHypothesis(m_int, matchCoefs(m_int, "^male:"), test = "F")
int_test  # df1 = 5

###### (iv) Final model choice at 5% level (rule-based) ---
if (chow[["Pr(>F)"]][2] >= 0.05) {
  # No differences at all → pooled with no male
  final <- lm(rhs, data = df)
  choice <- "Pooled, no sex differences."
} else if (int_test[["Pr(>F)"]][2] >= 0.05) {
  # Intercepts differ, slopes equal
  final <- lm(update(rhs, . ~ male + .), data = df)
  choice <- "Intercept shift only (male)."
} else {
  # Intercepts and at least one slope differ
  final <- mU
  choice <- "Full set of sex-specific slopes."
}

choice
summary(final)

# --- Reporting helpers (compact) ---
list(
  chow_F = chow$F[2], chow_df1 = chow$Df[2], chow_df2 = chow$Res.Df[2], chow_p = chow$`Pr(>F)`[2],
  inter_F = int_test$F[2], inter_df1 = int_test$Df[2], inter_df2 = int_test$Res.Df[2], inter_p = int_test$`Pr(>F)`[2]
)


#### C7 — WAGE1: gender gap at educ = 12.5 via centering and linear combo tests ----
# Goal: Compute and test the female–male wage gap at a chosen education level.
# Units: lwage = log(hourly wage). educ in years.

# --- Setup ---
library(wooldridge)
library(car)              # linearHypothesis for linear-combination tests
data("wage1")

# Use one common sample across models
keep <- c("lwage","female","educ","exper","tenure")
df <- na.omit(subset(wage1, select = keep))
df$expersq <- df$exper^2
df$tenursq <- df$tenure^2

# Convenience: baseline RHS used in Example 7.18
rhs <- lwage ~ female*educ + exper + expersq + tenure + tenursq

###### (i) Baseline model (as 7.18). Gap at educ = 0 and at 12.5 ---
m0 <- lm(rhs, data = df)
summary(m0)

# Extract coefficients
b_f  <- coef(m0)["female"]
b_fe <- coef(m0)["female:educ"]

# Model-implied gender gap Δ(e) = b_f + b_fe * e
gap_e0   <- b_f
gap_e125 <- b_f + b_fe * 12.5

# Exact percent gaps
pct_e0   <- exp(gap_e0)   - 1
pct_e125 <- exp(gap_e125) - 1

list(
  log_gap_e0   = gap_e0,   pct_gap_e0   = pct_e0,
  log_gap_e125 = gap_e125, pct_gap_e125 = pct_e125
)

# Test significance of the gap at e = 12.5 directly in m0:
# H0: female + 12.5*female:educ = 0
test_e125 <- linearHypothesis(m0, "female + 12.5*female:educ = 0", test = "F")
test_e125

###### (ii) Recenter education at 12.5 so the female coefficient equals Δ(12.5) ---
df$educ_c <- df$educ - 12.5
m1 <- lm(lwage ~ educ + female + female:educ_c + exper + expersq + tenure + tenursq, data = df)
summary(m1)

# Verification: the recentered female coefficient equals Δ(12.5) from m0
all.equal(unname(coef(m1)["female"]), unname(gap_e125))

###### (iii) Report core quantities and compare interpretations ---
list(
  # Evaluation points
  e_ref_m0   = 0,                # female in m0 is Δ(0)
  e_ref_m1   = 12.5,             # female in m1 is Δ(12.5)
  # Point estimates
  m0_female_loggap   = coef(m0)["female"],
  m1_female_loggap   = coef(m1)["female"],
  # Exact percent versions
  m0_female_pctgap   = exp(coef(m0)["female"]) - 1,
  m1_female_pctgap   = exp(coef(m1)["female"]) - 1,
  # Interaction term: does the gap vary with education?
  slope_change_per_year = coef(m0)["female:educ"],
  slope_change_pvalue   = summary(m0)$coef["female:educ","Pr(>|t|)"],
  # Formal test of Δ(12.5)=0 from (i)
  F_e125   = test_e125$F[2],
  df1_e125 = test_e125$Df[2],
  df2_e125 = test_e125$Res.Df[2],
  p_e125   = test_e125$`Pr(>F)`[2]
)


#### C8 — LOANAPP: LPM discrimination, controls, interaction, many CIs (Confidence intervals / konfidensinterval)----

# --- Setup ---
library(wooldridge)      # data
library(dplyr)           # wrangling
library(lmtest)          # coeftest
library(sandwich)        # vcovHC
library(car)             # linearHypothesis
library(multcomp)        # glht, confint for linear combos
library(emmeans)         # contrasts at specified covariate values

library(boot)            # bootstrap CIs
library(broom)           # tidy helpers

data("loanapp")

# Keep only needed vars; drop NAs once to keep n identical across specs
vars <- c("approve","white","hrat","obrat","loanprc","unem","male","married",
          "dep","sch","cosign","chist","pubrec","mortlat1","mortlat2","vr")
df <- na.omit(loanapp[ , vars])

# --- (i) Baseline LPM: approve ~ white ---
m0 <- lm(approve ~ white, data = df)
summary(m0)
coeftest(m0, vcov. = vcovHC(m0, type = "HC1"))            # robust SEs for LPM
# Interpretation (exam):
# sign(β_white) > 0 implies higher approval for whites (discrimination vs minorities).

# --- (ii) Add controls (RRA) ---
m1 <- lm(approve ~ white + hrat + obrat + loanprc + unem + male + married +
           dep + sch + cosign + chist + pubrec + mortlat1 + mortlat2 + vr,
         data = df)
summary(m1)
coeftest(m1, vcov. = vcovHC(m1, type = "HC1"))

# Quick comparison of β_white across (i) and (ii)
cbind(
  baseline = tidy(m0)["white", c("estimate","p.value")],
  controls = tidy(m1)["white", c("estimate","p.value")]
)

# --- (iii) Interaction with debt burden: approve ~ white * obrat + controls ---
m2 <- lm(approve ~ white*obrat + hrat + loanprc + unem + male + married +
           dep + sch + cosign + chist + pubrec + mortlat1 + mortlat2 + vr,
         data = df)
summary(m2)
coeftest(m2, vcov. = vcovHC(m2, type = "HC1"))
# Significance of interaction term:
linearHypothesis(m2, "white:obrat = 0", vcov. = vcovHC(m2, type = "HC1"), test = "F")

# --- (iv) Effect of being white when obrat = 32  ---
# Target: θ = β_white + 32 * β_white:obrat  (partial effect at obrat=32)

## Helper to build L vector for θ
coef_names <- names(coef(m2))      # DENNE ER RET SMART: 
L <- setNames(rep(0, length(coef_names)), coef_names)  # DENNE ER RET SMART!
L["white"] <- 1
L["white:obrat"] <- 32

# FORKLARING: 


# --- CI Method A: Manual delta method (classical OLS vcov) ---
b  <- coef(m2)
V  <- vcov(m2)
theta_hat_A <- as.numeric(crossprod(L, b))
se_A        <- sqrt(as.numeric(t(L) %*% V %*% L))
df_A        <- m2$df.residual
crit_A      <- qt(0.975, df_A)
CI_A        <- c(theta_hat_A - crit_A*se_A, theta_hat_A + crit_A*se_A)
print(list(theta=theta_hat_A, se=se_A, CI_95=CI_A, method="Delta (OLS vcov)"))

# --- CI Method B: Manual delta with HC1 robust vcov ---
V_HC1 <- vcovHC(m2, type = "HC1")
se_B  <- sqrt(as.numeric(t(L) %*% V_HC1 %*% L))
CI_B  <- c(theta_hat_A - crit_A*se_B, theta_hat_A + crit_A*se_B)
print(list(theta=theta_hat_A, se_HC1=se_B, CI_95_HC1=CI_B, method="Delta (HC1 robust)"))

# --- CI Method C: multcomp::glht (classical vcov) ---
K <- rbind(L)                                    # 1 x p contrast matrix #DENNE ER RET SMART 
glht_C <- glht(m2, linfct = K)
summary(glht_C)
confint(glht_C)                                  # 95% CI for θ

# --- CI Method D: multcomp::glht with HC1 robust vcov ---
glht_D <- glht(m2, linfct = K, vcov = V_HC1)
summary(glht_D)
confint(glht_D)

glht_D <- glht(m2, linfct = K) # DENNE ER RET SMART # SE OVENFOR Ved HELPER
summary(glht_D)
confint(glht_D)

# --- CI Method E: emmeans contrast at obrat = 32 (classical vcov) ---
emm_E <- emmeans(m2, ~ white, at = list(obrat = 32))
contrast(emm_E, method = "revpairwise")          # difference white - nonwhite + CI

# --- CI Method F: emmeans with robust HC1 vcov ---
emm_F <- emmeans(m2, ~ white, at = list(obrat = 32), vcov. = V_HC1)
contrast(emm_F, method = "revpairwise")



# --- CI Method H: Nonparametric bootstrap percentile CI ---
set.seed(123)
boot_fun <- function(data, idx) {
  d <- data[idx, ]
  fit <- lm(approve ~ white*obrat + hrat + loanprc + unem + male + married +
              dep + sch + cosign + chist + pubrec + mortlat1 + mortlat2 + vr,
            data = d)
  b  <- coef(fit)
  as.numeric(b["white"] + 32*b["white:obrat"])
}
bt <- boot(df, statistic = boot_fun, R = 1000)
boot_theta <- mean(bt$t)
boot_ci    <- boot.ci(bt, type = c("perc","basic","bca"))
print(list(theta_boot_mean = boot_theta, CI_percentile = boot_ci$percent[4:5],
           CI_BCA = boot_ci$bca[4:5], method="Bootstrap (percentile/BCA)"))



# --- Notes ---
# LPM is linear: the treatment effect of white at obrat=e is β_white + e*β_white:obrat.
# All CI methods above target this same linear functional with different variance estimators or resampling.

#### C9 — 401(k) eligibility (Wooldridge 401ksubs) ----

library(wooldridge)

data("401ksubs")
df <- 401ksubs

# Common sample for required vars
vars <- c("e401k","inc","age","male","pira")
S <- complete.cases(df[, vars])

# (i) Fraction eligible
N  <- sum(S)
frac_eligible <- mean(df$e401k[S])
c(N = N, frac_eligible = frac_eligible)  # should be ~0.392 with N = 9275

# (ii) LPM with age and income in quadratic form + gender
m0 <- lm(e401k ~ male + age + I(age^2) + inc + I(inc^2), data = df, subset = S)
summary(m0)

# (iv) Fitted values outside [0,1]?
p_hat <- fitted(m0)
c(n_lt0 = sum(p_hat < 0), n_gt1 = sum(p_hat > 1))

# (v) Classify at 0.5 and count predicted eligible
y_hat <- as.integer(p_hat >= 0.5)
n_pred_eligible <- sum(y_hat)
c(N = N, predicted_eligible = n_pred_eligible)

# (vi) Percent correctly predicted within ACTUAL groups
y_act <- df$e401k[S]
tab <- table(pred = y_hat, actual = y_act)
tab
n_notelig_actual <- sum(y_act == 0)   # should be 5638
n_elig_actual    <- sum(y_act == 1)   # should be 3637
pct_correct_not_eligible <- tab["0","0"] / n_notelig_actual  # ≈ 0.817
pct_correct_eligible     <- tab["1","1"] / n_elig_actual     # ≈ 0.393
overall_correct <- sum(diag(tab)) / N                         # ≈ 0.651
c(pct_correct_not_eligible, pct_correct_eligible, overall_correct)

# (viii) Add pira; effect and 10% test
m2 <- lm(e401k ~ male + age + I(age^2) + inc + I(inc^2) + pira, data = df, subset = S)
sm2 <- summary(m2)
sm2
pira_coef <- sm2$coef["pira","Estimate"]         # change in probability (in points)
pira_pval <- sm2$coef["pira","Pr(>|t|)"]
c(pira_effect_points = pira_coef, pira_p_value = pira_pval, sig_at_10pct = (pira_pval < 0.10))

#### C10 ---- 
library(wooldridge)

data("nbasal")

df <- nbasal

head(df)

m <- lm(points ~ exper + I(exper^2) + guard + forward, data = df)
summary(m)

show_model_summary <- function(model, digits = 3) {
  if (!inherits(model, "lm")) stop("Input must be an lm object.")
  
  s <- summary(model)
  coefs <- coef(s)
  rse <- s$sigma
  r2 <- s$r.squared
  adjr2 <- s$adj.r.squared
  
  # Equation
  eq <- paste0(
    "ŷ = ",
    round(coefs[1, 1], digits), " + ",
    paste0(round(coefs[-1, 1], digits), "·", rownames(coefs)[-1], collapse = " + ")
  )
  
  cat("\n--- Model in standard form ---\n")
  cat(eq, "\n")
  
  # Coefficients table
  cat("\n--- Coefficients ---\n")
  printCoefmat(coefs, digits = digits, signif.stars = TRUE)
  
  # Model fit stats
  cat("\n--- Fit statistics ---\n")
  cat("Residual standard error (RSE):", round(rse, digits), "\n")
  cat("R-squared:", round(r2, digits), "\n")
  cat("Adjusted R-squared:", round(adjr2, digits), "\n")
}


show_model_summary(m)


# including all three dummy variables would cause the dummy variable trap, violating MLR.3 (perfect collinearity)

# 3: yes, on average, a guard scores about 2.3 points more pr. game, at a 5% siginficance level. (p-value 0.02144)

m1 <- lm(points ~ exper + I(exper^2) + guard + forward + marr, data = df)
show_model_summary(m1)


# Marriage status is not statistically significant at 10% (p-value 0.43), and cannot be assumed to be different than zero. 

m2 <- lm(points ~ marr*(exper + I(exper^2)) + guard + forward, data = df)
show_model_summary(m2)

# with 10% certainty, the interaktive terms between marriage and experience affects points pr. game. Marriage. 
# not strong evidence for martial status affecting points pr. game

# VI:
m3 <- lm(assists ~ exper + I(exper^2) + guard + forward + marr, data = df)
show_model_summary(m3)

# there is practically large difference on how experience affects assists and points. It appears that assists is less likely to be explained by experience
# It also apeears that the forward position is on average has fewer assists than points. However, this is statistically insignificant in both models. 

# the intercept is also alot smaller in the assist model, however insignificant and with higher degrees of standard error.

# The model for assists seem to be better fitted than the one for points. 
# marital status seems to have a smalelr (but insignificant) affect on points /assists, when comparing the assist model with the points pr. game model.

#### C11 ---- 

library(wooldridge)
data("k401ksubs")

df <- k401ksubs

sd <- sd(k401ksubs$nettfa, na.rm = TRUE)
avg <- mean(k401ksubs$nettfa, na.rm = TRUE)
min <- min(k401ksubs$nettfa, na.rm = TRUE)
max <- max(k401ksubs$nettfa, na.rm = TRUE)


sum(complete.cases(df))  # number of usable observations (no missing data)


summary_table <- data.frame(
  Statistic = c("Mean", "SD", "Min", "Max"),
  Value = c(avg, sd, min, max)
)
summary_table

summary(df)

# 2: H0: E(Nettfa|e401k) = E(Nettfa)

t.test(nettfa ~ e401k, data = df, na.rm = TRUE)

# we can reject the null at less than 1% signinficance level (p-value < 2.2e-16)

# nettfa in group 0 (not eligible for 401(k)) estimated to be 11.7, while group 1 is estimated to be 30.53, meaning at eligability status 
# is correlated with expected average nettfa. however, this may not be a causal effect, as it is likely the other way around,
# that eligibility for a 401(k) is predicted by something else that could be correlated with net financial assets. 
# strong evidence against the null, however not necessarily any causality.

11.67677 - 30.53509

# estimated difference:  ca. 18.9 (in thousands) = 18.900 dollars

# 3:

m <- lm(nettfa ~ inc + age + e401k + I(age^2) + I(inc^2), data = df)
show_model_summary <- function(model, digits = 3) {
  if (!inherits(model, "lm")) stop("Input must be an lm object.")
  
  s <- summary(model)
  coefs <- coef(s)
  rse <- s$sigma
  r2 <- s$r.squared
  adjr2 <- s$adj.r.squared
  
  # Equation
  eq <- paste0(
    "ŷ = ",
    round(coefs[1, 1], digits), " + ",
    paste0(round(coefs[-1, 1], digits), "·", rownames(coefs)[-1], collapse = " + ")
  )
  
  cat("\n--- Model in standard form ---\n")
  cat(eq, "\n")
  
  # Coefficients table
  cat("\n--- Coefficients ---\n")
  printCoefmat(coefs, digits = digits, signif.stars = TRUE)
  
  # Model fit stats
  cat("\n--- Fit statistics ---\n")
  cat("Residual standard error (RSE):", round(rse, digits), "\n")
  cat("R-squared:", round(r2, digits), "\n")
  cat("Adjusted R-squared:", round(adjr2, digits), "\n")
}
show_model_summary()
summary(m)

# with a high degree of probability againts it being equal to zero (p = 3.32e-14; less than 1% significance level), 
# we can conclude that eligability of a 401k plan is correlated with a higher
# Net financial assets: 9.700 dollars difference. Again, this might not be causal. 


# 4:
df$age_c <- df$age - 41

m1 <- lm(nettfa ~ inc + e401k*(age_c + I(age_c^2)) + I(inc^2), data = df)

summary(m1)
summary(m)
# centering makes only the normal unquadratic age function statistiaclly significant, at p = 6.44e-07, which a coefficient of 0.65


# firstly, the residuals are smaller within the second model, with centered age
# the coefficients very alot, however, this is expected, as centering based on age
# will need new interpretation
# the coeficient of e401k on the dependent variable, nettfa, is a bit larger in the centered version  
# as age is centered, the affect of being older than 41 has a positive affect on the expected Nettfa, which indicates, that being
# younger than 41 has a negative impact on expected nettfa. The higher statistical value indicates that this isolated is a better 
# fit for a model, based on age
# income is vurtually unchanged. 

# 6:
df$fsize1 <- df$fsize == 1 # base
df$fsize2 <- df$fsize == 2
df$fsize3 <- df$fsize == 3
df$fsize4 <- df$fsize == 4
df$fsize5 <- df$fsize >= 5

m3 <- lm(nettfa ~ inc + age + e401k + I(age^2) + I(inc^2) + fsize2 + fsize3 + fsize4 + fsize5, data = df)
summary(m)


# family size 2 is not statistically significant. size 3 is at 5% level, size 4 and 5 are statisticailly significaint atthe 1% level

# 7:

m_fs1 <- lm(nettfa ~ inc + I(inc^2) + age + I(age^2) + e401k, data = df, subset = fsize == 1)
m_fs2 <- lm(nettfa ~ inc + I(inc^2) + age + I(age^2) + e401k, data = df, subset = fsize == 2)
m_fs3 <- lm(nettfa ~ inc + I(inc^2) + age + I(age^2) + e401k, data = df, subset = fsize == 3)
m_fs4 <- lm(nettfa ~ inc + I(inc^2) + age + I(age^2) + e401k, data = df, subset = fsize == 4)
m_fs5 <- lm(nettfa ~ inc + I(inc^2) + age + I(age^2) + e401k, data = df, subset = fsize >= 5)

SSR_1 <- sum(resid(m_fs1)^2)
SSR_2 <- sum(resid(m_fs2)^2)
SSR_3 <- sum(resid(m_fs3)^2)
SSR_4 <- sum(resid(m_fs4)^2)
SSR_5 <- sum(resid(m_fs5)^2)

SSR_UR <- SSR_1 + SSR_2 + SSR_3 + SSR_4 + SSR_5 

SSR_R <- sum(resid(m3)^2)
g <- 5
k <- 5

df_num <- (k+1)*(g-1)
df_denum <- NROW(df) - g * (k + 1)


Chow_f <- ((SSR_R - SSR_UR)/SSR_UR) * (df_num/df_denum) 

Chow_f
p_val <- pf(Chow_f, df_num, df_denum, lower.tail = FALSE)

c(F = Chow_f, df1 = df_num, df2 = df_denum, p_value = p_val)


### Kapital 8 ----

#### C1 — Group-heteroskedasticity by sex (SLEEP75) -------------------------------

# --- Setup ------------------------------------------------------------------- -
library(wooldridge)   # data
library(lmtest)       # bptest
library(nlme)         # gls, varIdent

data("sleep75")
# One common sample; keep only needed vars
keep <- c("sleep","totwrk","educ","age","agesq","yngkid","male")
df <- na.omit(subset(sleep75, select = keep))

# --- (i) Model specification -------------------------------------------------- -
# Mean equation: sleep = b0 + b1*totwrk + b2*educ + b3*age + b4*agesq + b5*yngkid + b6*male + u
# Heteroskedasticity: Var(u | male=0) = sigma_f^2; Var(u | male=1) = sigma_m^2; no other drivers

# --- (ii) Estimate variances and FGLS ----------------------------------------- -
m_ols <- lm(sleep ~ totwrk + educ + age + agesq + yngkid + male, data = df)  # first-stage OLS
e <- resid(m_ols)

# Group-specific variance estimates from OLS residuals
sigma_f2 <- mean(e[df$male == 0]^2)   # women
sigma_m2 <- mean(e[df$male == 1]^2)   # men
c(sigma_f2 = sigma_f2, sigma_m2 = sigma_m2, ratio_m_over_f = sigma_m2/sigma_f2)

# Feasible GLS with inverse-variance weights by group
w <- ifelse(df$male == 1, 1/sigma_m2, 1/sigma_f2)
m_fgls <- lm(sleep ~ totwrk + educ + age + agesq + yngkid + male, data = df, weights = w)
summary(m_fgls)  # efficient if group-variance model is correct

# --- (iii) Is the variance different across sexes? --------------------------- -
# A) LM/Score test: Breusch–Pagan with male as the only variance regressor
bptest(m_ols, ~ male, data = df)   # H0: equal variances; small p => variances differ

# B) Likelihood ratio using GLS (homoskedastic vs group-hetero)
g_homo <- gls(sleep ~ totwrk + educ + age + agesq + yngkid + male, data = df, method = "REML")
g_hetero <- gls(sleep ~ totwrk + educ + age + agesq + yngkid + male,
                data = df, weights = varIdent(form = ~ 1 | male), method = "REML")
anova(g_homo, g_hetero)            # χ^2(1) test; small p => accept group-heteroskedasticity

# C) Classical F test on residual variances (normality assumption; use with caution)
var.test(e[df$male == 0], e[df$male == 1])  # H0: sigma_f2 = sigma_m2

# Readouts:
# - Compare sigma_f2 vs sigma_m2 and the ratio printed above for (ii)
# - Use p-values from BP, LR, and F to answer (iii)



#(i) Mean model: ( \text{sleep}_i=\beta_0+\beta_1\text{totwrk}_i+\beta_2\text{educ}_i+\beta_3\text{age}_i+\beta_4\text{agesq}_i+\beta_5\text{yngkid}_i+\beta_6\text{male}_i+u_i).
# Variance model: (\operatorname{Var}(u_i|\text{male}_i=0)=\sigma_f^2), (\operatorname{Var}(u_i|\text{male}_i=1)=\sigma_m^2). No other drivers.

# (ii) Estimated variances from OLS residuals:
#   (\hat\sigma_f^2 = 189{,}359), (\hat\sigma_m^2 = 160{,}509), ratio (=0.848) (men/women).
C# onclusion: variance is higher for women.

# (iii) Equality test results:
#   Breusch–Pagan (χ²(1)) p=0.2903; GLS LR (χ²(1)) p=0.1221; F-test p=0.1206.
# Conclusion: fail to reject ( \sigma_f^2=\sigma_m^2 ) at 5% and 10%. Variances are not statistically different.



#### (EGEN) C2 ----
library(wooldridge)
library(sandwich)
data("hprice1")
df <- hprice1

# 1
m0 <- lm(price ~ lotsize + sqrft + bdrms, data = df)
coeftest(m0, vcov. = vcovHC(m0, type = "HC1"))
summary(m0)

# we see that the robust standard error for the intercept is larger, than the usual SE.
# We also see that the lotsize-standard errors are larger when accouting for heteroskedasticity
# The larger standard errors (when robust) for lotsize makes the coefficient insignificant


# the bdrms robust SE are smaller than the usual standard errors, both SE being making them insignificant
# the standard errors on sqrft weakens the significance on the coefficient, however
# this is still signigicant at the less than 1% level. (0 to 10 zeros)

# 2
head(df)
m1 <- lm(lprice ~ llotsize + lsqrft + bdrms, data = df)
coeftest(m1, vcov. = vcovHC(m1, type = "HC1"))
summary(m1)

# again we see that the robust standard errors are larger than the usual standard errors, expect for bdrms
# this weakens the significane of the llotsize from highliy significant (zero% to 6 zeros) to 
# significant at the 1% (p-value 0.00147)

# even though the significance on the other variables are affected, they remain significant at the same level
# however, with commenting is that bdrms 

# 3.
# this suggests that the sample data is not homoskedastitic and that some of the variables have heteroskedasticity. 
# the transformation suggests that the previously significant findings might not be significant if MLR.5 is violated


# NOTE: FEJL I INTERPRETATIOn


#### C2 — Heteroskedasticity-Robust SEs and Log Transformation --------------------

# --- Setup ------------------------------------------------------------------- -
library(wooldridge)   # dataset hprice1
library(lmtest)       # coeftest()
library(sandwich)     # vcovHC()
data("hprice1")
df <- hprice1

# --- (i) Levels model: price in levels --------------------------------------- -
m_lvl <- lm(price ~ lotsize + sqrft + bdrms, data = df)

# Standard OLS output (homoskedasticity assumed)
summary(m_lvl)

# Heteroskedasticity-robust standard errors (HC1)
coeftest(m_lvl, vcov. = vcovHC(m_lvl, type = "HC1"))

# --- Interpretation ---------------------------------------------------------- -
# Coefficients (β-hats) are identical across OLS and HC1.
# The intercept and lotsize SEs increase substantially under HC1,
# indicating heteroskedasticity inflates variance of these estimates.
# lotsize: SE rises from ≈0.00064 → ≈0.00125 → loses significance (p ≈ 0.10).
# sqrft: SE rises from ≈0.013 → ≈0.018 but remains highly significant (<1%).
# bdrms: SE decreases slightly but coefficient remains insignificant.
# Conclusion: heteroskedasticity affects inference; OLS SEs are too small.

# --- (ii) Log model: log(price) and log regressors --------------------------- -
m_log <- lm(lprice ~ llotsize + lsqrft + bdrms, data = df)

# OLS (usual SEs)
summary(m_log)

# Robust SEs (HC1)
coeftest(m_log, vcov. = vcovHC(m_log, type = "HC1"))

# --- Interpretation ---------------------------------------------------------- -
# In the log–log model, HC1 and OLS SEs are much closer in size.
# llotsize: remains significant at 1% (p ≈ 0.002) though SE rises slightly.
# lsqrft: unaffected; stays highly significant (<1%).
# bdrms: minor change; remains significant at 5%.
# Conclusion: the log transformation stabilizes variance and reduces heteroskedasticity.

# --- (iii) Discussion -------------------------------------------------------- -
# The level model (price) exhibits clear heteroskedasticity: robust SEs differ notably.
# The log transformation compresses scale differences, making residual variance more constant.
# Hence, taking logs of price yields a model closer to homoskedastic errors (MLR.5).
# Implication: log-transformed price models are preferred for consistent and efficient inference.




#### (EGEN) C3 ----
library(wooldridge)
data("hprice1")
df <- hprice1

m_log <- lm(lprice ~ llotsize + lsqrft + bdrms, data = df)
library(lmtest)

df$m_res <- (resid(m_log))^2

white <- lm(m_res ~ llotsize + lsqrft + bdrms +
              I(llotsize^2) + I(lsqrft^2) + I(bdrms^2) + I(llotsize*lsqrft) 
            + I(llotsize*bdrms) + I(lsqrft*bdrms), data = df)

summary(white)
LM <- nobs(white) * summary(white)$r.squared

dfW <- length(coef(white)) - 1

pval <- 1 - pchisq(LM, df = dfW)
c(LM = LM, df = dfW, p_value = pval)

# based on the chi-squared p-value for the test, we fail to reject the null that there is homoskedasticity, at a p-value of 0.3882
# We cannot conclude that the model has heteroskedasticity even at a 30% level


#### C3 — Full White test for eq. (8.18): lprice ~ llotsize + lsqrft + bdrms ----
# Goal: Test H0: errors are homoskedastic. Use White’s full auxiliary regression.

# --- Setup ------------------------------------------------------------------- -
library(wooldridge)   # hprice1
library(lmtest)       # bptest()
data("hprice1")
df <- hprice1

# --- Base model (8.18) ------------------------------------------------------- -
m_log  <- lm(lprice ~ llotsize + lsqrft + bdrms, data = df)
uhat2  <- resid(m_log)^2

# --- White auxiliary: levels, squares, and all pairwise interactions --------- -
aux <- lm(uhat2 ~ llotsize + lsqrft + bdrms +
            I(llotsize^2) + I(lsqrft^2) + I(bdrms^2) +
            I(llotsize*lsqrft) + I(llotsize*bdrms) + I(lsqrft*bdrms),
          data = df)

# --- Chi-square form (LM = n * R^2) ---------------------------------------- --
LM  <- nobs(aux) * summary(aux)$r.squared
dfW <- length(coef(aux)) - 1        # exclude intercept
p   <- 1 - pchisq(LM, dfW) # restriktioner i forhold til SMART
c(LM = LM, df = dfW, p_value = p)

# --- Cross-check (equivalent): unstudentized BP with same RHS --------------- --
bptest(m_log,
       ~ llotsize + lsqrft + bdrms +
         I(llotsize^2) + I(lsqrft^2) + I(bdrms^2) +
         I(llotsize*lsqrft) + I(llotsize*bdrms) + I(lsqrft*bdrms),
       data = df, studentize = FALSE)


#### (EGEN) C4 ----

library(wooldridge)
data("vote1")
df <- vote1

names(df)
summary(df)



m0 <- lm(voteA ~ prtystrA + democA + lexpendA + lexpendB, data = df)

summary(m0)
df$uhat <- resid(m0)

m1 <- lm(uhat ~ prtystrA + democA + lexpendA + lexpendB, data = df)
summary(m1)

# you get an R-squared that is almost equal to zero, because the model residuals seem
# not to have any explainatory value. This is a strong indication that MLR.4 (zero conditional mean) 
# is not violated

# 2:
library(lmtest)

uhat2 <- (resid(m0))^2

fitted_m <- fitted(m0)

bp_f <- lm(uhat2 ~ fitted_m + I(fitted_m^2), data = df)

summary(bp_f)

bptest(bp_f)

# with a p-value of 0.06553, we can reject the null (of homoskedasticity) at the 10% level 
# 

# 3


white <- lm(uhat2 ~ prtystrA + democA + lexpendA + lexpendB +
              I(prtystrA^2) + I(democA^2) + I(lexpendA^2) + I(lexpendB^2) + 
              I(prtystrA*democA) + I(prtystrA*lexpendA) + I(prtystrA*lexpendB) +
              I(democA*lexpendA) + I(democA*lexpendB) + I(lexpendA*lexpendB), data = df)

white1 <- summary(white)
F_stat_white <- white1$fstatistic
p_value_white <- pf(F_stat_white[1], F_stat_white[2], F_stat_white[2])

p_value_white
white1

# based on the special case, we can reject the null at the 1% level with a p-value of 0.001973
# very strong evidence for heteroskedasticity

rm(list = ls())


#### C4 — Heteroskedasticity diagnostics on VOTE1 ---------------------------------
# Packages
library(wooldridge)
library(lmtest)     # bptest
library(sandwich)   # vcovHC for robust SEs (not used inside BP/White aux F-tests)

# Data
data("vote1")
df <- vote1

# ----------------------------------------------------------------------------- -
# (i) Main model + residual-on-X regression
# Model: vote share on party strength, Democrat dummy, and log expenditures
m0 <- lm(voteA ~ prtystrA + democA + lexpendA + lexpendB, data = df)
summary(m0)                          # classical SEs (report; robust SEs optional below)
nobs(m0)                             # sample size

# Robust SEs for the main model (good practice, not part of BP/White construction)
lmtest::coeftest(m0, vcov. = sandwich::vcovHC(m0, type = "HC1"))

# Obtain residuals and regress on ALL original regressors (including intercept)
df$uhat <- resid(m0)
m_resid_on_X <- lm(uhat ~ prtystrA + democA + lexpendA + lexpendB, data = df)
summary(m_resid_on_X)

# WHY R^2 = 0 EXACTLY:
# By OLS first-order conditions, residuals are orthogonal to each included regressor
# and to the intercept. Therefore the fitted values in this residual-on-X auxiliary
# regression are identically zero, so R^2 = 0 and every t-statistic = 0.
# MISTAKE BEFORE: Interpreting “R^2 ≈ 0” as evidence for MLR.4 (zero conditional mean).
# It is an algebraic identity, not an empirical check of exogeneity.

# ---------------------------------------------------------------------------- --
# (ii) Breusch–Pagan test (F-form). Regress uhat^2 on ORIGINAL regressors.
# BP auxiliary regression uses the same X as the main model (with intercept).
# MISTAKE BEFORE: You regressed uhat^2 on fitted_m and fitted_m^2.
# That is the White special case, not the BP test.

df$u2 <- df$uhat^2
bp_aux <- lm(u2 ~ prtystrA + democA + lexpendA + lexpendB, data = df)
fs_bp <- summary(bp_aux)$fstatistic          # named numeric: value, df1, df2
fs_bp                                        # inspect F and dfs
p_bp <- pf(fs_bp["value"], fs_bp["numdf"], fs_bp["dendf"], lower.tail = FALSE)
p_bp                                         # report BP F-test p-value

# Optional: studentized BP via lmtest (same null; different small-sample behavior)
bptest(m0, ~ prtystrA + democA + lexpendA + lexpendB, data = df)

# Interpretation template:
# If p_bp < α (e.g., 0.05) ⇒ reject homoskedasticity. Otherwise fail to reject.

# ---------------------------------------------------------------------------- --
# (iii) White test — special case (F-form): uhat^2 on fitted and fitted^2
# This is the “parsimonious” White. It does NOT add all cross terms.
# NOTE: This is the test you effectively ran in (ii) earlier by mistake.
fhat <- fitted(m0)
white_aux_sc <- lm(u2 ~ fhat + I(fhat^2), data = df)
fs_wsc <- summary(white_aux_sc)$fstatistic
p_wsc  <- pf(fs_wsc["value"], fs_wsc["numdf"], fs_wsc["dendf"], lower.tail = FALSE)
fs_wsc; p_wsc

# MISTAKE BEFORE: p-value was computed with wrong degrees of freedom.
# Always use both df1 and df2 from summary(... )$fstatistic when calling pf().

# ----------------------------------------------------------------------------- -
# (iii, alt) White “fuller” form with sensible terms
# Include squares of continuous regressors and selected cross-products.
# Avoid squaring dummies: democA^2 == democA (perfect collinearity).
white_full <- lm(u2 ~ prtystrA + democA + lexpendA + lexpendB +
                   I(lexpendA^2) + I(lexpendB^2) +
                   I(prtystrA*lexpendA) + I(prtystrA*lexpendB) +
                   I(democA*lexpendA) + I(democA*lexpendB) +
                   I(lexpendA*lexpendB),
                 data = df)
fs_wfull <- summary(white_full)$fstatistic
p_wfull  <- pf(fs_wfull["value"], fs_wfull["numdf"], fs_wfull["dendf"], lower.tail = FALSE)
fs_wfull; p_wfull

# ----------------------------------------------------------------------------- -
# Interpretation reminders and pitfalls ---------------------------------------- -
# • Semi-elasticities: voteA is in levels; lexpendA/lexpendB are logs.
#   A 1% increase in spending by A changes voteA by 0.01*β_lexpendA percentage points.
# • Use robust SEs for the MAIN model’s inference (HC1 or similar). BP/White F-forms
#   are based on the auxiliary regressions with classical SEs by construction.
# • Do not include squares of binary indicators in White: they are collinear.
# • Do not read the residual-on-X R^2 as a diagnostic; it is zero by OLS design.
# • Label tests correctly:
#     - BP: uhat^2 ~ original X (intercept included).
#     - White special case: uhat^2 ~ fitted + fitted^2.
#     - “Fuller” White: uhat^2 ~ X, X^2 (continuous), and cross terms (sensible set).
# • Always report: F statistic, df1, df2, and p-value. State α and the conclusion.




# Modeller ---- 

#### 1: Core OLS and Linear Models ----

# --- Setup ---
library(wooldridge)     # data
library(dplyr)          # data manipulation
library(ggplot2)        # plots
data("wage2")
df <- wage2

###### (i) Simple Linear Regression (SLR) ----
# y = β0 + β1x + u
m_slr <- lm(wage ~ educ, data = df)
summary(m_slr)                     # effect of education on wage (ceteris paribus)
# β1 ≈ average change in wage for one more year of education

###### (ii) Multiple Linear Regression (MLR) ----
# y = β0 + β1x1 + ... + βkxk + u
m_mlr <- lm(wage ~ educ + exper + tenure, data = df)
summary(m_mlr)
# Adds controls to isolate ceteris paribus effects

###### (iii) Functional Forms ----
## Variant A: Level–Log
m_lvl_log <- lm(wage ~ log(exper), data = df)
## Variant B: Log–Level
m_log_lvl <- lm(log(wage) ~ exper, data = df)
## Variant C: Log–Log
m_log_log <- lm(log(wage) ~ log(exper), data = df)
summary(m_log_log)
# Log–Log: β1 = elasticity (%Δy for 1%Δx)

###### (iv) Regression Through the Origin ----
m_origin <- lm(wage ~ educ + exper - 1, data = df)
summary(m_origin)
# Intercept omitted; appropriate only if E(y|x=0)=0

###### (v) Partialling-out / Frisch–Waugh theorem ----
# β_educ from regression of wage on exper, tenure, educ
r_y  <- lm(wage ~ exper + tenure, data = df)$residuals
r_x1 <- lm(educ ~ exper + tenure, data = df)$residuals
m_fw <- lm(r_y ~ r_x1)
summary(m_fw)
# Identical slope estimate to β_educ in full MLR model

###### (vi) Dummy Variable Models ----
df <- mutate(df, female = ifelse(female == 1, 1, 0))  # ensure binary
m_dummy <- lm(wage ~ female + educ + exper, data = df)
summary(m_dummy)
# β_female measures mean wage difference (intercept shift)

###### (vii) Interaction Models (group-specific slopes) ----
m_interact <- lm(wage ~ female * educ + exper, data = df)
summary(m_interact)
# β_female: intercept difference | β_female:educ: slope difference across gender

###### (viii) Polynomial and Logarithmic Transformations ----
m_poly <- lm(wage ~ exper + I(exper^2), data = df)
summary(m_poly)
# Quadratic term captures diminishing marginal returns to experience

# --- Notes ---
# 1. All models satisfy MLR.1 (linear in parameters).
# 2. Transformations alter β-interpretation, not linearity.
# 3. Use log() for elasticity, I(x^2) for curvature, dummies/interactions for groups.

#### 2: Model specification and functional form tests ----

# --- Setup ---
library(wooldridge)    # data
library(dplyr)         # mutate
library(lmtest)        # resettest
library(car)           # linearHypothesis
data("wage2")

df <- wage2
df <- na.omit(df[, c("wage","educ","exper","tenure","female")])

###### (i) Baseline model (reference for tests) ----
m0 <- lm(log(wage) ~ educ + exper + tenure, data = df)
summary(m0)  # coefficients and adj. R^2

###### (ii) Ramsey RESET test (omitted nonlinearity / misspecification) ----
## Variant A: Fitted-power RESET (powers of ŷ)
reset_A <- resettest(m0, power = 2:3, type = "fitted")
print(reset_A)  # H0: no omitted nonlinear terms; reject => misspecification

####### Variant B: Regressor-power RESET (powers of X)
reset_B <- resettest(m0, power = 2:3, type = "regressor")
print(reset_B)  # Similar H0; uses polynomial expansions of regressors

###### (iii) Link test (ŷ and ŷ^2) ----
df$link_yhat <- fitted(m0)
m_link <- lm(log(wage) ~ link_yhat + I(link_yhat^2), data = df)
summary(m_link)
# H0: I(ŷ^2) not significant => functional form adequate; signif. I(ŷ^2) => misspecification

###### (iv) Chow test (structural stability across groups) ----
# Group: female (0 vs 1). Test equality of intercept and slopes.
mU <- lm(log(wage) ~ female*(educ + exper + tenure), data = df)  # unrestricted: intercept + slope shifts
# H0_all: same intercept and slopes across groups (no shifts)
H0_all <- c("female = 0",
            "female:educ = 0", "female:exper = 0", "female:tenure = 0")
chow_all <- linearHypothesis(mU, H0_all, test = "F")
print(chow_all)

# Optional: slope-equality only (allow intercept shift under H0)
H0_slopes <- c("female:educ = 0", "female:exper = 0", "female:tenure = 0")
chow_slopes <- linearHypothesis(mU, H0_slopes, test = "F")
print(chow_slopes)
# Reject => different slopes (group-specific returns)

###### (v) Adjusted R^2 comparison (parsimony-aware fit) ----
m_quad  <- lm(log(wage) ~ educ + exper + I(exper^2) + tenure, data = df)   # add curvature
m_inter <- lm(log(wage) ~ female*educ + exper + tenure, data = df)         # add interaction
adjR2 <- c(
  baseline = summary(m0)$adj.r.squared,
  with_quad = summary(m_quad)$adj.r.squared,
  with_interaction = summary(m_inter)$adj.r.squared
)
print(adjR2)
# Higher adjusted R^2 indicates better fit after penalizing complexity; use with theory + tests.

# --- Notes ---
# RESET: global check for omitted nonlinear terms; low p-value => add transforms/interactions.
# Link test: significant ŷ^2 flags wrong functional form.
# Chow: use interactions + F-test; homoskedasticity assumed for classic Chow.
# Model choice: combine tests, theory, and adjusted R^2; avoid overfitting.



#### 3: Classical Gauss–Markov Assumption Tests — short descriptions ----

# --- Setup ---
library(wooldridge)   # data
library(dplyr)        # data wrangling
library(car)          # vif, linearHypothesis
library(lmtest)       # resettest, dwtest, bgtest, bptest, gqtest
library(AER)          # ivreg + Wu–Hausman diagnostics
library(tseries)      # jarque.bera.test
library(ggplot2)      # residual visuals

###### Cross-section (CS) model for most tests -----
data("wage2")
df_cs <- na.omit(wage2[, c("wage","educ","exper","tenure","female","motheduc","fatheduc")])
m_cs  <- lm(log(wage) ~ educ + exper + tenure, data = df_cs)

###### Time-series (TS) model for independence tests----
data("phillips")
df_ts <- phillips
m_ts  <- lm(inf ~ unem, data = df_ts)  # simple Phillips curve

###### IV model for MLR.4 (endogeneity of educ)----
# Endog: educ | Instruments: motheduc, fatheduc (+ exogenous controls)
m_iv  <- ivreg(log(wage) ~ educ + exper + I(exper^2) | motheduc + fatheduc + exper + I(exper^2), data = df_cs)

###### (i) MLR.1 Linear in parameters: functional form checks ----
####### Variant A: Ramsey RESET (fitted powers)
reset_fit <- resettest(m_cs, power = 2:3, type = "fitted")
print(reset_fit)  # H0: correct functional form (no omitted nonlinearities)
####### Variant B: Chow via interactions (structural specification by group = female)
mU_chow <- lm(log(wage) ~ female*(educ + exper + tenure), data = df_cs)
H0_chow <- c("female = 0","female:educ = 0","female:exper = 0","female:tenure = 0")
print(car::linearHypothesis(mU_chow, H0_chow, test = "F"))  # H0: same intercept and slopes

###### (ii) MLR.2 Random sampling: independence (TS only) ----
## Variant A: Durbin–Watson
print(dwtest(m_ts))  # H0: no AR(1) serial correlation
## Variant B: Breusch–Godfrey (higher-order)
print(bgtest(m_ts, order = 1))      # AR(1)
print(bgtest(m_ts, order = 1:2))    # AR(1–2)

###### (iii) MLR.3 No perfect collinearity: multicollinearity diagnostics (CS) ----
## Variant A: VIF
print(vif(m_cs))     # rule-of-thumb: VIF > 10 high multicollinearity
## Variant B: Correlation matrix among regressors
print(cor(df_cs[, c("educ","exper","tenure")], use = "pairwise.complete.obs"))

###### (iv) MLR.4 Zero conditional mean: endogeneity tests (CS with IV) ----
## Variant A: Wu–Hausman / Durbin–Wu–Hausman from IV summary
print(summary(m_iv, diagnostics = TRUE))  # 'Wu-Hausman' tests endogeneity of educ
## Variant B: Manual 2SLS vs OLS comparison via Hausman-style regression (control function)
# 1st stage
fs <- lm(educ ~ motheduc + fatheduc + exper + I(exper^2), data = df_cs)
df_cs$uhat_educ <- resid(fs)
# 2nd stage with residuals
m_cf <- lm(log(wage) ~ educ + exper + I(exper^2) + uhat_educ, data = df_cs)
summary(m_cf)  # H0: coef on uhat_educ = 0 => no endogeneity

###### (v) MLR.5 Homoskedasticity: constant error variance (CS) ----
## Variant A: Breusch–Pagan (variance ~ regressors)
print(bptest(m_cs))  # H0: homoskedasticity
## Variant B: White test (general form) using squares and cross-terms via formula ~ .
print(bptest(m_cs, ~ educ + exper + tenure + I(educ^2) + I(exper^2) + I(tenure^2) +
               I(educ*exper) + I(educ*tenure) + I(exper*tenure), data = df_cs))
## Variant C: Goldfeld–Quandt (variance shift across ordered subsamples)
print(gqtest(m_cs, order.by = ~ exper, data = df_cs))  # H0: equal variances
## Variant D: Visual residual plots
df_cs$.resid <- resid(m_cs); df_cs$.fit <- fitted(m_cs)
print( ggplot(df_cs, aes(x = .fit, y = .resid)) +
         geom_point(alpha = 0.6) + geom_hline(yintercept = 0, linetype = 2) +
         labs(title = "Residuals vs Fitted", x = "Fitted", y = "Residuals") )

###### (vi) MLR.6 Normality: small-sample inference validity (CS) ----
## Variant A: Jarque–Bera
print(jarque.bera.test(resid(m_cs)))  # H0: normal residuals
## Variant B: Shapiro–Wilk (n <= ~5000 recommended)
print(shapiro.test(sample(resid(m_cs), size = min(5000, nrow(df_cs)))))

# --- Notes ---
# RESET/Chow address MLR.1 (specification). DW/BG address MLR.2 (independence, TS).
# VIF/cor address MLR.3. Wu–Hausman/CF address MLR.4 (endogeneity).
# BP/White/GQ/plots address MLR.5. JB/Shapiro address MLR.6.

#### 4: Hypothesis and significance testing ----

# --- Setup ---
library(wooldridge)   # data
library(dplyr)        # select, mutate
library(lmtest)       # coeftest, waldtest, lrtest, scoretest
library(sandwich)     # vcovHC (robust)
library(car)          # linearHypothesis

# --- Data ---
data("wage2")
df_ols <- na.omit(wage2[, c("wage","educ","exper","tenure","female")])
m_ols  <- lm(log(wage) ~ educ + exper + tenure + female, data = df_ols)   # baseline OLS

data("mroz")
df_glm <- na.omit(mroz[, c("inlf","educ","exper","kidslt6","kidsge6","age")])
mU_glm <- glm(inlf ~ educ + exper + I(exper^2) + kidslt6 + kidsge6 + age,
              data = df_glm, family = binomial(link = "logit"))           # unrestricted logit
mR_glm <- glm(inlf ~ educ + exper + I(exper^2) + age,                     # restricted (drop kids vars)
              data = df_glm, family = binomial(link = "logit"))

###### (i) t-test (single-coefficient) ----
# H0: β_educ = 0 in OLS
coeftest(m_ols)["educ", ]                                    # usual SE
coeftest(m_ols, vcov. = vcovHC(m_ols, type = "HC1"))["educ", ] # robust HC1
# Decision: small p-value => reject H0 (educ individually significant)

###### (ii) F-test (joint restrictions, linear) ----
# H0: exper = tenure = 0
car::linearHypothesis(m_ols, c("exper = 0", "tenure = 0"), test = "F")
# Interpretation: joint significance of a block (exclusion restriction)

###### (iii) Wald test (general linear/nonlinear restrictions) ----
# Linear Wald = F above; Nonlinear example: H0: exper^2 effect equals (tenure)^2 effect after log transform
# Implement via reparametrization or delta-method using 'car::linearHypothesis' on created terms:
df_ols$lexper  <- log(df_ols$exper + 1)     # guard zeros
df_ols$ltenure <- log(df_ols$tenure + 1)
m_wald <- lm(log(wage) ~ educ + lexper + ltenure + female, data = df_ols)
car::linearHypothesis(m_wald, "lexper = ltenure", test = "F")
# Wald: uses β-hat and Var(β-hat); valid asymptotically

###### (iv) Likelihood Ratio (LR) test (GLM/logit) ----
# H0: kidslt6 = kidsge6 = 0  (compare restricted vs unrestricted)
lrtest(mU_glm, mR_glm)   # 2*(ℓU - ℓR) ~ χ^2_df
# Interpretation: reject => children variables improve likelihood fit

###### (v) Lagrange Multiplier (LM / Score) test (GLM/logit) ----
# Score test uses restricted model only
scoretest(mR_glm, mU_glm)  # H0: restrictions valid; large stat => reject
# (Asymptotically equivalent to LR and Wald under correct specification)

###### (vi) Joint significance of regression (overall F-test, OLS) ----
# H0: all slopes = 0  (model explains no variation)
summary(m_ols)$fstatistic
# Use pf() to obtain p-value:
with(as.list(summary(m_ols)$fstatistic),
     1 - pf(value, numdf, dendf))  # small p => model has explanatory power

###### (vii) p-values and confidence intervals ----
# p-values already shown above; 95% CIs:
confint(m_ols, parm = c("educ","exper","tenure","female"))         # OLS, usual
confint(mU_glm, parm = c("educ","exper","I(exper^2)","kidslt6","kidsge6"), level = 0.95) # GLM
# Interpretation: CI excluding 0 => significant at corresponding α

# --- Notes ---
# t-test: single β.  F-test/Wald: multiple or nonlinear restrictions.
# LR/LM/Wald (GLM): asymptotically equivalent under correct specification.
# Overall F: tests joint null of all slopes = 0 in OLS.
# Report robust SE for heteroskedasticity; interpret CIs alongside p-values.



#### 5: Robust and Alternative Estimators ----

# --- Setup ---
library(wooldridge)    # data
library(dplyr)         # wrangling
library(lmtest)        # coeftest, wald tests
library(sandwich)      # vcovHC, NeweyWest
library(nlme)          # gls (FGLS with AR(1))

# --- Data ---
data("wage2")
df_cs <- na.omit(wage2[, c("wage","educ","exper","tenure","female")])
m0_cs <- lm(log(wage) ~ educ + exper + tenure + female, data = df_cs)   # baseline OLS

data("phillips")
df_ts <- phillips
m0_ts <- lm(inf ~ unem, data = df_ts)                                   # baseline TS OLS

###### (i) Heteroskedasticity-robust SEs (HC0–HC3) ----
hc_list <- c("HC0","HC1","HC2","HC3")
robust_tables <- lapply(hc_list, function(t) coeftest(m0_cs, vcov. = vcovHC(m0_cs, type = t)))
names(robust_tables) <- hc_list
robust_tables$HC3  # print preferred finite-sample correction
# Interpretation: coefficients unchanged; SEs adjust for heteroskedasticity.

###### (ii) Weighted Least Squares (WLS) ----
## Variant A: Known/assumed variance ∝ (exper + 1)  => weights = 1/(exper + 1)
w_known <- 1/(df_cs$exper + 1)
m_wls_A <- lm(log(wage) ~ educ + exper + tenure + female, data = df_cs, weights = w_known)
summary(m_wls_A)
# Efficiency gain if weight model is correct.

## Variant B: Feasible WLS via proxy weight 1/ŷ^2 (variance ∝ level^2)
yhat   <- fitted(m0_cs)
w_proxy <- 1/(yhat^2)
m_wls_B <- lm(log(wage) ~ educ + exper + tenure + female, data = df_cs, weights = w_proxy)
summary(m_wls_B)
# Heuristic when heteroskedasticity scales with outcome level.

###### (iii) Feasible GLS (FGLS) ----
## Variant A: Two-step heteroskedasticity FGLS (Harvey/White-style)
u2      <- resid(m0_cs)^2
aux     <- lm(log(u2) ~ educ + exper + tenure + female, data = df_cs)   # variance function
hhat    <- fitted(aux)
w_fgls  <- exp(-hhat)                                                    # w ≈ 1/σ^2(x)
m_fgls_H <- lm(log(wage) ~ educ + exper + tenure + female, data = df_cs, weights = w_fgls)
summary(m_fgls_H)
# Consistent and asymptotically efficient if variance model is correct.

## Variant B: Serial correlation FGLS (AR(1) errors) via GLS on TS
m_gls_ar1 <- gls(inf ~ unem, data = df_ts, correlation = corAR1(form = ~ 1 | 1))
summary(m_gls_ar1)
# Uses estimated ρ (AR1) to GLS-transform; efficient under AR(1).

###### (iv) Newey–West HAC (TS) ----
# Choose lag via rule-of-thumb (fixed small lag also common)
L <- 4L
coeftest(m0_ts, vcov. = NeweyWest(m0_ts, lag = L, prewhite = FALSE))
# Valid SEs under heteroskedasticity + autocorrelation.

# --- Notes ---
# HC0–HC3: inference robust to heteroskedasticity; HC3 often preferred in small samples.
# WLS: needs correct/credible weights; coefficients change due to reweighting.
# FGLS: estimate variance/correlation first, then GLS; efficient if error model holds.
# HAC/Newey–West: time series robust SEs; lag selection affects inference.

#### 6: Time Series extensions ----

# --- Setup ---
library(wooldridge)    # data
library(lmtest)        # dwtest, bgtest
library(tseries)       # adf.test, kpss.test, garch
library(dynlm)         # dynamic linear models with lags
library(orcutt)        # cochrane.orcutt
library(prais)         # prais_winsten

# --- Data ---
data("phillips")       # quarterly US inflation-unemployment
df_ts <- phillips
m_ols <- lm(inf ~ unem, data = df_ts)   # baseline TS regression

data("nyse")           # monthly excess returns (finance)
r <- na.omit(nyse$return)

###### (i) Autocorrelation / Serial correlation tests ----
## Durbin–Watson: AR(1) in residuals
dw_out <- dwtest(m_ols)                 # H0: no AR(1); values far from 2 => serial corr
print(dw_out)

## Breusch–Godfrey: AR(p) general, valid with lagged dep. var.
bg_ar1 <- bgtest(m_ols, order = 1)
bg_ar4 <- bgtest(m_ols, order = 4)
print(bg_ar1); print(bg_ar4)            # H0: no serial correlation up to p

###### (ii) Stationarity tests (levels) ----
adf_inf  <- adf.test(df_ts$inf, k = 4)  # H0: unit root; reject => stationary
adf_unem <- adf.test(df_ts$unem, k = 4)
print(adf_inf); print(adf_unem)

kpss_inf  <- kpss.test(df_ts$inf, null = "Level")    # H0: level-stationary
kpss_unem <- kpss.test(df_ts$unem, null = "Level")
print(kpss_inf); print(kpss_unem)       # Use ADF+KPSS jointly for diagnostics

###### (iii) Corrections for serial correlation ----
## Variant A: Cochrane–Orcutt (iterative AR(1) GLS)
co_out <- cochrane.orcutt(m_ols)        # estimates rho and refits via quasi-differencing
print(co_out)

## Variant B: Prais–Winsten (keeps first observation)
pw_out <- prais_winsten(inf ~ unem, data = df_ts)
summary(pw_out)                         # transformed OLS with AR(1) correction

## Variant C: AR(1)/AR(p) models with lagged dependent variable
m_ar1 <- dynlm(inf ~ L(inf, 1) + unem, data = df_ts)
summary(m_ar1)                          # explicit persistence via L(inf,1)

###### (iv) ARCH/GARCH volatility (financial returns) ----
# Mean equation: demeaned returns; Variance: GARCH(1,1)
r_c   <- r - mean(r, na.rm = TRUE)
adf_r <- adf.test(r_c)                  # check stationarity of returns (usually stationary)
print(adf_r)

set.seed(1)
g11 <- garch(r_c, order = c(1, 1))      # tseries::garch fits ARCH/GARCH by MLE
summary(g11)                            # conditional variance dynamics (α1, β1)

# --- Notes ---
# DW: AR(1) only; BG: AR(p) and valid with lagged y.
# ADF H0: unit root. KPSS H0: stationarity. Use both for balanced inference.
# Cochrane–Orcutt/Prais–Winsten: efficient if AR(1) errors; AR models capture persistence directly.
# GARCH(1,1): volatility clustering; α1+β1 close to 1 implies high persistence.

#### 7: Endogeneity and IV Estimation ----

# --- Setup ---
library(wooldridge)    # data
library(dplyr)         # wrangling
library(AER)           # ivreg, sargan, diagnostics
library(lmtest)        # coeftest
library(sandwich)      # vcovHC (robust SE)
library(car)           # linearHypothesis
library(systemfit)     # 3SLS

data("wage2")
df <- na.omit(wage2[, c("wage","educ","exper","tenure","female","motheduc","fatheduc")])
df$lwage <- log(df$wage)

###### (i) IV model (educ endogenous; instruments = motheduc, fatheduc) ----
iv1 <- ivreg(lwage ~ educ + exper + tenure + female |
               motheduc + fatheduc + exper + tenure + female,
             data = df)
summary(iv1)                                    # IV coefficients (classical SE)
summary(iv1, vcov = sandwich, df = Inf)         # IV with HC robust SE
summary(iv1, diagnostics = TRUE)                 # Wu–Hausman, weak IV F, Sargan (if overid)

###### (ii) Two-Stage Least Squares (2SLS) manual construct ----
# 1st stage: educ on instruments + exogenous controls
fs  <- lm(educ ~ motheduc + fatheduc + exper + tenure + female, data = df)
summary(fs)$r.squared                            # instrument relevance (R^2)
fs_F <- waldtest(fs, . ~ exper + tenure + female)[2, "F"]  # first-stage partial F (rule: > 10)
fs_F

# 2nd stage: replace educ with fitted values
df$educ_hat <- fitted(fs)
tsls2 <- lm(lwage ~ educ_hat + exper + tenure + female, data = df)
coeftest(tsls2, vcov. = vcovHC(tsls2, type = "HC1"))

# Durbin–Wu–Hausman via control function (residual inclusion)
df$u1 <- resid(fs)
cf2 <- lm(lwage ~ educ + exper + tenure + female + u1, data = df)
coeftest(cf2)["u1", ]                            # H0: coef(u1)=0 => no endogeneity

###### (iii) Three-Stage Least Squares (3SLS) system (educ jointly determined) ----
# Eq1: wage equation with endogenous educ
# Eq2: education equation determined by parental schooling + controls
eq1 <- lwage ~ educ + exper + tenure + female
eq2 <- educ  ~ motheduc + fatheduc + exper + tenure + female
sys <- list(wage_eq = eq1, educ_eq = eq2)
fit_3sls <- systemfit(sys, data = df, method = "3SLS")
summary(fit_3sls)                                 # system estimates, cross-eq efficiency

###### (iv) Hausman / Wu–Hausman / DWH tests (OLS vs IV) ----
ols1 <- lm(lwage ~ educ + exper + tenure + female, data = df)
# Wu–Hausman from AER diagnostics (requires iv1 to be over- or exactly identified)
summary(iv1, diagnostics = TRUE)$diagnostics

# Classic Hausman via control-function regression above:
car::linearHypothesis(cf2, "u1 = 0", test = "F")  # reject => educ endogenous

###### (v) First-stage F-test for instrument relevance ----
# Partial F already computed as fs_F; print explicitly
fs_F

###### (vi) Overidentification tests (Sargan / Hansen J) ----
# Sargan test (requires overidentification: #instruments > #endogenous regressors)
sargan(iv1)                                      # H0: instruments valid (uncorrelated with error)

# --- Notes ---
# IV/2SLS: consistency when instruments are relevant (Cov(Z, X_endog) ≠ 0) and exogenous (Cov(Z, u) = 0).
# 3SLS: gains efficiency in systems with correlated equation errors; needs valid instruments system-wide.
# Use robust SE (HC) for heteroskedasticity; weak instruments (F < 10) => unreliable IV.

#### 8: Policy Evaluation Models — short descriptions ----

# --- Setup ---
library(wooldridge)    # data
library(dplyr)         # wrangling
library(tidyr)         # pivot_longer
library(plm)           # FE/RE/FD and Hausman
library(lmtest)        # coeftest
library(sandwich)      # robust vcov

###### (i) Difference-in-Differences (DiD) ----
# Construct two-period DiD from JTRAIN: pre = re75, post = re78, treat = train
data("jtrain")
df_did <- jtrain |>
  select(re75, re78, train, age, educ, unem74, unem75) |>
  mutate(id = row_number()) |>
  pivot_longer(c(re75, re78), names_to = "period", values_to = "earn") |>
  mutate(post = ifelse(period == "re78", 1L, 0L),
         treat = train,
         did = treat * post)

# DiD w/ controls (RRA inside DiD)
m_did <- lm(earn ~ treat + post + did + age + educ + unem74 + unem75, data = df_did)
coeftest(m_did, vcov. = vcovHC(m_did, type = "HC1"))
# did = ATT under parallel trends; HC1 SEs allow heteroskedasticity

###### (ii) Regression Adjustment (RRA / URA) ----
# Post-only outcome with covariates; treat effect conditional on X
m_rra <- lm(re78 ~ train + age + educ + unem74 + unem75, data = jtrain)
coeftest(m_rra, vcov. = vcovHC(m_rra, type = "HC1"))
# Coef on train = partial effect under conditional independence

###### (iii) Fixed Effects (FE) estimator ----
# Wage panel (wagepan): remove time-invariant heterogeneity by within transform
data("wagepan")
df_pan <- wagepan |> select(nr, year, lwage, educ, exper, expersq, married, union)
pan    <- pdata.frame(df_pan, index = c("nr","year"))

m_fe <- plm(lwage ~ exper + expersq + married + union, data = pan, model = "within")
coeftest(m_fe, vcov. = vcovHC(m_fe, type = "HC1", cluster = "group"))
# FE identifies from within-person changes; educ drops if time-invariant

###### (iv) Random Effects (RE) estimator ----
m_re <- plm(lwage ~ educ + exper + expersq + married + union, data = pan, model = "random")
coeftest(m_re, vcov. = vcovHC(m_re, type = "HC1", cluster = "group"))
# RE efficient if Corr(α_i, X_it) = 0

###### (v) First Differencing (FD) ----
m_fd <- plm(lwage ~ exper + expersq + married + union, data = pan, model = "fd")
coeftest(m_fd, vcov. = vcovHC(m_fd, type = "HC1", cluster = "group"))
# FD removes α_i by Δ; robust when serial correlation undermines FE assumptions

###### (vi) Hausman FE vs RE test ----
haus <- phtest(m_fe, m_re)   # H0: RE consistent (no correlation with effects)
print(haus)
# Reject => prefer FE

# --- Notes ---
# DiD: ATT requires parallel trends; add group/time-specific controls as needed.
# RRA: unbiased if selection on observables holds; combine with DiD for DiD-RRA.
# FE: removes time-invariant confounding; RE needs exogeneity of effects.
# FD: targets short panels; can reduce persistence bias but increases noise.
# Use cluster-robust SEs by unit for panel estimators.

#### 9: Binary and Limited Dependent Variable Models ----

# --- Setup ---
library(wooldridge)      # data
library(dplyr)           # wrangling
library(lmtest)          # coeftest
library(sandwich)        # robust vcov
library(AER)             # tobit
library(sampleSelection) # Heckman two-step
library(MASS)            # glm.nb (Negative Binomial)

###### (i) Linear Probability Model (LPM) ----
data("mroz")
df_bin <- na.omit(mroz[, c("inlf","educ","exper","I(exper^2)","kidslt6","kidsge6","age")])
m_lpm  <- lm(inlf ~ educ + exper + I(exper^2) + kidslt6 + kidsge6 + age, data = df_bin)
coeftest(m_lpm, vcov. = vcovHC(m_lpm, type = "HC1"))
# LPM: OLS on binary Y; interpretable slopes; heteroskedasticity-robust SEs needed; ŷ may be outside [0,1].

###### (ii) Logit and Probit (MLE) ----
m_logit  <- glm(inlf ~ educ + exper + I(exper^2) + kidslt6 + kidsge6 + age,
                data = df_bin, family = binomial(link = "logit"))
m_probit <- glm(inlf ~ educ + exper + I(exper^2) + kidslt6 + kidsge6 + age,
                data = df_bin, family = binomial(link = "probit"))
summary(m_logit)
summary(m_probit)
# Logit/Probit: Pr(Y=1|X) in (0,1). Coefs are in log-odds (logit) or probit index; compute marginal effects if needed.

###### (iii) Tobit (censored regression) ----
# Use hours worked (0 = not working) as left-censored at 0
df_tob <- na.omit(mroz[, c("hours","educ","exper","I(exper^2)","kidslt6","kidsge6","age")])
m_tobit <- tobit(hours ~ educ + exper + I(exper^2) + kidslt6 + kidsge6 + age,
                 left = 0, right = Inf, data = df_tob)
summary(m_tobit)
# Tobit: joint model for Pr(censoring) and E[Y|Y>0,X]; coefficients are on latent index.

###### (iv) Truncated regression (sample excludes Y ≤ c) ----
# Truncate to positive hours (sample selection by design), declare truncation at 0
df_trunc <- subset(df_tob, hours > 0)
m_trunc  <- truncreg::truncreg(hours ~ educ + exper + I(exper^2) + kidslt6 + kidsge6 + age,
                               point = 0, direction = "left", data = df_trunc)
summary(m_trunc)
# Truncated regression corrects for truncation bias (observations below cutoff not in sample).

###### (v) Heckman selection (two-step) ----
# Outcome observed only when working; selection eq: inlf; outcome eq: log wage
df_sel <- mroz %>%
  mutate(lwage = log(wage)) %>%
  select(lwage, inlf, educ, exper, I(exper^2), kidslt6, kidsge6, age, hushrs, mtr, motheduc, fatheduc)
# Selection: inlf ~ Z; Outcome: lwage ~ X (observed if inlf==1)
heck2s <- sampleSelection::heckit(selection = inlf ~ educ + kidslt6 + kidsge6 + age + hushrs + mtr,
                                  outcome   = lwage ~ educ + exper + I(exper^2),
                                  data = df_sel, method = "2step")
summary(heck2s)
# λ (IMR) significant => selection present; controls second-stage bias.

###### (vi) Count models: Poisson and Negative Binomial ----
data("crime1")  # narr86 = number of arrests in 1986
df_cnt <- na.omit(crime1[, c("narr86","pcnv","avgsen","tottime","ptime86","qemp86","black","hispan","educ","age")])
m_pois <- glm(narr86 ~ pcnv + avgsen + tottime + ptime86 + qemp86 + black + hispan + educ + age,
              data = df_cnt, family = poisson(link = "log"))
summary(m_pois)
# Check overdispersion: if Var(Y) >> E(Y), prefer NegBin
m_negbin <- MASS::glm.nb(narr86 ~ pcnv + avgsen + tottime + ptime86 + qemp86 + black + hispan + educ + age,
                         data = df_cnt)
summary(m_negbin)
# Compare via AIC or LR test (theta>0 indicates overdispersion captured).

# --- Notes ---
# LPM: simple and interpretable; use HC SEs; predictions can leave [0,1].
# Logit/Probit: bounded probabilities; use marginal effects for economic size.
# Tobit: censoring at known limit; coefficients on latent variable; report unconditional/conditional effects if needed.
# Truncated: corrects when data below/above cutoff are unobserved.
# Heckman: corrects selection via IMR; identification needs exclusion restriction in selection eq.
# Poisson vs NegBin: NegBin handles overdispersion; use LR/AIC and dispersion checks to choose.


#### 10: Maximum Likelihood and GMM frameworks ----

# --- Setup ---
library(wooldridge)   # data
library(dplyr)        # wrangling
library(lmtest)       # lrtest, waldtest, scoretest
library(sandwich)     # robust vcov
library(gmm)          # GMM estimator
library(AER)          # ivreg (reference)

###### (i) Maximum Likelihood Estimation (MLE): Logit example ----
data("mroz")
df_mle <- na.omit(mroz[, c("inlf","educ","exper","kidslt6","kidsge6","age")])

# Unrestricted logit (U) and restricted (R: drop kids variables)
mU <- glm(inlf ~ educ + exper + I(exper^2) + kidslt6 + kidsge6 + age,
          data = df_mle, family = binomial("logit"))
mR <- glm(inlf ~ educ + exper + I(exper^2) + age,
          data = df_mle, family = binomial("logit"))
summary(mU)  # MLE estimates; asymptotically efficient if correctly specified

###### (ii) Log-likelihood ratio test (LR) ----
# H0: restrictions in mR valid (kidslt6 = kidsge6 = 0)
lrtest(mU, mR)  # 2*(ℓU - ℓR) ~ χ^2_df; reject => restrictions invalid

###### (iii) Wald and LM (Score) tests; large-sample equivalence with LR ----
# Wald (uses U): H0: kidslt6 = kidsge6 = 0
waldtest(mU, . ~ . - kidslt6 - kidsge6, test = "Chisq")
# LM/Score (uses R only, compares to U's expansion)
scoretest(mR, mU)  # Asymptotically equivalent to LR and Wald under correct spec

###### (iv) GMM: Linear IV-GMM with wage equation (moment E[Z * (y - Xβ)] = 0) ----
data("wage2")
df_gmm <- na.omit(wage2[, c("wage","educ","exper","tenure","female","motheduc","fatheduc")])
df_gmm$lwage <- log(df_gmm$wage)

# Define variables
y  <- df_gmm$lwage
X  <- cbind(1, df_gmm$educ, df_gmm$exper, df_gmm$tenure, df_gmm$female)              # [1, educ, exper, tenure, female]
Z  <- cbind(1, df_gmm$motheduc, df_gmm$fatheduc, df_gmm$exper, df_gmm$tenure, df_gmm$female) # instruments

# Moment function g(θ) = Z' * (y - Xβ)
gfun <- function(theta, x) {
  y  <- x$y
  X  <- x$X
  Z  <- x$Z
  u  <- as.numeric(y - X %*% theta)
  g  <- Z * u  # moments per observation
  return(g)
}

# Starting values via OLS
theta0 <- coef(lm(lwage ~ educ + exper + tenure + female, data = df_gmm))

# Two-step efficient GMM (heteroskedasticity-robust)
fit_gmm <- gmm(gfun, x = list(y = y, X = X, Z = Z), t0 = theta0, type = "twoStep")
summary(fit_gmm)  # β_GMM, robust SEs, Hansen J test for overidentifying restrictions

# Compare with IV (2SLS) for reference
fit_iv <- ivreg(lwage ~ educ + exper + tenure + female | motheduc + fatheduc + exper + tenure + female,
                data = df_gmm)
summary(fit_iv, vcov = sandwich, df = Inf)

# --- Notes ---
# MLE: maximize ℓ(θ); efficient under correct distribution. LR compares ℓ_U vs ℓ_R.
# Wald/LM/LR: asymptotically equivalent; differ in which model they use (U vs R).
# GMM: solve sample moments; two-step uses optimal weighting; Hansen J tests instrument validity.

#### 11: Panel Data Diagnostics — short descriptions ----

# --- Setup ---
library(wooldridge)   # data
library(plm)          # panel models and tests: phtest, plmtest, pwartest, pcdtest
library(lmtest)       # coeftest
library(sandwich)     # vcovHC (cluster-robust)

# --- Data and baseline models (wagepan) ---
data("wagepan")
df <- wagepan[, c("nr","year","lwage","educ","exper","expersq","married","union")]
pan <- pdata.frame(df, index = c("nr","year"))

m_pool <- plm(lwage ~ educ + exper + expersq + married + union, data = pan, model = "pooling")
m_fe   <- plm(lwage ~ educ + exper + expersq + married + union, data = pan, model = "within")
m_re   <- plm(lwage ~ educ + exper + expersq + married + union, data = pan, model = "random")

###### (i) Hausman test (FE vs RE) ----
haus <- phtest(m_fe, m_re)   # H0: RE consistent (no Corr(alpha_i, X_it))
print(haus)                  # Reject => prefer FE

###### (ii) Breusch–Pagan LM test for random effects (RE vs pooled OLS) ----
bp_lm <- plmtest(m_pool, type = "bp")  # H0: Var(alpha_i) = 0 (no RE)
print(bp_lm)                           # Reject => RE preferred over pooled OLS

###### (iii) Serial correlation in panels (Wooldridge test) ----
# Wooldridge AR(1) test for FE residuals
ar_fe <- pwartest(m_fe)                # H0: no AR(1) in idiosyncratic errors
print(ar_fe)

###### (iv) Cross-sectional dependence (Pesaran CD test) ----
cd_fe <- pcdtest(m_fe, test = "cd")    # H0: no cross-sectional dependence
print(cd_fe)

###### (v) Optional: cluster-robust SEs for FE/RE (by individual) ----
coeftest(m_fe, vcov. = vcovHC(m_fe, type = "HC1", cluster = "group"))
coeftest(m_re, vcov. = vcovHC(m_re, type = "HC1", cluster = "group"))

# --- Notes ---
# Hausman: significant => FE consistent, RE inconsistent.
# BP LM: significant => RE better than pooled OLS.
# Wooldridge (pwartest): detects AR(1); use cluster-robust or model AR structure if present.
# Pesaran CD: significant => cross-sectional dependence; consider Driscoll–Kraay SEs or factor models.


#### 12: Advanced and Nonlinear Models ----

# --- Setup ---
library(wooldridge)    # data
library(dynlm)         # dynamic regressions with lags
library(systemfit)     # SEM: 2SLS/3SLS and recursive systems
library(urca)          # Johansen cointegration
library(tseries)       # ADF test
library(lmtest)        # coeftest
library(sandwich)      # robust vcov
library(zoo)           # time-series indexing

###### (i) Simultaneous Equation Models (SEM) ----
## Example system: wage equation with endogenous education + education equation (instruments: parental educ)
data("wage2")
df_sem <- na.omit(wage2[, c("wage","educ","exper","tenure","female","motheduc","fatheduc")])
df_sem$lwage <- log(df_sem$wage)

eq1 <- lwage ~ educ + exper + tenure + female
eq2 <- educ  ~ motheduc + fatheduc + exper + tenure + female
sys <- list(wage_eq = eq1, educ_eq = eq2)

fit_2sls <- systemfit(sys, data = df_sem, method = "2SLS")
summary(fit_2sls)                                # equation-by-equation IV (consistent)

fit_3sls <- systemfit(sys, data = df_sem, method = "3SLS")
summary(fit_3sls)                                # gains efficiency if cross-eq errors correlated

###### (ii) Recursive systems (triangular, ordered) ----
# If system is recursive and disturbances are uncorrelated across equations, OLS is consistent eq-by-eq.
# Order: educ determined by Z, then lwage depends on educ and X.
ols_educ <- lm(educ ~ motheduc + fatheduc + exper + tenure + female, data = df_sem)
ols_wage <- lm(lwage ~ educ + exper + tenure + female, data = df_sem)
summary(ols_educ); summary(ols_wage)             # valid if errors are orthogonal across equations

###### (iii) Distributed Lag and Koyck models ----
data("phillips")                                 # quarterly US data
ph <- zooreg(phillips, start = c(1948,1), frequency = 4)

## Finite distributed lag: inf_t on current and lagged unemployment
m_dlag <- dynlm(inf ~ unem + L(unem, 1) + L(unem, 2) + L(unem, 3), data = ph)
coeftest(m_dlag, vcov. = sandwich)

## Koyck approximation: include lagged dependent variable
m_koyck <- dynlm(inf ~ unem + L(inf, 1), data = ph)
coeftest(m_koyck, vcov. = sandwich)
# Koyck implies geometrically declining weights on past x via AR(1) in y.

###### (iv) Error Correction Model (ECM) ----
# Long-run relation (Fisher effect): i3 vs inf; then ECM in first differences
data("intdef")
id <- zooreg(intdef, start = c(1948,1), frequency = 4)

# Step 1: test unit roots (informal)
adf_i3  <- adf.test(id$i3, k = 4)    # H0: unit root
adf_inf <- adf.test(id$inf, k = 4)

# Step 2: Engle–Granger cointegration test via residual ADF
coint_lr <- dynlm(i3 ~ inf, data = id)
u_hat    <- resid(coint_lr)
adf_u    <- adf.test(u_hat, k = 4)   # reject => cointegration

# Step 3: ECM: Δi3_t = α(u_{t-1}) + γΔinf_t + controls
ecm <- dynlm(diff(i3) ~ L(u_hat, 1) + diff(inf), data = id)
coeftest(ecm, vcov. = sandwich)
# L(u_hat,1) < 0 implies adjustment back to long-run equilibrium.

###### (v) Cointegration tests (Engle–Granger, Johansen) ----
## Engle–Granger already shown via residual ADF (adf_u).
## Johansen test for multiple I(1) variables (here: i3 and inf)
Y <- cbind(id$i3, id$inf)
cj <- ca.jo(na.omit(Y), type = "trace", ecdet = "const", K = 2)  # small lag for demo
summary(cj)                                       # number of cointegrating relations

###### (vi) Forecast evaluation tests (Theil’s U, MSE comparison) ----
# Compare AR(1) vs naive random-walk forecast for inflation; hold-out last 8 obs
y <- na.omit(id$inf)
T <- length(y); h <- 8L
train <- y[1:(T - h)]; test <- y[(T - h + 1):T]

# Model A: AR(1)
ar1 <- dynlm(train ~ L(train,1))
phi <- coef(ar1)["L(train, 1)"]; mu <- coef(ar1)["(Intercept)"]
f_ar1 <- numeric(h); y_last <- tail(train, 1)
for (t in 1:h) { f_ar1[t] <- mu + phi * ifelse(t == 1, y_last, f_ar1[t-1]) }

# Model B: Naive (random walk) => forecast = last observed
f_naive <- rep(tail(train, 1), h)

# Errors and metrics
e_ar1   <- test - f_ar1
e_naive <- test - f_naive
MSE_ar1 <- mean(e_ar1^2)
MSE_nv  <- mean(e_naive^2)

# Theil’s U (relative RMSE vs naive)
U <- sqrt(MSE_ar1) / sqrt(MSE_nv)
print(list(MSE_AR1 = MSE_ar1, MSE_Naive = MSE_nv, Theils_U = U))
# U < 1 => AR(1) improves on naive; U ≈ 1 => similar performance.

# --- Notes ---
# SEM: use 2SLS/3SLS when simultaneity creates endogeneity; 3SLS exploits cross-eq error correlation.
# Recursive: triangular structure permits consistent OLS if equation errors are orthogonal and ordering is correct.
# DL/Koyck: finite vs geometric lag structures; Koyck uses L(y,1) to summarize infinite lags.
# ECM: valid when variables are cointegrated; error-correction term drives adjustment to long-run.
# Cointegration: Engle–Granger for two series, Johansen for systems (rank via trace/eigen tests).
# Forecast evaluation: compare out-of-sample accuracy; Theil’s U < 1 indicates improvement over naive.




# ANDET -----
# ====================================================== - 
## Multiple regression  ----

rm(list=ls())

# load data
library(haven)
wage1 <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Lectures/A1 OLS/WAGE1.DTA")

# estimate model:
reg1 <- lm(lwage ~ educ + exper + tenure, data=wage1)
summary(reg1)

# compute VIF (Variance Inflator Factor) for all regressors

vif(reg1)

# ====================================================== - 
## LM test ----
# Here we test for two exclusion restrictions

rm(list=ls())

# load data
library(haven)
wage1 <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2024/Lectures/A1 OLS/wage1.dta")

# estimate model:
reg1 <- lm(lwage~ educ + exper, data=wage1)
summary(reg1)


# auxiliary regression:

reg2 <- lm(resid(reg1)~ educ + exper + tenure + married, data=wage1)
summary(reg2)


# calculating the LM test statistic
LM <- summary(reg2)$r.squared*526
LM

# p value
1-pchisq(LM,2)
#H_0 clearly rejected


# ====================================================== - 
## SARGAN ----
rm(list=ls(all=TRUE))
library(foreign)
library(AER)

Data<-read.dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Lectures/B1 Endogeneity/MROZ.DTA")
Data<-Data[1:428,]

#H_o : E[z'u]=0
l_2sls<-ivreg(lwage~educ+exper+expersq|.-educ+motheduc+fatheduc+huseduc,data=Data)

#Sargan test
u_hat<-l_2sls$residuals

aux<-lm(u_hat~motheduc+fatheduc+huseduc+exper+expersq,data=Data)

LM_statistic<-summary(aux)$r.squared*dim(Data)[1]
p_value<-1-pchisq(LM_statistic,2)
#We do not reject H_0


# ====================================================== - 
## SHRINKAGE ------- 
rm(list=ls(all=TRUE))

# Example code for common shrinkage estimators: Lasso, Ridge, Group Lasso
# This is optional material for the course BA-BMECV1031U Econometrics

# Uncomment on first run
# install.packages("glmnet", repos = "http://cran.us.r-project.org")
# For more information see https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html

# install.packages("gglasso", repos = "http://cran.us.r-project.org")


library(glmnet)
library(repmis)
library(gglasso)

set.seed(123)

# Generate example data 
u <- rnorm(100)
beta <- runif(20, -1, 2) # random coefficients 
beta <- pmax(beta, 0) # set negative to zero 
x = matrix(rnorm(100 *20), 100, 20)
y = x%*%beta+u # standard linear model 

# Linear regression 
# Regression with a weighted penalty term
### Lasso ###
# alpha=1 gives lasso
fit_lasso <- glmnet(x, y, alpha=1)
print(fit_lasso)
plot(fit_lasso, xvar = "lambda", label = TRUE)
plot(fit_lasso, xvar = "dev", label = TRUE)
# Model selection
cv.lasso <- cv.glmnet(x,y)
plot(cv.lasso)
# best model - Coefficient extraction 
best_lambda_lasso <- cv.lasso$lambda.min
best_model_lasso <- glmnet(x, y, alpha = 1, lambda = best_lambda_lasso)
coef(best_model_lasso)

### Ridge ###
# alpha=0 gives ridge 
fit_ridge <- glmnet(x, y, alpha=0)
print(fit_ridge)
plot(fit_ridge, xvar = "lambda", label = TRUE)
plot(fit_ridge, xvar = "dev", label = TRUE)
# Model selection
cv.ridge <- cv.glmnet(x,y)
plot(cv.ridge)
# best model - Coefficient extraction 
best_lambda_ridge <- cv.ridge$lambda.min
best_model_ridge <- glmnet(x, y, alpha = 0, lambda = best_lambda_ridge)
coef(best_model_ridge)

### Group Lasso ###
# Select or unselect groups of covariates/regressors
# in the following example genes are grouped into 5 groups and the Group Lasso 
# is used find out whether a diseases is linked to certain groups of genes.

# The example Data consist of colon tissue samples from 62 patients. y indicates tumor (1)
# or normal (-1). x is a 62x100 matrix (expanded from a 62x20 matrix) giving the 
# expression levels of 20 genes. 5 consecutive columns corresponds to a grouped gene. 
# for more information see https://cran.r-project.org/web/packages/gglasso/gglasso.pdf

# load data 
data(colon)

# define 20 groups 
group <- rep(1:20,each=5)

# group lasso 
m <- gglasso(x=colon$x,y=colon$y,group=group,loss="ls")

# plot results 
plot(m) # plots the coefficients against the log-lambda sequence
plot(m,group=TRUE) # plots group norm against the log-lambda sequence
plot(m,log.l=FALSE) # plots against the lambda sequence

# coefficients at specified lambdas
coef(m, s=c(0.02, 0.03, 0.04))

# There are not yet many examples of the  Group Lasso in Economics and Business.
# A relevant one is the study "Variable selection with group structure: exiting
# employment at retirement age - A competing risks quantile regression analysis"
# By Shuolin Shi and Ralf A.Wilke, 2020, in Empirical Economics.


# ====================================================== - 
## AR(1) t-test ----
# test for the presence of AR(1)

rm(list=ls())

# load data
library(haven)
phillips <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Lectures/A2 OLS topics/phillips.dta")

# define yearly time series beginning in 1948
tsdata <- ts(phillips, start=1948)

library(dynlm)

# estimation of static Phillips curve:
reg1 <- dynlm(inf~ unem, data=tsdata, end=1996)
summary(reg1)

# compute OLS residuals:
residuals <- resid(reg1)

# t-test without constant in AR(1) model:
library(lmtest)
coeftest(dynlm(residuals~ L(residuals) + 0))

# t-test with constant in AR(1) model: 
coeftest(dynlm(residuals~ L(residuals)))

# heteroskedasticity robust version of t-test without constant in AR(1) model:
library(sandwich)
coeftest(dynlm(residuals~ L(residuals) + 0), vcovHAC)


# all tests suggest at 1% that there is evidence for positive serial correlation in errors

# ====================================================== - 
## Arg test -----


rm(list=ls(all=TRUE))

# for read.dta
library(foreign)

# for mutate 
library(dplyr)

# for bgtest
library(lmtest)

# setwd(".")
setwd("H:/Teaching/CBS/BA-BMECV1031U/2025/Lectures/A2 OLS topics")

phillips <- read.dta("phillips.dta")

### F-/LMTest for presence of AR(3)

phillips.limited <- subset(phillips, year<=1996)

reg <- lm(inf ~ unem, data=phillips.limited)
# obtain OLS residuals 
phillips.limited$res <- reg$residuals 

# F-test without constant in AR(3)model 
# generate lags 
phillips.limited <- phillips.limited %>%
  mutate(res_1 = dplyr::lag(res, n=1, default = NA)) %>%
  mutate(res_2 = dplyr::lag(res, n=2, default = NA)) %>%
  mutate(res_3 = dplyr::lag(res, n=3, default = NA)) 

ar3 <- lm(res ~ res_1 + res_2 + res_3 -1, data=phillips.limited)
summary(ar3)
# observe the reduced number of observations.
# F statistic has (3,43) d.f. and is >20.
# strong evidence of serial correlation of order 3.

# LM version of the test (special case of Breusch-Godfrey):
LM <- summary(ar3)$r.squared*(summary(ar3)$df[1]+summary(ar3)$df[2])
pvalue <- pchisq(LM, 3, lower.tail=FALSE)

message("Breusch-Godfrey statistic: LM = ", LM,", p-value = ", pvalue)

# Breusch Godfrey, 3rd order serial correlation 
bgtest(inf ~ unem, order = 3, type = "Chisq", data = phillips.limited)

# ====================================================== - 
## Breusch Pagan Test for Heteroscedasticity ----

rm(list=ls())

# load data
library(haven)
hprice1 <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Lectures/A2 OLS topics/hprice1.dta")

# estimate model:
reg1 <- lm(price~ lotsize + sqrft + bdrms, data=hprice1)
summary(reg1)


# automatic BP test:

library(lmtest)
bptest(reg1)


# manual regression of squared residuals:

reg2 <- lm(resid(reg1)^2~ lotsize + sqrft + bdrms, data=hprice1)
summary(reg2)
rsquared <- summary(reg2)$r.squared

# calculating of LM test statistic
LM = rsquared*length(reg2$residuals) 
LM

# p value
1-pchisq(LM,3)



# ====================================================== - 
## FD -----

# FD model

rm(list=ls())

# load data
library(haven)
intdef <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Lectures/A2 OLS topics/intdef.dta")

# define yearly time series beginning in 1948
tsdata <- ts(intdef, start=1948)
library(dynlm)

reg1 <- dynlm(i3~ inf + def, data=tsdata)
summary(reg1)

# compute OLS residuals:
residuals1 <- resid(reg1)

# t-test without constant in AR(1)model
library(lmtest)
coeftest(dynlm(residuals1~ L(residuals1) + 0))

# evidence for positive serial correlation in errors


# first differenced model:
reg2 <- dynlm(d(i3)~ d(inf) + d(def) + 0, data=tsdata)
summary(reg2)

# compute OLS residuals in differenced model
residuals2 <- resid(reg2)

# t-test without constant in AR(1)model
coeftest(dynlm(residuals2~ L(residuals2) + 0))


################################################# -
# Remark
################################################# -
# comparing the results from the two regressions:
# the coefficients in the level model appear to have a plausible sign and they are significant
# the coefficients in the differenced model are insignificant.
# the coefficients vary statistically across the models:
# evidence for violation of the Gaus-Maarkov assumptions in the level model (exogeneity)



# ====================================================== - 
## FGLS ----
# FGLS

rm(list=ls())

# load data
library(haven)
smoke <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Lectures/A2 OLS topics/smoke.dta")

# OLS regression:

reg1 <- lm(cigs~ lincome + lcigpric + educ + age + agesq + restaurn, data=smoke)
summary(reg1)

yhat <- sum(reg1$fitted.values <=0)
yhat
# percentage of observations with fitted value <0
yhat/807


# Breusch- Pagan test for heteroscedasticity: 

reg1 <- lm(cigs~ lincome + lcigpric + educ + age + agesq + restaurn, data=smoke)
summary(reg1)

# automatic BP test:
library(lmtest)
bptest(reg1)

# manual regression of squared residuals:
reg2 <- lm(resid(reg1)^2~ lincome + lcigpric + educ + age + agesq + restaurn, data=smoke)
summary(reg2)
rsquared <- summary(lm(resid(reg1)^2~ lincome + lcigpric + educ + age + agesq + restaurn, data=smoke))$r.squared
# calculating of LM test statistic
LM = rsquared*807
LM
# p value
1-pchisq(LM,6)


# estimate by FGLS

# FGLS: estimation of the variance function:
logu2 <- log(resid(reg1)^2)
varreg <- lm(logu2~ lincome + lcigpric + educ + age + agesq + restaurn, data=smoke)

# FGLS: run WLS:
w <- 1/exp(fitted(varreg))

WLS <- lm(cigs~ lincome + lcigpric + educ + age + agesq + restaurn, weight=w, data=smoke)
summary(WLS)


# ====================================================== - 
## Heteroskedasticity robust  -----

# Heteroscedasticity robust

rm(list=ls())

# load data
library(haven)
wage1 <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2025/Lectures/A2 OLS topics/wage1.dta")

# generating missing variables:
wage1$marrmale <- wage1$married*(1-wage1$female)
wage1$marrfem <- wage1$married*wage1$female
wage1$singfem <- (1-wage1$married)*wage1$female

# regression with "normal" standard errors:
reg1 <- lm(lwage~ marrmale + marrfem + singfem + educ + exper + expersq + tenure + tenursq, data=wage1)
summary(reg1)

# regression with heteroscedasticity robust standard errors:
library(lmtest)
library(car)
#refined White robust SE:
coeftest(reg1, vcov=hccm)
#or classical White robust SE:
coeftest(reg1, vcov=hccm(reg1, type="hc0"))



# ====================================================== - 
## Newey West Standard Errors ------

rm(list=ls())

# load data
library(haven)
phillips <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Lectures/A2 OLS topics/phillips.dta")

library(dynlm)
tsdata <- ts(phillips, start=1948)

# OLS estimation:
reg1 <- dynlm(inf~ unem, data=tsdata, end=1996)
summary(reg1)


# OLS with robust SE (Newey-West correction):
#one needs to specify maximum order of autocorrelation 
#resulting SE are robust up to the chosen order
#They are also robust with respect to heteroskedasticity
library(sandwich)
coeftest(reg1, vcov=NeweyWest(reg1, lag=3))

# For comparison: OLS with heteroskedasticity robust SE:
coeftest(reg1, vcov=hccm)


# ====================================================== - 
## DID estimator ----
# for read.dta
library(foreign)

# run from the same folder as data
# setwd(".")

kielmc <- read.dta("kielmc.dta")

# did estimator (by hand)
gamma81 <- coefficients(lm(rprice~nearinc,data= kielmc, subset= y81==1))[1]
gamma78 <- coefficients(lm(rprice~nearinc,data= kielmc, subset= y81==0))[1]

# did_estimator
(gamma78 - gamma81)

# estimate the DID estimator (directly)
summary(lm(rprice~y81+nearinc+y81*nearinc,data= kielmc))
# with age and age^2
summary(lm(rprice~y81+nearinc+y81*nearinc+age+agesq,data= kielmc))



# ====================================================== - 
## TAU ATE -----
rm(list=ls(all=TRUE))

# for read_dta
library(haven)

# foreign::read.dta() promoted the follow error:
# Error in read.dta("jtrain98.dta") : not a Stata version 5-12 .dta file
# For .dta files outside this range one can use haven::read_data 

# for mutate 
library(dplyr)

# setwd(".")
setwd("H:/Teaching/CBS/BA-BMECV1031U/2020/Lectures/A3 Policy Analysis")

jtrain <- read_dta("jtrain98.dta")

# generate interaction terms 
jtrain <- jtrain %>%
  mutate(train.demeaned_earn96 = train*(earn96-mean(earn96))) %>%
  mutate(train.demeaned_educ = train*(educ-mean(educ))) %>%
  mutate(train.demeaned_age = train*(age-mean(age))) %>%
  mutate(train.demeaned_married = train*(married-mean(married))) 

# ura
ura <- lm(earn98 ~ train+earn96+educ+age+married
          +train.demeaned_earn96+train.demeaned_educ
          +train.demeaned_age+train.demeaned_married, data=jtrain)
tau_ura <- summary(ura)$coefficients[2]
tau_ura_se <- summary(ura)$coefficients[2,2]

# Difference-in-means 
diff_means <- lm(earn98 ~ train, data=jtrain)
tau_diff <- summary(diff_means)$coefficients[2]
tau_diff_se <- summary(diff_means)$coefficients[2,2]

# rra 
rra <- lm(earn98 ~ train+earn96+educ+age+married, data=jtrain)
tau_rra <- summary(rra)$coefficients[2]
tau_rra_se <- summary(rra)$coefficients[2,2]

# Summarise results 
est <- c('tau_ura', 'tau_diff', 'tau_rra')
coeff <- round(c(tau_ura, tau_diff, tau_rra),2)
se <- round(c(tau_ura_se, tau_diff_se, tau_rra_se),2)
cbind(est,coeff,se)

# ====================================================== - 
## Endogeneity test ----

rm(list=ls(all=TRUE))
library(foreign)
library(AER)

Data<-read.dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Lectures/B1 Endogeneity/MROZ.DTA")

Data_omit<-Data[1:428,]

l_1<-lm(educ~exper+expersq+motheduc+fatheduc,data=Data_omit)

u_hat<-l_1$residuals

l_2<-lm(lwage~educ+exper+expersq+u_hat,data=Data_omit)

l_2sls<-ivreg(lwage~educ+exper+expersq|.-educ+motheduc+fatheduc,data=Data_omit)

#Durbin-Wu-Hausman test
ivreg2 <- function(form,endog,iv,data,digits=3){
  # library(MASS)
  # model setup
  r1 <- lm(form,data)
  y <- r1$fitted.values+r1$resid
  x <- model.matrix(r1)
  aa <- rbind(endog == colnames(x),1:dim(x)[2])  
  z <- cbind(x[,aa[2,aa[1,]==0]],data[,iv])  
  colnames(z)[(dim(z)[2]-length(iv)+1):(dim(z)[2])] <- iv  
  # iv coefficients and standard errors
  z <- as.matrix(z)
  pz <- z %*% (solve(crossprod(z))) %*% t(z)
  biv <- solve(crossprod(x,pz) %*% x) %*% (crossprod(x,pz) %*% y)
  sigiv <- crossprod((y - x %*% biv),(y - x %*% biv))/(length(y)-length(biv))
  vbiv <- as.numeric(sigiv)*solve(crossprod(x,pz) %*% x)
  res <- cbind(biv,sqrt(diag(vbiv)),biv/sqrt(diag(vbiv)),(1-pnorm(biv/sqrt(diag(vbiv))))*2)
  res <- matrix(as.numeric(sprintf(paste("%.",paste(digits,"f",sep=""),sep=""),res)),nrow=dim(res)[1])
  rownames(res) <- colnames(x)
  colnames(res) <- c("Coef","S.E.","t-stat","p-val")
  # First-stage F-test
  y1 <- data[,endog]
  z1 <- x[,aa[2,aa[1,]==0]]
  bet1 <- solve(crossprod(z)) %*% crossprod(z,y1)
  bet2 <- solve(crossprod(z1)) %*% crossprod(z1,y1)
  rss1 <- sum((y1 - z %*% bet1)^2)
  rss2 <- sum((y1 - z1 %*% bet2)^2)
  p1 <- length(bet1)
  p2 <- length(bet2)
  n1 <- length(y)
  fs <- abs((rss2-rss1)/(p2-p1))/(rss1/(n1-p1))
  firststage <- c(fs)
  firststage <- matrix(as.numeric(sprintf(paste("%.",paste(digits,"f",sep=""),sep=""),firststage)),ncol=length(firststage))
  colnames(firststage) <- c("First Stage F-test")
  # Hausman tests
  bols <- solve(crossprod(x)) %*% crossprod(x,y) 
  sigols <- crossprod((y - x %*% bols),(y - x %*% bols))/(length(y)-length(bols))
  vbols <- as.numeric(sigols)*solve(crossprod(x))
  sigml <- crossprod((y - x %*% bols),(y - x %*% bols))/(length(y))
  x1 <- x[,!(colnames(x) %in% "(Intercept)")]
  z1 <- z[,!(colnames(z) %in% "(Intercept)")]
  pz1 <- z1 %*% (solve(crossprod(z1))) %*% t(z1)
  biv1 <- biv[!(rownames(biv) %in% "(Intercept)"),]
  bols1 <- bols[!(rownames(bols) %in% "(Intercept)"),]
  # Durbin-Wu-Hausman chi-sq test:
  # haus <- t(biv1-bols1) %*% ginv(as.numeric(sigml)*(solve(crossprod(x1,pz1) %*% x1)-solve(crossprod(x1)))) %*% (biv1-bols1)
  # hpvl <- 1-pchisq(haus,df=1)
  # Wu-Hausman F test
  resids <- NULL
  resids <- cbind(resids,y1 - z %*% solve(crossprod(z)) %*% crossprod(z,y1))
  x2 <- cbind(x,resids)
  bet1 <- solve(crossprod(x2)) %*% crossprod(x2,y)
  bet2 <- solve(crossprod(x)) %*% crossprod(x,y)
  rss1 <- sum((y - x2 %*% bet1)^2)
  rss2 <- sum((y - x %*% bet2)^2)
  p1 <- length(bet1)
  p2 <- length(bet2)
  n1 <- length(y)
  fs <- abs((rss2-rss1)/(p2-p1))/(rss1/(n1-p1))
  fpval <- 1-pf(fs, p1-p2, n1-p1)
  #hawu <- c(haus,hpvl,fs,fpval)
  hawu <- c(fs,fpval)
  hawu <- matrix(as.numeric(sprintf(paste("%.",paste(digits,"f",sep=""),sep=""),hawu)),ncol=length(hawu))
  #colnames(hawu) <- c("Durbin-Wu-Hausman chi-sq test","p-val","Wu-Hausman F-test","p-val")
  colnames(hawu) <- c("Wu-Hausman F-test","p-val")  
  # Sargan Over-id test
  ivres <- y - (x %*% biv)
  oid <- solve(crossprod(z)) %*% crossprod(z,ivres)
  sstot <- sum((ivres-mean(ivres))^2)
  sserr <- sum((ivres - (z %*% oid))^2)
  rsq <- 1-(sserr/sstot)
  sargan <- length(ivres)*rsq
  spval <- 1-pchisq(sargan,df=length(iv)-1)
  overid <- c(sargan,spval)
  overid <- matrix(as.numeric(sprintf(paste("%.",paste(digits,"f",sep=""),sep=""),overid)),ncol=length(overid))
  colnames(overid) <- c("Sargan test of over-identifying restrictions","p-val")
  if(length(iv)-1==0){
    overid <- t(matrix(c("No test performed. Model is just identified")))
    colnames(overid) <- c("Sargan test of over-identifying restrictions")
  }
  full <- list(results=res, weakidtest=firststage, endogeneity=hawu, overid=overid)
  return(full)
}

end<-ivreg2(lwage~educ+exper+expersq,endog="exper",iv=c("motheduc","fatheduc"),data=Data_omit,digits=3)
F<-end$endogeneity[1]
p<-end$endogeneity[2]

# ====================================================== - 
## IV_error in variables ----
rm(list=ls(all=TRUE))

library(foreign)
library(AER)

Data<-read.dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Lectures/B1 Endogeneity/wage2.dta")
#OLS
l<-lm(lwage~educ+exper+tenure+married+south+urban+black,data=Data)
#IV pour aborder l'erreur dans le probl?me des variables
l_IV<-ivreg(lwage~educ+exper+tenure+married+south+urban+black+IQ|educ+exper+tenure+married+south+urban+black+KWW,data=Data)
#reduced form
l_IQ<-lm(IQ~exper+tenure+married+south+urban+black+KWW,data=Data)

h_IQ<-predict(l_IQ)
l_h_IQ<-lm(h_IQ~educ+exper+tenure+married+south+urban+black,data=Data)

# ====================================================== - 
## IV reg ---- 
rm(list=ls(all=TRUE))
library(foreign)
library(AER)

Data<-read.dta("H:/Teaching/CBS/BA-BMECV1031U/2024/Lectures/B1 Endogeneity/CARD.DTA")

#test for the validity of the instrument nearc4
t<-lm(educ~nearc4+exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669,data=Data)
#IV regression with nearc4 as IV for educ
l_iv<-ivreg(lwage~educ+exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669|nearc4+exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669,data=Data)
#simple OLS
l<-lm(lwage~educ+exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669,data=Data)


summary(t)
summary(l_iv)
summary(l)


# Heteroskedasticity robust inference after IV:
coeftest(l_iv, vcov=vcovHC(l_iv, type="HC0")) 

# ====================================================== - 
## overid_test ---- 
rm(list=ls(all=TRUE))
library(foreign)
library(AER)

Data<-read.dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Lectures/B1 Endogeneity/CARD.DTA")
#delete observations with missing values for fatheduc
Data<-Data[!is.na(Data$fatheduc),]
# 2sls with nearc4 and fatheduc as instruments for educ
l_2sls<-ivreg(lwage~educ+exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669|.-educ+nearc4+fatheduc,data=Data)

#determine p
u_hat<-l_2sls$residuals

aux<-lm(u_hat~nearc4+fatheduc+exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669,data=Data)

LM_statistic<-summary(aux)$r.squared*dim(Data)[1]
p_value<-1-pchisq(LM_statistic,1)


## Proxy variable ----
rm(list=ls(all=TRUE))
#R package which permits the import of .dta files
library(foreign)
MyData <- read.dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Lectures/B1 Endogeneity/WAGE2.DTA")

#linear regression without proxy
l<-lm(lwage~educ+exper+tenure+married+black+south+urban,data=MyData)
#linear rergression with proxy IQ for ability
l_IQ<-lm(lwage~educ+exper+tenure+married+black+south+urban+IQ,data=MyData)

summary(l)
summary(l_IQ)

# ====================================================== - 
## RESET test ----
rm(list=ls(all=TRUE))
library(foreign)
library(AER)
library(lmtest)

Data<-read.dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Lectures/B1 Endogeneity/CARD.DTA")

l_1<-lm(lwage~educ+exper+black+south,data=Data)

yhatsq<-predict(l_1)^2
yhatcub<-predict(l_1)^3

l_2<-lm(lwage~educ+exper+black+south+yhatsq+yhatcub,data=Data)

linearHypothesis(l_2,c("yhatsq=0","yhatcub=0"))

#alternative : use the function resettest in the package lmtest
aux<-lm(lwage~educ+exper+black+south,data=Data)

r_test<-resettest(aux,2:3,c("fitted"),data=Data)


############################################## -
# INTDEF DATA - TIME SERIES MODELLING GUIDE ----
# Purpose: Template for handling INTDEF.dta datasets in econometric analysis
# Search tag: [INTDEF_TEMPLATE_GUIDE]
############################################## -

# --- Load libraries ---
library(haven)  # To import Stata .dta files
library(dplyr)  # For data handling and overview

# --- Step 1: Import dataset ---
intdef <- read_dta("Opgaver datasæt/P3/ps3 data/INTDEF.DTA")

# --- Step 2: Inspect structure ---
# View variable names and first few rows
names(intdef)
head(intdef)

# --- Step 3: Data structure overview ---
# Variables in INTDEF.dta:
#  - year  : time index (1948–...)
#  - i3    : 3-month T-bill rate (interest rate)
#  - inf   : inflation rate
#  - def   : government deficit (optional regressor)
#  - i3_1, inf_1, def_1 : lagged values from previous year (t−1)
#  - ci3, cinf, cdef : first-differences, i.e. Δi3_t = i3_t − i3_(t−1)
#  - y77   : dummy variable (year 1977), used in later exercises
#
# The dataset is already *time structured* and contains both:
#  - level variables (i3, inf)
#  - lagged variables (inf_1)
#  - first-differences (cinf)
#
# → No need to create your own lags or diffs; they are precomputed.

# --- Step 4: Econometric interpretation ---
# The core equation models the Fisher-type relationship:
#    i3_t = β0 + β1 * inf_t + u_t
# where i3_t = nominal short-term interest rate
# and inf_t = inflation rate.
# Economically, β1 measures the sensitivity of nominal interest rates to inflation.
# In theory, β1 ≈ 1 under the Fisher effect (one-for-one adjustment).
#
# We can also estimate:
#  - in *levels* → long-run equilibrium relation
#  - in *first differences* → short-run response, removes trends
#  - with *lags* → delayed adjustment dynamics

# --- Step 5: Estimate models ---

## (a) Static levels model
m1 <- lm(i3 ~ inf, data = intdef)
summary(m1)
# Interprets how contemporaneous inflation affects interest rates.

## (b) Lagged predictor model
m2 <- lm(i3 ~ inf_1, data = intdef)
summary(m2)
# Tests whether last year's inflation affects the current interest rate.
# Economically: persistence or adjustment lag.

## (c) First-difference model
m3 <- lm(ci3 ~ cinf, data = intdef)
summary(m3)
# Focuses on changes over time, i.e., Δi3_t = β0 + β1*Δinf_t + Δu_t.
# Removes non-stationary trends and isolates short-run co-movements.

# --- Step 6: Compare results ---
# Compare β1 from m1, m2, m3.
# - If β1(m1) is significant but β1(m3) is smaller → indicates long-run trend.
# - If β1(m3) remains significant → implies strong short-run link.

############################################## -
# === Summary ===
# Use case:
#  - Use m1 for level analysis (long-run)
#  - Use m2 to test lagged effects
#  - Use m3 for first-difference (short-run)
#
# Dataset already provides all transformations (lag, diff),
# so `dynlm()` is not required.
#
# Econometric intuition:
#  - This dataset captures macroeconomic time dependence.
#  - Check for serial correlation (Durbin-Watson, Breusch-Godfrey)
#  - Consider robust SEs (Newey-West) for time-series inference.
############################################## -
# END BLOCK [INTDEF_TEMPLATE_GUIDE] 
############################################## -



# Problem Set 6

###Solution to Problem 2

rm(list=ls())
library(haven)
Data <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2024/Problem Sets/PS6/401ksubs.dta")

#a)
table1 <- list(count=table(Data$e401k),percent=round(as.numeric(table(Data$e401k)/length(Data$e401k)),3))
as.data.frame(table1)


#b)
reg1<-lm(e401k ~ inc + incsq + age + agesq + male, data=Data)
summary(reg1)

#with heteroskedasticity robust statistics
library(lmtest)
library(car)

coeftest(reg1, vcov=hccm(reg1, type="hc0"))

#d)
Data$fit<-reg1$fitted.values
sum(Data$fit>1)
sum(Data$fit<0)
summary(Data$fit)

#e)
Data$te401k<-as.numeric(Data$fit>=0.5)
as.data.frame(list(count=table(Data$te401k),percent=round(as.numeric(table(Data$te401k)/length(Data$te401k)),3)))

#f)
#predicted y among those with e401k=1
as.data.frame(list(count=table(subset(Data$te401k,Data$e401k==1)),
                   percent=round(as.numeric(table(subset(Data$te401k,Data$e401k==1))/length(subset(Data$te401k,Data$e401k==1))),3)))

#predicted y among those with e401k=0
as.data.frame(list(count=table(subset(Data$te401k,Data$e401k==0)),
                   percent=round(as.numeric(table(subset(Data$te401k,Data$e401k==0))/length(subset(Data$te401k,Data$e401k==0))),3)))

#g)
as.data.frame(table(Data$e401k,Data$te401k,dnn = list("e401k","te401k")))

(4607+1429)/nrow(Data)

rm(list=ls())



###Solution to Problem 4

Data <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2024/Problem Sets/PS6/GROGGER.dta")

#a)
Data$arr86<-as.numeric(Data$narr86>0)

reg1<-lm(arr86 ~ pcnv + avgsen + tottime + ptime86 + inc86 + black + hispan + born60, data=Data)
summary(reg1)


coeftest(reg1, vcov=hccm(reg1, type="hc0"))  #robust standard errors


0.5*reg1$coefficients[2]

#b)
#for homoscedastic SE:
library(car)
linearHypothesis(reg1, c("avgsen = tottime", "tottime=0"))

#for heteroscedasticity robust SE
linearHypothesis(reg1, c("avgsen = tottime", "tottime=0"), vcov. = hccm(reg1))

#c) 
probit<-glm(arr86 ~ pcnv + avgsen + tottime + ptime86 + inc86 + black + hispan + born60, family = binomial(link=probit), data=Data)
summary(probit)

logLik(probit) # Log Likelihood value
pseudoR2_probit<-1-probit$deviance/probit$null.deviance
pseudoR2_probit  # McFadden (1974) Pseudo R-squared
lrtest(probit) # LR test for overall significance

library(margins)
lev0<-data.frame(0.25,mean(Data$avgsen),mean(Data$tottime),mean(Data$ptime86),mean(Data$inc86),1,0,1)
names(lev0)<-c("pcnv","avgsen","tottime","ptime86","inc86","black","hispan","born60")
margins(probit, at=lev0)
p0<-predict(probit,lev0,type = "response")

lev1<-data.frame(0.75,mean(Data$avgsen),mean(Data$tottime),mean(Data$ptime86),mean(Data$inc86),1,0,1)
names(lev1)<-c("pcnv","avgsen","tottime","ptime86","inc86","black","hispan","born60")
margins(probit, at=lev1)
p1<-predict(probit,lev1,type = "response")

p1-p0 #effect on probability of arrest for pcnv that goes from 0.25 to 0.75


#d)
#compute fitted values
Data$fit<-probit$fitted.values
summary(Data$fit)

Data$tarr86<-as.numeric(Data$fit>=0.5)
as.data.frame(table(Data$arr86,Data$tarr86,dnn = list("arr86","tarr86")))

#compute percent correctly predicted
(1903+78)/2725

#pcp for arr86=0
1903/(1903+67)

#pcp for arr86=1
78/(677+78)


#e)
probit2<-glm(arr86 ~ pcnv + avgsen + tottime + ptime86 + inc86 + black + hispan + born60 + pcnvsq + pt86sq + inc86sq, family = binomial(link=probit), data=Data)
summary(probit2)

logLik(probit2) # Log Likelihood value
pseudoR2_probit2<-1-probit2$deviance/probit2$null.deviance
pseudoR2_probit2  # McFadden (1974) Pseudo R-squared
lrtest(probit2)

#Use in build Wald test
linearHypothesis(probit2, c("pcnvsq = pt86sq","pt86sq=inc86sq","inc86sq=0"))

#Alternatively perform LR test 
l_ur<-logLik(probit2) 
l_r <-logLik(probit)

#Value of the test statistic is
lr=2*(l_ur-l_r)
lr

#The P value is
1-pchisq(lr, 3)
#or
pchisq(lr, 3, lower.tail=FALSE)

#alternatively you can use a build in command for the LR test:
lrtest(probit,probit2)

rm(list=ls())


rm(list=ls(all=TRUE))

# Solutions to Problem set 5

rm(list=ls())


###Solution to Problem 1

# load data
library(haven)
smoke <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Problem Sets/PS5/SMOKE.DTA")


#d)
reg1<-lm(cigs ~ educ + age + agesq + lcigpric + restaurn, data=smoke)
summary(reg1)

library(car)
linearHypothesis(reg1, c("lcigpric=restaurn","restaurn=0"))


#e)
library(AER)

reg_IV<-ivreg(lincome~cigs + educ +  age + agesq|lcigpric + restaurn  + educ + age + agesq, data=smoke)
summary(reg_IV)

reg2<-lm(lincome ~ cigs + educ + age + agesq, data=smoke)
summary(reg2)


rm(list=ls())



###Solution to Problem 2

airfare <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Problem Sets/PS5/airfare.dta")
airfare97<-subset(airfare,year==1997)


#b)
reg1<-lm(lpassen ~ lfare + ldist + ldistsq, data=airfare97)
summary(reg1)


#d)
reg2<-lm(lfare ~ concen + ldist + ldistsq, data=airfare97)
summary(reg2)


#e)
reg_IV<-ivreg(lpassen ~ lfare + ldist + ldistsq |concen + ldist + ldistsq, data=airfare97)
summary(reg_IV)


#f)
#minimum in ldist
coef(reg_IV)[3]/(-2*coef(reg_IV)[4])

#minimum in dist
exp(coef(reg_IV)[3]/(-2*coef(reg_IV)[4]))


sum(airfare97$dist<336)

sum(airfare97$dist<336)/nrow(airfare97)


rm(list=ls())


# Solutions to Problem Set 4

rm(list=ls())


###Solution to Problem 1

# load data
library(haven)
wage2 <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2024/Problem Sets/PS4/WAGE2.DTA")


#a)
#lecture's model (2)
reg1<-lm(lwage ~ educ + exper + tenure + married + south + urban + black + IQ, data=wage2)
summary(reg1)

# use KWW
reg2<-lm(lwage ~ educ + exper + tenure + married + south + urban + black + KWW, data=wage2)
summary(reg2)


#b)
reg3<-lm(lwage ~ educ + exper + tenure + married + south + urban + black + IQ + KWW, data=wage2)
summary(reg3)


#c)
library(car)
linearHypothesis(reg3, c("IQ=KWW","KWW=0"))


rm(list=ls())



###Solution to Problem 2

wage2 <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2024/Problem Sets/PS4/WAGE2.DTA")


#a)
library(AER)

reg_IV<-ivreg(lwage~educ|sibs ,data=wage2)
summary(reg_IV)

reg1<-lm(lwage ~ sibs, data=wage2)
summary(reg1)


#b)
reg2<-lm(educ ~ brthord, data=wage2)
summary(reg2)


#c)
reg_IV.2<-ivreg(lwage~educ|brthord ,data=wage2)
summary(reg_IV.2)


#d)
reg3<-lm(educ ~ sibs + brthord, data=wage2)
summary(reg3)


#e)
reg_IV.3<-ivreg(lwage~educ + sibs|brthord +sibs,data=wage2)
summary(reg_IV.3)


#f)
educhat<-reg3$fitted.values
sibs.naomit<-subset(wage2$sibs, !is.na(wage2$brthord)) #remove obs in sibs for which there are not fitted values (to compute correlation the two vectors need to have the same dimension)

cor(educhat, sibs.naomit)


rm(list=ls())




###Solution to Problem 3

Data <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2024/Problem Sets/PS4/401KSUBS.DTA")

#a)
reg1<-lm(pira ~ p401k + inc + incsq + age + agesq, data=Data)
summary(reg1)
coeftest(reg1, vcov=hccm(reg1, type="hc0"))


#d)
reg2<-lm(p401k ~ e401k + inc + incsq + age + agesq, data=Data)
summary(reg2)
coeftest(reg2, vcov=hccm(reg2, type="hc0"))


#e)
reg_IV<-ivreg(pira~p401k + inc + incsq + age + agesq|e401k + inc + incsq + age + agesq, data=Data)
summary(reg_IV)

#the diagonal elements are the se:
sqrt(vcovHC(reg_IV, type = "HC0"))

#alternatively: ivpack currently not supported by R
#library(ivpack)
#robust.se(reg_IV) #to obtain robust standard errors


#f)
Data$nuhat<-reg2$residuals

reg3<-lm(pira ~ p401k + inc + incsq + age + agesq + nuhat, data=Data)
summary(reg3)
coeftest(reg3, vcov=hccm(reg3, type="hc0"))




###Solution to Problem 6

rm(list=ls())
card <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2024/Problem Sets/PS4/CARD.DTA")
attach(card)

# (a)
# OLS
summary(lm(lwage~educ+exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669))
#IV
summary(ivreg(lwage~educ+exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669
              |nearc4+exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669))
# reduced form regresion
educ.m<-lm(educ~nearc4+exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669)
summary(educ.m)

u_hat<-educ.m$residuals

summary(lm(lwage~educ+exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669+u_hat))


# (b)
summary(ivreg(lwage~educ+exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669|
                nearc4+nearc2+exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669))

# (c)
wage.m<-ivreg(lwage~educ+exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669|
                nearc4+nearc2+exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669)
summary(wage.m)
resid.m<-lm(residuals(wage.m)~exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669+nearc4+nearc2)
summary(resid.m)
(LM<-summary(resid.m)$r.squared*nobs(resid.m))
qchisq(0.95,1)
qchisq(0.9,1)

detach(card)


