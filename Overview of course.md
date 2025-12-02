**1. Introduction to Econometrics and Data Structures – Detailed Breakdown**

### 1.1 What econometrics is used for

Econometrics, in this course, is positioned as a toolbox to:

- **Quantify economic relationships**
    
    - Estimate parameters in models such as wage equations, demand/supply, return-to-education, etc.
        
    - Translate verbal/theoretical statements (“higher education increases wages”) into estimable equations.
        
- **Test economic hypotheses**
    
    - Null vs. alternative about parameters (e.g. “return to education = 0” vs. “> 0”).
        
    - Check whether theoretical restrictions are consistent with observed data.
        
- **Evaluate policies and interventions**
    
    - Estimate average treatment effects (ATE, ATT, etc.).
        
    - Compare outcomes for treated vs. control groups, controlling for covariates.
        
    - Use regression as an adjustment tool to mimic ceteris paribus comparisons.
        
- **Forecast and predict**
    
    - Use estimated models to predict outcomes for given covariate values.
        
    - Produce point predictions and prediction intervals for decision support.
        
- **Support decision-making in firms and policy institutions**
    
    - Quantify cost–benefit trade-offs.
        
    - Identify drivers of key performance indicators (sales, productivity, risk measures).
        

---

### 1.2 Economic vs. econometric models

The material distinguishes clearly between the conceptual/theoretical model and the empirical/statistical implementation.

- **Economic model (theoretical / structural)**
    
    - Abstract representation of agents’ behaviour and markets.
        
    - Specifies functional relationships driven by economic theory, e.g.:
        
        - Labour supply model, demand–supply model, human capital model, etc.
            
    - Typically written as:
        
        - ( y = f(x_1, x_2, \dots, x_k, u) )  
            where (u) captures unobserved factors, preferences, technology, etc.
            
    - Parameters have a **structural interpretation** (e.g. marginal propensity to consume, wage elasticity of labour supply).
        
- **Econometric model (statistical / estimable)**
    
    - Operational version of the economic model imposed with:
        
        - A specific functional form (often linear in parameters).
            
        - A stochastic specification for the error term.
            
    - Canonical regression representation:
        
        - ( y_i = \beta_0 + \beta_1 x_{1i} + \dots + \beta_k x_{ki} + u_i )
            
    - Includes explicit assumptions about:
        
        - Sampling process (random sampling of units).
            
        - Error term (zero conditional mean, homoskedasticity, etc., depending on context).
            
    - Bridges theory and data by making the model **estimable** with OLS, ML, etc.
        
- **Link between the two**
    
    - Start with an economic story → derive a functional relationship → translate into an econometric specification.
        
    - The quality of empirical work depends on how well the econometric model maps the economic model and institutional context.
        

---

### 1.3 Data types / data structures

The course consistently works with four canonical data structures (following Wooldridge ch. 1):

#### 1.3.1 Cross-sectional data

- **Definition**
    
    - Data on many units (individuals, firms, regions, countries, etc.) observed **once**.
        
    - Ordering of observations is irrelevant; each observation is conceptually a draw from a population.
        
- **Key properties / assumptions in this course**
    
    - Random sampling: ((y_i, x_i)) for (i=1,\dots,n) are treated as i.i.d. draws.
        
    - Independence across units; no time dimension.
        
    - Most of **Part A** (basic OLS, OLS topics, policy analysis with one cross-section) is framed in this setting.
        
- **Typical examples**
    
    - Wage data for a sample of workers in a given year.
        
    - Firm-level profitability data for a given accounting period.
        
    - Household consumption survey at one point in time.
        

#### 1.3.2 Time series data

- **Definition**
    
    - Observations on one unit (country, firm, market, asset) over multiple time periods.
        
    - Ordering by time is intrinsic.
        
- **Key features**
    
    - Time dependence: serial correlation in errors and variables.
        
    - Potential non-stationarity, trends, seasonality.
        
    - In this course, time series is mainly background (the core exam focus is cross-sectional / microeconometric applications), but terminology is relevant for understanding “data structure” and for selected examples.
        
- **Typical examples**
    
    - Quarterly GDP for one country.
        
    - Daily stock prices for one company.
        
    - Monthly unemployment rate for a given region.
        

#### 1.3.3 Pooled cross sections

- **Definition**
    
    - Two or more independent cross-sections drawn at different points in time.
        
    - Composition of the sample may change across periods; units are not tracked.
        
- **Use cases in the course**
    
    - Pre/post policy evaluation: compare outcomes before and after a reform with different individuals in each period.
        
    - Difference-in-differences style reasoning (at a simple level) with repeated cross-sections.
        
- **Key points**
    
    - Need to account for time indicators in regressions.
        
    - No panel structure: you do not follow the same unit over time.
        

#### 1.3.4 Panel (longitudinal) data

- **Definition**
    
    - Data that follow the **same units over time**: (i = 1, \dots, N), (t = 1, \dots, T).
        
    - Each unit has multiple observations across time.
        
- **Relevance in the course**
    
    - Conceptually important to understand unobserved heterogeneity and fixed vs. random effects.
        
    - In this specific course, panel concepts are mainly background/intuition; estimation of full panel models is not the primary exam focus, but you must recognise the structure and why it is powerful.
        
- **Key advantages**
    
    - Control for time-invariant unobserved heterogeneity (unit-specific effects).
        
    - Better identification of causal effects under weaker assumptions than pure cross-section.
        

---

### 1.4 Causality vs. correlation (ceteris paribus and counterfactual logic)

The introductory block explicitly distinguishes descriptive associations from causal relations:

- **Correlation / association**
    
    - Statistical co-movement between variables (e.g. wages and education move together).
        
    - Does not by itself justify a causal statement.
        
    - Can be driven by omitted variables, reverse causality, measurement error, or pure coincidence.
        
- **Causal effect (ceteris paribus effect)**
    
    - Effect of changing one variable (e.g. years of education) **holding all else equal**.
        
    - Formalised via a **counterfactual**:
        
        - For individual (i), compare (y_i(1)) and (y_i(0)) (treated vs. untreated potential outcomes).
            
    - The key econometric problem: we never observe both potential outcomes at once → requires assumptions and research design.
        
- **Role of econometric assumptions**
    
    - Zero conditional mean / exogeneity: (E(u_i \mid x_i) = 0) is the basic regression condition that enables a ceteris paribus interpretation of (\beta)-coefficients in cross-sectional OLS.
        
    - Violations (endogeneity) undermine causal interpretation; this sets up later topics (OVB, IV, policy evaluation).
        
- **Practical exam-relevant framing**
    
    - Be able to articulate the difference between:
        
        - “We observe a positive correlation between education and wages.”
            
        - “Under the regression assumptions, we can interpret the estimated coefficient on education as the causal ceteris paribus effect of an additional year of education on wages.”
            
    - Recognise that the entire course can be viewed as refining when and how the regression coefficients can be given a causal interpretation.
        

---

If you want, next step can be to decompose **“Multiple Linear Regression & OLS (Core Model)”** in the same level of detail (model specification, matrix notation, OLS formula, finite-sample properties, interpretation, R², etc.)


**2. Multiple Linear Regression & OLS (Core Model) – Detailed Breakdown**

### 2.1 Model specification in scalar form

Baseline multiple regression model:

- **Population model**
    
    - For observation (i = 1, \dots, n):
        
        - (y_i = \beta_0 + \beta_1 x_{1i} + \dots + \beta_k x_{ki} + u_i)
            
- **Objects**
    
    - (y_i): dependent (explained) variable
        
    - (x_{ji}): regressor (j) for unit (i) (explanatory variables; can be continuous or dummy)
        
    - (\beta_j): unknown parameters (intercept and slopes)
        
    - (u_i): error term capturing all unobserved factors affecting (y_i) not explicitly included in the model
        
- **Ceteris paribus interpretation (given assumptions)**
    
    - (\beta_j) measures the partial effect of (x_j) on (y) holding all other regressors constant:
        
        - (\beta_j = \partial E(y \mid x) / \partial x_j) in the linear model
            

---

### 2.2 Model specification in matrix form

Compact representation for all (n) observations:

- **Stacked notation**
    
    - (y): (n \times 1) vector of dependent variable
        
    - (X): (n \times (k+1)) regressor matrix:
        
        - First column is a column of ones for the intercept
            
        - Remaining columns are the (k) regressors
            
    - (\beta): ((k+1) \times 1) parameter vector
        
    - (u): (n \times 1) error vector
        
- **Matrix model**
    
    - (y = X \beta + u)
        
- **Interpretation**
    
    - This form is used to derive the OLS estimator compactly:
        
        - Minimisation of (u'u = (y - X\beta)'(y - X\beta)) with respect to (\beta)
            

---

### 2.3 OLS estimation (derivation and formulas)

OLS is defined as the estimator that minimises the **sum of squared residuals (SSR)**.

- **Objective function**
    
    - (SSR(\beta) = \sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_{1i} - \dots - \beta_k x_{ki})^2)
        
    - In matrix form: (SSR(\beta) = (y - X\beta)'(y - X\beta))
        
- **Normal equations (first-order conditions)**
    
    - Take derivative of (SSR(\beta)) w.r.t. (\beta) and set equal to zero:
        
        - (X'(y - X\beta) = 0)
            
    - Rearrange:
        
        - (X'X\hat{\beta} = X'y)
            
- **Closed-form OLS estimator**
    
    - Provided (X'X) is invertible (no perfect multicollinearity):
        
        - (\hat{\beta} = (X'X)^{-1} X'y)
            
- **Component-wise interpretation**
    
    - (\hat{\beta}_0, \hat{\beta}_1, \dots, \hat{\beta}_k) are sample-based estimates of the unknown parameters
        
    - Computation is handled by software (e.g. R), but the exam logic relies on knowing:
        
        - OLS minimises SSR
            
        - Solution is (\hat{\beta} = (X'X)^{-1} X'y)
            
        - Normal equations hold: (X'e = 0), where (e = y - X\hat{\beta})
            

---

### 2.4 Fitted values and residuals

Once (\hat{\beta}) is obtained:

- **Fitted values**
    
    - For each observation:
        
        - (\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_{1i} + \dots + \hat{\beta}_k x_{ki})
            
    - In matrix form:
        
        - (\hat{y} = X\hat{\beta})
            
    - (\hat{y}) is the projection of (y) onto the column space of (X).
        
- **Residuals**
    
    - For each observation:
        
        - (\hat{u}_i = y_i - \hat{y}_i)
            
    - In matrix form:
        
        - (e = y - \hat{y} = y - X\hat{\beta})
            
    - Residuals are sample counterparts to the unobserved errors (u_i).
        
- **Projection matrices (optional but often used in slides)**
    
    - Projection matrix:
        
        - (P = X(X'X)^{-1} X'), then (\hat{y} = Py)
            
    - Residual-maker matrix:
        
        - (M = I - P), then (e = My)
            

---

### 2.5 Algebraic properties of OLS (finite-sample, no distributional assumptions)

These properties follow purely from the OLS minimisation, no probability assumptions needed:

- **Orthogonality conditions**
    
    - (X'e = 0)
        
        - Residuals are orthogonal to each regressor (including the intercept).
            
    - Implications:
        
        - Sample covariance between each regressor and residual is zero.
            
- **Residuals and fitted values**
    
    - With an intercept in the model:
        
        - (\sum_{i=1}^n \hat{u}_i = 0)
            
        - The residuals sum to zero.
            
        - The average fitted value equals the average actual value:
            
            - (\bar{\hat{y}} = \bar{y})
                
    - Residuals and fitted values are orthogonal:
        
        - (\sum_{i=1}^n \hat{u}_i \hat{y}_i = 0)
            
- **Decomposition of total variation**
    
    - Total Sum of Squares (TSS):
        
        - (TSS = \sum_{i=1}^n (y_i - \bar{y})^2)
            
    - Explained Sum of Squares (ESS):
        
        - (ESS = \sum_{i=1}^n (\hat{y}_i - \bar{y})^2)
            
    - Sum of Squared Residuals (SSR):
        
        - (SSR = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \sum_{i=1}^n \hat{u}_i^2)
            
    - Exact identity:
        
        - (TSS = ESS + SSR)
            

These identities underpin the definition of (R^2).

---

### 2.6 Interpretation of coefficients

Given the linear model and under the usual exogeneity conditions (zero conditional mean), coefficients have a ceteris paribus interpretation:

- **Continuous regressors (level–level specification)**
    
    - (y_i = \beta_0 + \beta_1 x_{1i} + \dots + \beta_k x_{ki} + u_i)
        
    - (\beta_j) is the change in the conditional mean of (y) associated with a one-unit increase in (x_j), holding all other regressors fixed:
        
        - (E(y \mid x_1, \dots, x_j + 1, \dots, x_k) - E(y \mid x_1, \dots, x_j, \dots, x_k) = \beta_j)
            
- **Dummy (binary) regressors**
    
    - If (x_j) is an indicator (0/1):
        
        - (\beta_j) measures the discrete change in the conditional mean of (y) when the indicator switches from 0 to 1, holding other regressors constant.
            
        - Interpretation: difference in means between the group with (x_j = 1) and the group with (x_j = 0), conditional on controls.
            
- **Intercept**
    
    - (\beta_0) is the expected value of (y) when all regressors equal zero.
        
    - Interpretation is context dependent; often not economically meaningful but algebraically necessary for unbiasedness of slopes and for (R^2) properties.
        
- **Link to causal interpretation**
    
    - If regressors are exogenous (in the sense defined formally later in the course), (\beta_j) can be interpreted as a causal effect; otherwise, it is only a partial correlation.
        

---

### 2.7 Goodness of fit: (R^2) and adjusted (R^2)

The course treats (R^2) as the main summary of in-sample fit for linear regression.

- **Coefficient of determination ((R^2))**
    
    - Definition:
        
        - (R^2 = \dfrac{ESS}{TSS} = 1 - \dfrac{SSR}{TSS})
            
    - Range:
        
        - (0 \leq R^2 \leq 1) when an intercept is included.
            
    - Interpretation:
        
        - Fraction of sample variation in (y) around its mean that is explained by the fitted values from the regression.
            
        - Higher (R^2) means a better in-sample fit, but:
            
            - It does **not** guarantee causal validity.
                
            - It always (weakly) increases when adding regressors, even if they are irrelevant.
                
- **Adjusted (R^2) (if included in the slides under this heading)**
    
    - Corrects for the number of regressors and sample size:
        
        - (\bar{R}^2 = 1 - \dfrac{SSR/(n - k - 1)}{TSS/(n - 1)})
            
    - Penalises adding regressors that do not improve the model sufficiently.
        
    - Useful for model comparison with different numbers of regressors.
        
- **Exam-relevant points**
    
    - Be able to:
        
        - State the decomposition (TSS = ESS + SSR).
            
        - Write the formula for (R^2).
            
        - Explain what “(x%) of the variation in (y) is explained by the regression” means.
            
        - Emphasise that (R^2) is not a measure of model correctness or causal identification.
            

---

### 2.8 Link to sampling properties (bridge to Gauss–Markov topic)

Although the detailed assumptions are formally handled under the “Gauss–Markov Assumptions & OLS Properties” heading, the multiple regression core topic already sets up:

- **Sampling perspective**
    
    - Treat data as one realisation from a population.
        
    - (\hat{\beta}) is a random vector with:
        
        - (E(\hat{\beta}) = \beta) under exogeneity (unbiasedness).
            
        - (\text{Var}(\hat{\beta})) as a function of (\sigma^2) and ((X'X)^{-1}).
            
- **Error variance estimate**
    
    - (\hat{\sigma}^2 = \dfrac{SSR}{n - k - 1})
        
    - Used to construct standard errors:
        
        - (\widehat{\text{Var}}(\hat{\beta}) = \hat{\sigma}^2 (X'X)^{-1})
            

These elements are expanded later with formal assumptions, hypothesis testing, and robust standard errors, but conceptually they belong to the core multiple regression and OLS toolkit.

---

If you want to proceed systematically, next block would be **3. Gauss–Markov Assumptions & OLS Properties** (formal assumptions, unbiasedness, variance, Gauss–Markov theorem, and what happens when assumptions fail).


**3. Gauss–Markov Assumptions & OLS Properties – Detailed Breakdown**

### 3.1 The five Gauss–Markov assumptions (MLR.1–MLR.5)

The core content is Wooldridge’s five Gauss–Markov assumptions for cross-sectional multiple regression.

#### 3.1.1 MLR.1 – Linearity in parameters

- **Statement**  
    The population model is:  
    [  
    y = \beta_0 + \beta_1 x_1 + \dots + \beta_k x_k + u  
    ]  
    i.e. linear in the parameters (\beta_0, \dots, \beta_k).
    
- **Scope**
    
    - Allows nonlinear functions of the regressors (logs, squares, interactions), as long as the model is linear in the (\beta)’s.
        
    - Defines the **multiple linear regression (MLR) model** used throughout the course.
        

#### 3.1.2 MLR.2 – Random sampling

- **Statement**  
    The sample ({(x_{i1},\dots,x_{ik}, y_i): i=1,\dots,n}) is a random sample from the population that satisfies MLR.1.
    
- **Implications**
    
    - Observations are i.i.d. (independent and identically distributed).
        
    - Justifies application of law of large numbers and CLT to derive sampling and asymptotic properties of OLS.
        

#### 3.1.3 MLR.3 – No perfect collinearity

- **Statement**  
    No independent variable is constant, and there are **no exact linear relationships** among the independent variables (no perfect multicollinearity).
    
- **Technical role**
    
    - Guarantees that (X'X) is invertible so that (\hat{\beta} = (X'X)^{-1}X'y) is well-defined.
        
- **Course emphasis**
    
    - Distinguish between:
        
        - Perfect multicollinearity (violates MLR.3 → model not estimable).
            
        - High but not perfect multicollinearity (inflates variances but does not violate the assumption).
            
    - Covered explicitly as a learning objective in your notes.
        

#### 3.1.4 MLR.4 – Zero conditional mean (exogeneity)

- **Statement**  
    The error has zero expected value given any values of the regressors:  
    [  
    E(u \mid x_1,\dots,x_k) = 0  
    ]
    
- **Economic meaning**
    
    - All omitted factors captured in (u) are, on average, unrelated to the included regressors.
        
    - No omitted variable bias, no simultaneity bias, and no measurement error that correlates with regressors.
        
- **Role in the course**
    
    - This is the **key exogeneity condition**; your own notes emphasise that the “key assumption” in the MLR model is exactly this.
        
    - Underpins the unbiasedness results and the causal ceteris paribus interpretation of (\beta_j).
        

#### 3.1.5 MLR.5 – Homoskedasticity

- **Statement**  
    The variance of the error, conditional on the regressors, is constant:  
    [  
    \text{Var}(u \mid x_1,\dots,x_k) = \sigma^2  
    ]
    
- **Technical meaning**
    
    - Error variance does not depend on levels of the regressors.
        
    - Error has a **scalar variance–covariance matrix** (\sigma^2 I_n).
        
- **Course positioning**
    
    - Needed for the **usual OLS variance formulas** and for OLS to be BLUE.
        
    - Your notes specifically list “the role of the homoskedasticity assumption” as a separate learning target.
        

---

### 3.2 Which assumptions do what? (unbiasedness vs. efficiency)

The slides and Wooldridge separate the functions of the assumptions:

- **Unbiasedness of OLS**
    
    - Requires: MLR.1 (linearity), MLR.2 (random sampling), MLR.3 (no perfect collinearity), and MLR.4 (zero conditional mean).
        
    - Homoskedasticity (MLR.5) is **not** needed for unbiasedness.
        
- **Standard variance formulas and Gauss–Markov efficiency**
    
    - Requires: MLR.1–MLR.5 jointly.
        
    - MLR.5 is crucial for:
        
        - The closed-form variance formula under homoskedasticity.
            
        - The statement that OLS is the **best linear unbiased estimator** (BLUE).
            

Your own note learning objectives mirror this breakdown (“minimum requirements for an unbiased OLS estimator”, “role of the homoskedasticity assumption”, “Gauss–Markov theorem and variance decomposition”).

---

### 3.3 Unbiasedness of OLS

Under MLR.1–MLR.4:

- **OLS estimator**  
    [  
    \hat{\beta} = (X'X)^{-1}X'y  
    ]
    
- **Rewrite using model**  
    [  
    y = X\beta + u \Rightarrow \hat{\beta} = (X'X)^{-1}X'(X\beta + u) = \beta + (X'X)^{-1}X'u  
    ]
    
- **Conditional expectation**  
    [  
    E(\hat{\beta} \mid X) = \beta + (X'X)^{-1}X' E(u \mid X) = \beta  
    ]  
    because (E(u \mid X) = 0) by MLR.4.
    
- **Result**
    
    - (\hat{\beta}) is **unbiased** for (\beta) (conditional and unconditional).
        
    - At component level: (E(\hat{\beta}_j) = \beta_j) for all (j).
        

This is exactly what your learning objective 3.8 refers to (“minimum requirements for an unbiased OLS estimator”).

---

### 3.4 Variance of OLS under homoskedasticity

Given MLR.1–MLR.5 and using matrix notation:

- **Variance–covariance matrix of (\hat{\beta})**  
    Under homoskedasticity with (\text{Var}(u \mid X) = \sigma^2 I_n), Wooldridge derives:  
    [  
    \text{Var}(\hat{\beta} \mid X) = \sigma^2 (X'X)^{-1}  
    ]
    
- **Estimator of (\sigma^2)**
    
    - Use residuals (\hat{u} = y - X\hat{\beta}) and define:  
        [  
        \hat{\sigma}^2 = \frac{\hat{u}'\hat{u}}{n - k - 1}  
        ]  
        which is the unbiased estimator of the error variance.
        
- **Estimated variance matrix**
    
    - Plug (\hat{\sigma}^2) in:  
        [  
        \widehat{\text{Var}}(\hat{\beta} \mid X) = \hat{\sigma}^2 (X'X)^{-1}  
        ]
        
- **Use cases in the course**
    
    - Construction of:
        
        - Standard errors for each (\hat{\beta}_j).
            
        - t-statistics and confidence intervals in the “classical” homoskedastic OLS framework.
            
    - Later contrasted with heteroskedasticity-robust formula under violations (A2 topic).
        

Your notes explicitly flag “use the Gauss–Markov Theorem to formulate and interpret the components of the variance of the OLS estimators.”

---

### 3.5 Gauss–Markov theorem and BLUE

The course uses Wooldridge’s formulation of the Gauss–Markov theorem:

- **Theorem (Gauss–Markov)**  
    Under MLR.1–MLR.5:
    
    - Each OLS estimator (\hat{\beta}_j) has **minimum variance** among all **linear unbiased** estimators of (\beta_j).
        
    - More generally, any linear combination (c'\beta) is best estimated by (c'\hat{\beta}) among linear unbiased estimators.
        
- **Interpretation**
    
    - “Best” means **smallest variance** in the class of linear unbiased estimators → **BLUE**:
        
        - Best Linear Unbiased Estimator.
            
    - Justifies using OLS rather than searching for other linear unbiased estimators when Gauss–Markov assumptions hold.
        
- **Course-level takeaway**
    
    - If assumptions MLR.1–MLR.5 hold, OLS is:
        
        - Unbiased (by MLR.1–MLR.4).
            
        - Minimum-variance among linear unbiased estimators (by adding MLR.5).
            
    - If MLR.4 fails → OLS biased; Gauss–Markov theorem collapses.
        
    - If MLR.5 fails → OLS remains unbiased but no longer efficient (you can do better with GLS/FGLS).
        

---

### 3.6 Efficiency notions: finite-sample vs. asymptotic

The course also connects Gauss–Markov to asymptotic efficiency (Wooldridge, Ch. 5).

- **Finite-sample efficiency (Gauss–Markov)**
    
    - Under MLR.1–MLR.5:
        
        - OLS has the **smallest finite-sample variance** in the class of linear unbiased estimators (BLUE).
            
- **Asymptotic efficiency**
    
    - Under the same assumptions and in large samples, OLS has the **smallest asymptotic variance** among a broad class of consistent linear estimators that can be written in the generalized form (e.g., GLS-type estimators under homoskedasticity).
        
- **Impact of heteroskedasticity**
    
    - Under heteroskedasticity (Assumption 5’ in the OLS topics slides):
        
        - OLS remains unbiased and consistent if MLR.4 still holds.
            
        - But:
            
            - Homoskedastic variance formula is wrong.
                
            - t- and F-tests based on the classical formula are invalid.
                
            - OLS is no longer efficient; feasible GLS or robust methods can dominate.
                

---

### 3.7 How this is framed in your course objectives

Your consolidated notes explicitly encode the Gauss–Markov block as a cluster of learning goals:

- Explain the **importance and interpretation** of the Gauss–Markov assumptions.
    
- Discuss **multicollinearity** in relation to MLR.3.
    
- Use the Gauss–Markov theorem to **write down and interpret the variance of OLS**.
    
- List the **minimum conditions for unbiased OLS** (MLR.1–MLR.4).
    
- Assess the **role of homoskedasticity** (MLR.5) in multiple regression.
    

That is essentially the full scope of what sits under the heading:

> 3. **Gauss–Markov Assumptions & OLS Properties** –  
>     formal assumptions, unbiasedness, variance of OLS, Gauss–Markov theorem, and efficiency.



**4. Violations of Gauss–Markov & OLS Topics – Detailed Breakdown**

### 4.1 Heteroskedasticity

**Definition**

- Homoskedasticity (MLR.5):  
    (\text{Var}(u_i \mid x_i) = \sigma^2) for all (i).
    
- Heteroskedasticity:  
    (\text{Var}(u_i \mid x_i) = \sigma_i^2) where (\sigma_i^2) can differ across observations.
    

So the error variance depends on the level of one or more regressors or other characteristics of the observation.

**Typical economic sources**

- Scale effects: variance of income errors larger for high-income individuals than low-income individuals.
    
- Proportionality: variance of sales errors increasing with firm size.
    
- Model misspecification: omitted variables that interact with regressors, or non-linearities approximated by linear models.
    

**Consequences for OLS**

- If MLR.1–MLR.4 hold (in particular exogeneity), but MLR.5 fails:
    
    - (\hat{\beta}) remains **unbiased** and **consistent**.
        
    - The usual homoskedastic variance formula (\sigma^2 (X'X)^{-1}) is **wrong**.
        
    - Classical t- and F-tests based on homoskedastic standard errors are **invalid** (size distortion).
        
    - OLS is no longer efficient; there exist alternative linear unbiased estimators with smaller variance (GLS/FGLS).
        

So the main issue is **inference and efficiency**, not bias, as long as exogeneity is intact.

---

### 4.2 Detecting heteroskedasticity

The course typically uses a mix of informal diagnostics and formal tests:

**Informal diagnostics**

- Plot residuals (\hat{u}_i) against:
    
    - Fitted values (\hat{y}_i).
        
    - Individual regressors (x_{ji}).
        
- Look for patterns where the spread of residuals **systematically increases or decreases** with the level of a variable (e.g., “fan-shaped” pattern).
    

**Formal tests**

1. **Breusch–Pagan (BP) test**
    
    - Null hypothesis (H_0): homoskedasticity  
        (\text{Var}(u_i \mid x_i) = \sigma^2) (constant).
        
    - Alternative (H_1): variance is a linear function of one or more regressors.
        
    - Implementation (LM version):
        
        1. Estimate the original OLS model and obtain residuals (\hat{u}_i).
            
        2. Compute squared residuals (\hat{u}_i^2).
            
        3. Regress (\hat{u}_i^2) on chosen variables (often the original regressors).  
            Let the R² from this auxiliary regression be (R^2_{\text{aux}}).
            
        4. Test statistic:  
            ( \text{BP} = n \cdot R^2_{\text{aux}} )  
            which is asymptotically (\chi^2_q) under (H_0), where (q) is the number of regressors in the auxiliary regression (excluding intercept).
            
    - Decision rule: Reject (H_0) for large values of BP (p-value small).
        
2. **White test**
    
    - Null (H_0): homoskedasticity, no particular functional form imposed.
        
    - Alternative (H_1): very general heteroskedasticity (variance depending arbitrarily on regressors).
        
    - Implementation:
        
        1. Estimate OLS and compute (\hat{u}_i^2).
            
        2. Run auxiliary regression of (\hat{u}_i^2) on:
            
            - Original regressors,
                
            - Their squares,
                
            - Selected cross-products.
                
        3. Let (R^2_{\text{aux}}) be from this regression.
            
        4. Test statistic:  
            ( \text{White} = n \cdot R^2_{\text{aux}} \sim \chi^2_m ) asymptotically under (H_0), where (m) is the number of explanatory variables in the auxiliary regression (excluding intercept).
            
    - White test is more flexible but uses more degrees of freedom.
        

In R, these tests are typically called via functions (e.g. `bptest` for Breusch–Pagan) but you must know the underlying logic and hypotheses.

---

### 4.3 Heteroskedasticity-robust standard errors (Eicker–Huber–White)

When heteroskedasticity is present, the variance of (\hat{\beta}) is no longer (\sigma^2 (X'X)^{-1}). The course uses the **“sandwich”** (robust) estimator:

**Population form**

- Under heteroskedasticity:  
    [  
    \text{Var}(\hat{\beta} \mid X) = (X'X)^{-1} \left( X' \Omega X \right) (X'X)^{-1}  
    ]  
    where (\Omega = \text{diag}(\sigma_1^2, \dots, \sigma_n^2)).
    

**EHW (HC) sample estimator**

- Replace (\sigma_i^2) with (\hat{u}_i^2):  
    [  
    \widehat{\text{Var}}_{\text{robust}}(\hat{\beta} \mid X)  
    = (X'X)^{-1} \left( \sum_{i=1}^n x_i x_i' \hat{u}_i^2 \right) (X'X)^{-1}  
    ]  
    where (x_i) is the ((k+1))-vector of regressors for observation (i).
    
- The diagonal elements give **heteroskedasticity-robust variances** for each coefficient; square roots are **robust standard errors**.
    

**Implications for inference**

- Use robust standard errors in t- and F-tests:
    
    - t-statistic: (\displaystyle t_j = \frac{\hat{\beta}_j - \beta_{j,0}}{\text{s.e.}_{\text{robust}}(\hat{\beta}_j)}).
        
    - F-tests use robust covariance matrix for joint hypotheses.
        
- Asymptotically valid under very general forms of heteroskedasticity, assuming exogeneity and standard regularity conditions.
    
- Software: in R, usually by specifying a robust covariance matrix in `coeftest`/`vcovHC` etc.
    

Key exam point: **robust standard errors fix inference under heteroskedasticity but do not change the OLS estimates (\hat{\beta})**.

---

### 4.4 Generalized Least Squares (GLS) and Feasible GLS (FGLS)

Robust standard errors keep OLS but adapt inference. GLS/FGLS change the estimator to improve efficiency when error variance–covariance structure is known or well-approximated.

**GLS setup**

- Suppose:  
    [  
    \text{Var}(u \mid X) = \Omega \neq \sigma^2 I_n  
    ]  
    where (\Omega) is **known**, symmetric and positive definite (e.g., diagonal but with unequal diagonal elements, or more general correlation structures).
    
- Then the GLS estimator is:  
    [  
    \hat{\beta}_{\text{GLS}} = (X' \Omega^{-1} X)^{-1} X' \Omega^{-1} y  
    ]
    
- Equivalent interpretation:
    
    - Transform model by premultiplying with (\Omega^{-1/2}) (or per-observation scaling).
        
    - After transformation, the error term is homoskedastic; OLS on transformed data = GLS.
        

**Special case: Weighted Least Squares (WLS)**

- If (\Omega) is diagonal with entries (\sigma_i^2), then:
    
    - Multiply each observation by weight (w_i = 1/\sigma_i).
        
    - This yields WLS; observations with lower error variance get larger weight.
        

**Feasible GLS (FGLS)**

- In practice, (\Omega) is unknown.
    
- Strategy:
    
    1. Specify a **parametric form** of heteroskedasticity, e.g.  
        (\sigma_i^2 = \sigma^2 h(z_i, \gamma)) where (z_i) are known variables (e.g. some regressors) and (\gamma) parameters.
        
    2. Estimate parameters from an auxiliary regression (often on squared residuals).
        
    3. Construct (\hat{\Omega}) using estimated variances.
        
    4. Compute:  
        [  
        \hat{\beta}_{\text{FGLS}} = (X' \hat{\Omega}^{-1} X)^{-1} X' \hat{\Omega}^{-1} y  
        ]
        
- Properties:
    
    - Not exactly unbiased in finite samples.
        
    - Under correct specification of (\Omega) and standard conditions, FGLS is **asymptotically more efficient** than OLS.
        

**Course-level trade-off**

- **Robust OLS**:
    
    - Simple, model-free with respect to heteroskedasticity.
        
    - Corrects inference but does not gain efficiency relative to GLS.
        
- **FGLS/WLS**:
    
    - Can be more efficient than robust OLS if the heteroskedasticity model is correct.
        
    - Sensitive to misspecification of (\Omega).
        

You should be able to state the GLS estimator and explain when and why it is used, not necessarily implement complex FGLS specifications in detail under exam conditions.

---

### 4.5 Other “OLS topics” beyond the basic model

Under the “OLS topics” umbrella, the course typically collects a set of additional issues that either relax Gauss–Markov assumptions or refine model specification:

#### 4.5.1 Functional form misspecification & RESET-type reasoning

- Concern: The true conditional expectation (E(y \mid x)) might not be linear in the included regressors.
    
- Symptoms:
    
    - Nonlinear patterns in residual plots.
        
    - Systematic under-/over-prediction in certain ranges.
        
- RESET-style approach:
    
    - Augment the regression with powers or functions of the fitted values (or regressors), e.g. (\hat{y}^2, \hat{y}^3).
        
    - Perform an F-test for joint significance of added terms.
        
    - Rejection suggests misspecified functional form or omitted nonlinearities.
        

Even if a formal RESET test is not emphasised, the **idea** is important: check whether linear specification is adequate.

#### 4.5.2 Multicollinearity (non-perfect)

- Distinct from MLR.3 (no **perfect** collinearity).
    
- When regressors are **highly correlated**, but not perfectly:
    
    - OLS is still unbiased.
        
    - Var((\hat{\beta}_j)) becomes large → standard errors inflate, t-statistics small.
        
    - Coefficient estimates can be numerically unstable and sensitive to small data changes.
        
- Diagnostics:
    
    - Correlation matrix of regressors.
        
    - Variance Inflation Factors (VIF).
        
- Remedies:
    
    - Re-specification (merge or drop nearly redundant variables).
        
    - Focus on joint tests instead of individual coefficients.
        

#### 4.5.3 Omitted and redundant variables (within the OLS framework)

- Omitted relevant variable (correlated with included regressors):
    
    - Violates MLR.4 → endogeneity and bias (treated more systematically under the “Endogeneity” topic).
        
- Redundant variable (irrelevant regressor):
    
    - Does not bias coefficients of interest but can increase variance, so trade-off between bias and variance in model selection.
        

Even though omitted-variable bias is central to the “Endogeneity” block, it also shows up in OLS topics as a specification issue.

#### 4.5.4 Outliers, leverage, and influence

- **Outliers**: observations with extreme (y)-values relative to fitted regression.
    
- **Leverage points**: observations with extreme (x)-values (far from the bulk of the data).
    
- **Influential observations**: points that have a large impact on (\hat{\beta}) when included/excluded (e.g. measured via Cook’s distance).
    
- Implications:
    
    - Can distort OLS estimates and standard errors.
        
    - Diagnostic plots and influence measures are standard; robust regression is an alternative but often beyond basic exam scope.
        

#### 4.5.5 Prediction vs. estimation

- For prediction:
    
    - Interest is in minimising prediction error (MSE) for (y^{\text{new}}), not necessarily in consistent estimation of structural (\beta).
        
    - Even a mildly misspecified or purely predictive model can be useful if it forecasts well.
        
- For causal estimation:
    
    - Correct specification and exogeneity dominate; prediction performance is secondary.
        

This distinction frames why violations of assumptions are more serious in causal analysis than in purely predictive tasks.

---

Net result: under the heading

> **4. Violations of Gauss–Markov & OLS Topics**

you are expected to:

- Define heteroskedasticity and explain its consequences for OLS.
    
- Derive/recognise robust variance formula and explain robust standard errors.
    
- Explain and outline Breusch–Pagan and White tests.
    
- State GLS/FGLS and its efficiency role under known/parametric (\Omega).
    
- Discuss broader OLS topics: multicollinearity, functional form, specification errors, outliers, and the estimation vs. prediction perspective.


**5. Policy Analysis & Treatment Effects – Detailed Breakdown**

### 5.1 Potential outcomes framework

- **Outcome variable**
    
    - (y): observed outcome (e.g. earnings, test score, house price).
        
- **Treatment indicator**
    
    - (w \in {0,1}): binary policy/treatment indicator
        
        - (w = 1): unit is treated (participates in programme / affected by policy).
            
        - (w = 0): unit is untreated / control.
            
- **Potential outcomes**
    
    - (y_i(0)): outcome for unit (i) **without** treatment.
        
    - (y_i(1)): outcome for unit (i) **with** treatment.
        
    - For each (i), we conceptually have two potential outcomes, but we only ever observe one:
        
        - If (w_i=1): observe (y_i = y_i(1)), (y_i(0)) is counterfactual.
            
        - If (w_i=0): observe (y_i = y_i(0)), (y_i(1)) is counterfactual.
            
- **Individual treatment effect**
    
    - (\tau_i = y_i(1) - y_i(0)).
        
    - Typically unobserved because we never see both (y_i(1)) and (y_i(0)) for the same unit.
        

This framework is the conceptual backbone for all policy evaluation in the course.

---

### 5.2 Treatment vs. control groups

- **Treatment group**
    
    - Units with (w_i = 1).
        
    - Observed outcomes: (y_i = y_i(1)).
        
- **Control group**
    
    - Units with (w_i = 0).
        
    - Observed outcomes: (y_i = y_i(0)).
        
- **Naive mean comparison**
    
    - Simple difference in sample means:  
        [  
        \bar{y}_1 - \bar{y}_0  
        ]  
        where (\bar{y}_1) is the mean of (y) for treated units, (\bar{y}_0) for controls.
        
    - Equals the treatment effect **plus** any selection bias due to systematic differences between treated and control units.
        

The entire methodological agenda is to construct conditions under which such differences can be interpreted as causal.

---

### 5.3 Average treatment parameters (ATE, ATT, etc.)

The slides focus on **average** treatment effects rather than individual (\tau_i):

- **Average Treatment Effect (ATE)**  
    [  
    \tau_{ATE} = E\big[y(1) - y(0)\big]  
    ]
    
    - Expected effect of treatment for a **randomly drawn unit** from the population.
        
- **Average Treatment Effect on the Treated (ATT)**  
    [  
    \tau_{ATT} = E\big[y(1) - y(0) \mid w = 1\big]  
    ]
    
    - Expected effect for those who actually receive treatment.
        
    - Many policy questions are implicitly ATT-type (“what did the programme do for participants?”).
        
- **Average Treatment Effect on the Controls (ATC)**  
    [  
    \tau_{ATC} = E\big[y(1) - y(0) \mid w = 0\big]  
    ]
    
    - Effect the treatment would have had for those who did not receive it (less central but conceptually symmetric).
        

In the lecture, ATE is the canonical target; ATT is often discussed in examples (e.g. job training, policy adoption).

---

### 5.4 Fundamental identification problem

- **Observed outcome**  
    [  
    y_i = w_i y_i(1) + (1 - w_i) y_i(0)  
    ]
    
- **Problem**
    
    - For each (i) we observe only one of (y_i(0), y_i(1)); the other is missing.
        
    - ATE and ATT involve expectations over counterfactuals; cannot be computed directly from data.
        
- **Identification strategy**
    
    - Use assumptions (random assignment / unconfoundedness) and covariates (x_i) to render control outcomes a valid proxy for missing counterfactual outcomes.
        

This sets up the role of regression and design in recovering causal effects.

---

### 5.5 Random / “unconfounded” assignment

Core identifying condition:

- **Unconfoundedness (conditional independence)**  
    [  
    {y(0), y(1)} ;\perp; w ;\mid; x  
    ]
    
    - Conditional on observed covariates (x), treatment status is **as good as random**.
        
    - Interpretation:
        
        - Once we control for (x), treated and control units are comparable; selection into treatment does not depend on unobserved outcome determinants.
            
- **Special case – pure random assignment**
    
    - If treatment is assigned by a genuine randomized experiment:
        
        - ({y(0), y(1)} \perp w) even **without** conditioning on (x).
            
        - Then the simple difference in means identifies ATE:  
            [  
            E[y \mid w=1] - E[y \mid w=0] = \tau_{ATE}  
            ]
            
- **Connection to zero conditional mean**
    
    - In regression form, define an error term such that:  
        [  
        E(u \mid x, w) = 0  
        ]
        
    - Under unconfoundedness, the error term satisfies a zero conditional mean given ((x,w));  
        OLS on (y) versus ((w,x)) can then be used to estimate ATE.
        

Unconfoundedness is the conceptual analogue of exogeneity for the treatment assignment mechanism.

---

### 5.6 Regression adjustment – restricted vs. unrestricted

Regression is used as a **control strategy** to adjust for covariates (x) when estimating treatment effects.

#### 5.6.1 Restricted regression adjustment (RRA)

- **Model**  
    [  
    y_i = \alpha + \tau w_i + x_i' \beta + u_i  
    ]
    
    - Common slope (\beta) for all units.
        
    - Treatment effect (\tau) enters as a **single additive shift** in the intercept.
        
- **Interpretation**
    
    - (\tau) measures the **conditional ATE**, assuming:
        
        - Linear conditional expectation in ((w,x)).
            
        - Same slope coefficients for treated and control groups.
            
- **Estimation**
    
    - Run one OLS regression on the pooled sample.
        
    - Coefficient on (w) is the regression-adjusted estimate of the treatment effect.
        

#### 5.6.2 Unrestricted regression adjustment (URA)

- **Model with full interactions**  
    [  
    y_i = \alpha_0 + x_i' \beta_0 + w_i(\alpha_1 + x_i' \beta_1) + u_i  
    ]
    
    - Equivalently: allow intercept and slopes to differ by treatment status.
        
    - In practice implemented by:
        
        - Splitting sample by (w) and estimating separate regressions:
            
            - Treated: (y_i = \alpha_1 + x_i' \beta_1 + u_i).
                
            - Control: (y_i = \alpha_0 + x_i' \beta_0 + u_i).
                
    - Then compute fitted values for each group and construct an average of differences to get an estimate of (\tau_{ATE}) or (\tau_{ATT}).
        
- **Comparison with RRA**
    
    - **RRA**: Assumes same functional relationship between (y) and (x) for treated and controls (common slope).
        
    - **URA**: Allows fully group-specific functional forms; more flexible but uses more parameters and may be less precise in small samples.
        
- **Exam-relevant points**
    
    - Know that **regression adjustment** means including (x) as controls in a regression with (w).
        
    - Understand:
        
        - RRA: single pooled regression, common slopes.
            
        - URA: separate regressions by (w) or pooled with full interaction structure.
            
    - Recognise that both rely on unconfoundedness for causal interpretation.
        

---

### 5.7 Policy evaluation with cross-sectional data

Using cross-sectional data for policy analysis under unconfoundedness:

- **Simple difference in means** (no controls):
    
    - Valid in randomized experiments.
        
    - In observational data, generally biased unless treatment independent of potential outcomes.
        
- **Regression with controls**:
    
    - Use RRA-type model:  
        [  
        y_i = \alpha + \tau w_i + x_i'\beta + u_i  
        ]
        
    - Interpret (\hat{\tau}) as an estimate of ATE (or conditional ATE) **if**:
        
        - Unconfoundedness holds given (x).
            
        - Linear form is a good approximation to the true conditional expectation function.
            
- **Role of covariates**
    
    - Reduce bias by conditioning on observed confounders that jointly affect treatment and outcome.
        
    - Increase precision if they explain variation in (y).
        

The course positions this as the baseline empirical strategy for evaluating many policies (training programmes, educational reforms, etc.) when only one cross-section is available.

---

### 5.8 Policy evaluation with pooled cross sections and DID

When data are available for at least **two time periods**, before and after a policy change, and for a **treatment group** and a **control group**, the lecture introduces **difference-in-differences (DID)**:

- **Setting**
    
    - Groups: (G \in {T, C}) (treated vs. control).
        
    - Periods: (t \in {0,1}) (before vs. after policy).
        
    - Policy affects only the treated group in period 1.
        
- **DID estimator (four means formula)**  
    [  
    \text{DID} = \big[\bar{y}_{T,1} - \bar{y}_{T,0}\big] - \big[\bar{y}_{C,1} - \bar{y}_{C,0}\big]  
    ]
    
    - Change over time in treated group minus change over time in control group.
        
    - Under the **parallel trends** assumption, DID identifies the ATE of the policy on treated units.
        
- **Regression implementation**
    
    - Define:
        
        - (d_T = 1) for treated group, 0 otherwise.
            
        - (d_2 = 1) for post-period (time 1), 0 for pre-period (time 0).
            
        - Interaction (d_T \cdot d_2).
            
    - Run regression (possibly with extra controls):  
        [  
        y_{it} = \beta_0 + \beta_1 d_T + \delta_0 d_2 + \delta_1 (d_T \cdot d_2) + \text{other controls} + u_{it}  
        ]
        
    - Coefficient (\delta_1) is the **DID estimate** of the treatment effect (often interpreted as an average treatment effect).
        
- **Connection to pooled cross-sections**
    
    - Data are typically structured as pooled cross-sections: different units sampled before and after, not necessarily a true panel following the same units.
        
    - DID can still be applied as long as groups are comparable over time and parallel trends holds.
        
- **Examples in slides**
    
    - House price effects of an incinerator location (treatment: houses near incinerator; policy: siting decision; outcome: price).
        
    - DID regression on log house prices yields percentage effect of siting decision.
        

---

### 5.9 Integration into the overall course logic

Under **Policy Analysis & Treatment Effects**, you are expected to be able to:

- Work with **potential outcomes** (y(0), y(1)) and treatment indicator (w).
    
- Define and interpret **ATE**, **ATT**, and related parameters.
    
- Explain the **fundamental missing data problem** and why assumptions are needed.
    
- State and interpret **random / unconfounded assignment** and its link to zero conditional mean in regression.
    
- Describe and distinguish **restricted** vs. **unrestricted** regression adjustment, and how each is implemented.
    
- Explain how to use **cross-sectional** regression with controls for policy evaluation under unconfoundedness.
    
- Set up and interpret a **difference-in-differences** (DID) design based on pooled cross sections, including:
    
    - The four-means DID formula.
        
    - The DID regression with group, time, and interaction dummies.
        
    - Interpretation of the interaction coefficient as an average treatment effect under parallel trends.
        

That is the full scope under the heading:

> **5. Policy Analysis & Treatment Effects**  
> potential outcomes, treatment vs. control, ATE, random/unconfounded assignment, regression adjustment, and policy evaluation using cross-sectional and pooled data.


**6. Endogeneity & Instrumental Variables – Detailed Breakdown**

### 6.1 Exogenous vs. endogenous regressors

- **Baseline regression model**  
    [  
    y_i = \beta_0 + \beta_1 x_{1i} + \dots + \beta_k x_{ki} + u_i  
    ]
    
- **Exogenous regressor**
    
    - Satisfies the zero conditional mean condition:  
        [  
        E(u_i \mid x_{1i},\dots,x_{ki}) = 0  
        ]
        
    - Equivalently, each regressor is uncorrelated with the error term:  
        [  
        \text{Cov}(x_{ji}, u_i) = 0 \quad \forall j  
        ]
        
    - Under exogeneity, OLS slope coefficients have a ceteris paribus (causal) interpretation.
        
- **Endogenous regressor**
    
    - Violates zero conditional mean:  
        [  
        E(u_i \mid x_{1i},\dots,x_{ki}) \neq 0  
        ]  
        or (\text{Cov}(x_{ji},u_i) \neq 0) for at least one regressor.
        
    - Generates **biased and inconsistent** OLS estimates for coefficients on endogenous regressors.
        
    - Endogeneity is always defined relative to a specific equation (structural interpretation).
        

---

### 6.2 Sources of endogeneity

Standard decomposition (all imply (E(u\mid X)\neq 0)):

1. **Omitted variables**
    
    - A relevant determinant of (y) is excluded from the regression and:
        
        - Has a non-zero effect on (y) (enters the true model).
            
        - Is correlated with one or more included regressors.
            
    - Classical example:
        
        - True model: (y = \beta_0 + \beta_1 x + \beta_2 z + u).
            
        - Estimated model: (y = \tilde{\beta}_0 + \tilde{\beta}_1 x + v) with (v = \beta_2 z + u).
            
        - If (\text{Cov}(x,z) \neq 0), then (\text{Cov}(x,v)\neq 0) → endogeneity.
            
2. **Simultaneity (reverse causality)**
    
    - (y) and some regressors are jointly determined in an economic system.
        
    - Example: supply and demand, price and quantity determined simultaneously.
        
    - Structural equations contain (y) and an endogenous regressor on the right-hand side of each other’s equations; reduced-form errors are correlated with regressors from the system.
        
3. **Measurement error in regressors**
    
    - True regressor (x_i^* = x_i + \nu_i) where (\nu_i) is measurement error.
        
    - Under classical measurement error (independent of (x_i^*) and (u_i)):
        
        - The observed regressor is correlated with the composite error:  
            [  
            y_i = \beta_0 + \beta_1 x_i^* + u_i  
            = \beta_0 + \beta_1 (x_i - \nu_i) + u_i  
            = \beta_0 + \beta_1 x_i + (u_i - \beta_1 \nu_i)  
            ]  
            so (\text{Cov}(x_i, u_i - \beta_1\nu_i)\neq 0) in general.
            
    - Attenuation bias for the coefficient on the mismeasured regressor.
        
4. **Functional form misspecification**
    
    - True conditional mean is nonlinear in (X), but model is misspecified as linear in a way that leaves systematic components in the error.
        
    - Example:
        
        - True model: (y = \gamma_0 + \gamma_1 x + \gamma_2 x^2 + u).
            
        - Estimated model: (y = \beta_0 + \beta_1 x + v) with (v = \gamma_2 x^2 + u).
            
        - If (\text{Cov}(x, x^2)\neq 0) (which it is), then (\text{Cov}(x,v)\neq 0) → endogeneity.
            

All of these mechanisms can be summarised as **the regressor “soaking up” part of the error**.

---

### 6.3 Omitted variable bias (OVB)

Consider the true model:  
[  
y_i = \beta_0 + \beta_1 x_i + \beta_2 z_i + u_i  
]

Suppose we estimate a regression omitting (z_i):  
[  
y_i = \tilde{\beta}_0 + \tilde{\beta}_1 x_i + v_i  
]  
with (v_i = \beta_2 z_i + u_i).

Let the population regression of (z_i) on (x_i) be:  
[  
z_i = \delta_0 + \delta_1 x_i + r_i  
]

Then the population relationship between (\tilde{\beta}_1) and (\beta_1) is:  
[  
\tilde{\beta}_1 = \beta_1 + \beta_2 \delta_1  
]

- **Bias term**  
    [  
    \text{Bias}(\tilde{\beta}_1) = E(\tilde{\beta}_1) - \beta_1 = \beta_2 \delta_1  
    ]
    
- **Direction of bias**
    
    - If (\beta_2 > 0) (omitted variable increases (y)) and (\delta_1 > 0) (omitted variable positively correlated with (x)), then (\tilde{\beta}_1) is biased **upward**.
        
    - If signs differ, bias can be downward.
        
- **Conditions for no OVB**
    
    - Either (\beta_2 = 0) (omitted variable irrelevant in the true model), or
        
    - (\delta_1 = 0) (omitted variable uncorrelated with the included regressor).
        

In the multiple-regression case, the same logic applies with vector notation; OVB is simply the manifestation of (E(u\mid X)\neq 0).

---

### 6.4 Proxy variables

When a relevant variable (z) is unobserved, it may be possible to include a **proxy** variable (q) instead.

Target structure:

- True model:  
    [  
    y_i = \beta_0 + \beta_1 x_i + \beta_2 z_i + u_i  
    ]
    
- (z_i) unobserved, but a proxy (q_i) is observed.
    

A **useful proxy** in this context is one that:

1. Is correlated with the omitted variable:  
    [  
    \text{Cov}(q_i, z_i) \neq 0  
    ]
    
2. Is uncorrelated with the structural error term (conditional on other regressors):  
    [  
    \text{Cov}(q_i, u_i) = 0  
    ]
    

Then including (q_i) in the regression:  
[  
y_i = \gamma_0 + \gamma_1 x_i + \gamma_2 q_i + e_i  
]  
can reduce or remove the bias from omitting (z_i), because part of the effect of (z_i) is captured by (q_i).

Important distinction:

- A **proxy variable** is used as an additional regressor to approximate an omitted determinant.
    
- An **instrumental variable** is excluded from the structural equation and used only to identify the effect of an endogenous regressor (see below).
    

---

### 6.5 Instrumental variables (IV): conditions

Setup with one potentially endogenous regressor:

- Structural equation:  
    [  
    y_i = \beta_0 + \beta_1 x_i + w_i' \beta_w + u_i  
    ]  
    where:
    
    - (x_i): endogenous regressor (e.g. treatment, price, quantity).
        
    - (w_i): vector of exogenous controls.
        
    - (u_i): error term.
        
- Instrument(s): (z_i) (possibly vector).
    

**Two core IV conditions:**

1. **Relevance**  
    [  
    \text{Cov}(z_i, x_i) \neq 0  
    ]
    
    - Instruments must be correlated with the endogenous regressor after conditioning on controls.
        
    - In practice: strong first stage.
        
2. **Exogeneity (validity / exclusion)**  
    [  
    \text{Cov}(z_i, u_i) = 0  
    ]
    
    - Instruments are uncorrelated with the structural error term.
        
    - Economically: (z_i) affects (y_i) only through its effect on (x_i) (after conditioning on (w_i)).
        

For multiple endogenous regressors and instruments:

- Collect instruments into matrix (Z) and regressors into (X).
    
- **Rank condition for identification**:
    
    - The matrix (E(Z'X)) must have full column rank equal to the number of endogenous regressors (so there are enough independent instruments).
        

---

### 6.6 IV estimand and 2SLS implementation

#### 6.6.1 Just-identified case (one endogenous regressor, one instrument, plus exogenous controls)

- Structural equation:  
    [  
    y_i = \beta_0 + \beta_1 x_i + w_i' \beta_w + u_i  
    ]
    
- First-stage population relationship:  
    [  
    x_i = \pi_0 + \pi_1 z_i + w_i' \pi_w + v_i  
    ]
    

**IV/2SLS estimand for (\beta_1)**

- In the simplified case without controls:  
    [  
    \beta_1^{IV} = \frac{\text{Cov}(z_i, y_i)}{\text{Cov}(z_i, x_i)}  
    ]
    
- With controls, use partialling-out (residualise (y) and (x) on (w)) and apply the same ratio to residuals.
    

#### 6.6.2 Two-stage least squares (2SLS)

Matrix formulation with instruments (Z), regressors (X) and projection matrix (P_Z):

- **Projection matrix onto instrument space**:  
    [  
    P_Z = Z (Z' Z)^{-1} Z'  
    ]
    
- **2SLS estimator**:  
    [  
    \hat{\beta}_{2SLS} = (X' P_Z X)^{-1} X' P_Z y  
    ]
    

Operational two-stage interpretation:

1. **First stage**
    
    - Regress each endogenous regressor on all instruments and exogenous controls:  
        [  
        x_i = \hat{\pi}_0 + \hat{\pi}_1 z_i + w_i' \hat{\pi}_w + \hat{v}_i  
        ]
        
    - Obtain fitted values (\hat{x}_i).
        
2. **Second stage**
    
    - Replace (x_i) by (\hat{x}_i) in the structural equation and run OLS:  
        [  
        y_i = \hat{\beta}_0 + \hat{\beta}_1 \hat{x}_i + w_i' \hat{\beta}_w + \hat{u}_i  
        ]
        
    - The coefficient (\hat{\beta}_1) is the 2SLS estimate of the effect of (x) on (y).
        

Key properties under valid instruments:

- (\hat{\beta}_{2SLS}) is **consistent** for structural parameters.
    
- If (x) were actually exogenous, 2SLS would be consistent but less efficient than OLS (larger variance).
    

---

### 6.7 Hausman-type reasoning (OLS vs. IV)

The Hausman logic is used to test for endogeneity of regressors and to choose between OLS and IV:

- **Null hypothesis (H_0)**: Regressors are exogenous.
    
    - Under (H_0):
        
        - OLS is **consistent and efficient** (within the class of linear unbiased estimators, under homoskedasticity).
            
        - IV/2SLS is also **consistent** but less efficient (higher variance).
            
- **Alternative hypothesis (H_1)**: One or more regressors are endogenous.
    
    - Under (H_1):
        
        - OLS is **inconsistent** (biased in large samples).
            
        - IV/2SLS remains **consistent** (provided instruments are valid).
            

**Implication**

- Under (H_0), OLS and IV estimators converge to the **same** parameter vector.
    
- Under (H_1), they converge to **different** limits.
    

**Hausman test statistic (conceptually)**

- Generic form:  
    [  
    H = (\hat{\beta}_{IV} - \hat{\beta}_{OLS})'  
    \left[ \widehat{\text{Var}}(\hat{\beta}_{IV}) - \widehat{\text{Var}}(\hat{\beta}_{OLS}) \right]^{-1}  
    (\hat{\beta}_{IV} - \hat{\beta}_{OLS})  
    ]
    
- Under (H_0): (H) is asymptotically (\chi^2) with degrees of freedom equal to the number of tested coefficients.
    
- Interpretation:
    
    - If (H) is small → fail to reject (H_0) → OLS and IV are statistically similar → use OLS for efficiency.
        
    - If (H) is large → reject (H_0) → evidence of endogeneity → prefer IV/2SLS.
        

Courses often emphasise the **intuition** rather than full derivation:

- Compare two estimators:
    
    - One efficient under (H_0) (OLS).
        
    - One robust under (H_1) (IV).
        
- Significant difference suggests violation of exogeneity.
    

---

### 6.8 Integration in the course logic

Under **Endogeneity & Instrumental Variables**, you are expected to be able to:

- Define **exogenous** vs. **endogenous** regressors in terms of the zero conditional mean condition.
    
- Explain **sources of endogeneity**: omitted variables, simultaneity, measurement error, and functional form misspecification.
    
- Derive and interpret the **omitted variable bias** formula and sign.
    
- Explain when and how **proxy variables** can mitigate omitted variable bias.
    
- State the **IV conditions**: relevance and exogeneity, plus identification via the rank condition in multi-instrument settings.
    
- Write down and interpret the **IV/2SLS estimand** and the practical **two-stage procedure**.
    
- Apply **Hausman-type reasoning** to motivate when to trust OLS and when to prefer IV, based on a comparison of the two estimators.


7. **Simultaneous Equations Models (SEM)**
    

- **Motivation: simultaneity as endogeneity source**
    
    - When regressors and the dependent variable are jointly determined in an equilibrium system (e.g. crime–police, supply–demand), regressors become endogenous and OLS is inconsistent.
        
    - Distinction from other endogeneity sources (omitted variables, measurement error, functional form): here feedback works through a system of structural equations, not just missing controls.
        
- **Structural vs. reduced-form equations**
    
    - **Structural equations**: economically motivated, causal, ceteris paribus relationships (e.g. labour supply, labour demand, crime equation, police reaction equation). Parameters (e.g. supply/demand elasticities) have direct economic interpretation.
        
    - **Reduced-form equations**: express each endogenous variable purely as a function of exogenous variables and error terms (no endogenous regressors on RHS). Derived by solving the system; coefficients are nonlinear functions of structural parameters.
        
    - Use of reduced form to show that endogenous regressors are correlated with structural errors → simultaneity bias for OLS.
        
- **Canonical demand–supply style systems**
    
    - Standard setup: two structural equations (e.g. demand and supply for a good; labour hours supplied and demanded; crime and prison population / police force size).
        
    - Endogenous variables (e.g. price and quantity, crime and prisoners) jointly determined by equilibrium condition (e.g. quantity supplied = quantity demanded).
        
    - Exogenous “shifters” specific to each equation (e.g. supply shifters z₁ only in supply, demand shifters z₂ only in demand) are crucial for identification and IV construction.
        
    - Autonomy requirement: each structural equation must make economic sense on its own (e.g. some housing–saving systems violate this and are better modelled via single-equation demand functions).
        
- **Simultaneity bias and inconsistency of OLS**
    
    - Show formally in a 2-equation SEM that the endogenous regressor’s reduced-form error contains the structural error → Cov(endogenous regressor, structural error) ≠ 0.
        
    - Derive sign of simultaneity bias from feedback parameters (e.g. crime raises police, police reduce crime → OLS underestimates crime-reducing effect of police).
        
    - Link to exam-style reasoning: “lpris (prisoners) is endogenous because of simultaneity between crime and prison population” in crime example.
        
- **Identification in SEMs**
    
    - **Order condition (2-equation case)**: to identify a structural equation, the other equation must contain at least one exogenous variable that is excluded from the current equation (number of excluded exogenous ≥ number of included endogenous regressors).
        
    - **Rank condition (intuition)**: excluded exogenous variables must actually shift the endogenous regressors (non-zero reduced-form coefficients).
        
    - Examples used in the course: labour supply vs wage offer; inflation–openness system; crime–prison population; theft–security officers. Identify which equation is / is not identified based on excluded shifters.
        
    - Mention that in systems with ≥3 equations, identification becomes a system-wide (matrix) property but intuition stays: each structural equation needs “own” excluded exogenous variables.
        
- **IV and 2SLS estimation in SEMs**
    
    - Instrument set for a given structural equation: all exogenous variables in the system (included and excluded) are valid candidates; the excluded ones provide variation to identify endogenous regressors.
        
    - **Two-Stage Least Squares (2SLS) procedure**:
        
        - Stage 1: Regress each endogenous RHS variable on all exogenous variables; obtain fitted values (predicted endogenous variables).
            
        - Stage 2: Regress the dependent variable on these fitted values and any included exogenous regressors → structural parameter estimates.
            
    - Compare OLS vs 2SLS in examples (e.g. labour supply, crime–prison system): OLS shows simultaneity bias; 2SLS recovers economically plausible signs and magnitudes under valid instruments.
        
    - Link to exam tools:
        
        - Use F-test in first stage to check instrument relevance (“partial relationship with lpris”).
            
        - Use Sargan overidentification test when overidentified, and regression-based Hausman test to detect endogeneity of a regressor (as in crime-prison exam outline


**8. Maximum Likelihood Estimation (MLE) – Detailed Breakdown**

### 8.1 Setup: conditional likelihood for cross-section models

- **Data structure**
    
    - Cross-sectional observations: ((y_i, x_i)), (i = 1,\dots,n).
        
    - (y_i): scalar outcome (can be continuous, discrete, or mixed).
        
    - (x_i): (K)-dimensional regressor vector.
        
- **Parametric conditional model**
    
    - Assume the **conditional density / mass function** of (y) given (x) is known up to a finite-dimensional parameter vector (\theta):  
        [  
        f(y_i \mid x_i; \theta)  
        ]
        
    - “Conditional MLE”: everything is conditioned on (x_i); the randomness is in (y_i).
        
- **Independence assumption**
    
    - ((y_i, x_i)) are i.i.d. across (i) (standard cross-section assumption).
        
    - Conditional independence:  
        [  
        f(y_1,\dots,y_n \mid x_1,\dots,x_n; \theta) = \prod_{i=1}^n f(y_i \mid x_i; \theta).  
        ]
        
- **Likelihood function (conditional)**
    
    - Given observed sample ({(y_i,x_i)}_{i=1}^n), define:  
        [  
        L(\theta) = \prod_{i=1}^n f(y_i \mid x_i; \theta).  
        ]
        
    - This is the **conditional likelihood** of (\theta) given the data.
        

---

### 8.2 Log-likelihood and score (first-order conditions)

- **Log-likelihood**
    
    - Work with the log of the likelihood:  
        [  
        \ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log f(y_i \mid x_i; \theta).  
        ]
        
    - Sums are easier to differentiate and numerically more stable than products.
        
- **Maximum likelihood estimator (MLE)**
    
    - Define:  
        [  
        \hat{\theta} = \arg\max_{\theta \in \Theta} \ \ell(\theta).  
        ]
        
    - (\Theta) is the parameter space (assumed to be compact / appropriate set in the asymptotic theory).
        
- **Score function and first-order conditions (FOC)**
    
    - **Score** for observation (i):  
        [  
        s_i(\theta) = \frac{\partial}{\partial \theta} \log f(y_i \mid x_i; \theta).  
        ]
        
    - Sample score:  
        [  
        S_n(\theta) = \sum_{i=1}^n s_i(\theta)  
        = \frac{\partial}{\partial \theta} \ell(\theta).  
        ]
        
    - FOC characterising interior ML solutions:  
        [  
        S_n(\hat{\theta}) = 0.  
        ]
        
    - Typically solved numerically (Newton–Raphson, BFGS, etc.) except in special cases (e.g. normal linear model).
        
- **Hessian**
    
    - Matrix of second derivatives:  
        [  
        H_n(\theta) = \frac{\partial^2}{\partial \theta \partial \theta'} \ell(\theta)  
        = \sum_{i=1}^n \frac{\partial^2}{\partial \theta \partial \theta'} \log f(y_i \mid x_i; \theta).  
        ]
        
    - Concavity (negative definite Hessian) ensures global maximum.
        

---

### 8.3 Example: linear regression as an ML problem

- **Normal linear regression model**
    
    - Conditional distribution:  
        [  
        y_i \mid x_i \sim N(x_i' \beta, \sigma^2),  
        ]  
        where (\theta = (\beta, \sigma^2)).
        
    - Conditional density:  
        [  
        f(y_i \mid x_i; \beta,\sigma^2)  
        = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(  
        -\frac{(y_i - x_i'\beta)^2}{2\sigma^2}  
        \right).  
        ]
        
- **Log-likelihood**  
    [  
    \ell(\beta,\sigma^2)  
    = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2)
    
    - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - x_i'\beta)^2.  
        ]
        
- **Maximising w.r.t. (\beta)**
    
    - For fixed (\sigma^2), maximising (\ell) is equivalent to **minimising** (\sum (y_i - x_i'\beta)^2).
        
    - FOC gives:  
        [  
        \hat{\beta}_{ML} = \hat{\beta}_{OLS} = (X'X)^{-1}X'y.  
        ]
        
- **Maximising w.r.t. (\sigma^2)**
    
    - Leads to:  
        [  
        \hat{\sigma}^2_{ML} = \frac{1}{n}\sum_{i=1}^n (y_i - x_i'\hat{\beta})^2,  
        ]  
        which differs from the usual unbiased OLS estimator only by the denominator (n instead of n–k–1).
        
- **Takeaway**
    
    - Under normal errors, OLS is the **ML estimator** for (\beta).
        
    - This connects LS and ML and motivates why LS performs well under approximate normality.
        

---

### 8.4 Kullback–Leibler (KL) motivation and pseudo-true parameter

- **True conditional density**
    
    - Denote the **true** conditional density by:  
        [  
        f_0(y \mid x).  
        ]
        
- **Model family**
    
    - We specify a **parametric family**:  
        [  
        {f(y \mid x; \theta): \theta \in \Theta},  
        ]  
        which may or may not contain (f_0).
        
- **Kullback–Leibler divergence**
    
    - For a given (\theta), the KL divergence from the true conditional density to the model density:  
        [  
        KL(f_0, f_\theta)  
        = E_{f_0}\left[\log \frac{f_0(Y\mid X)}{f(Y\mid X; \theta)}\right].  
        ]
        
    - Properties:
        
        - (KL(f_0, f_\theta) \ge 0).
            
        - (KL(f_0, f_\theta) = 0) iff (f(y\mid x;\theta) = f_0(y\mid x)) almost surely.
            
- **Pseudo-true parameter**
    
    - Define:  
        [  
        \theta_0 = \arg\min_{\theta \in \Theta} KL(f_0, f_\theta).  
        ]
        
    - (\theta_0) minimises KL divergence; it is the **best approximation** within the parametric family.
        
- **Link to expected log-likelihood**
    
    - Note:  
        [  
        KL(f_0, f_\theta)  
        = E_{f_0}[\log f_0(Y\mid X)] - E_{f_0}[\log f(Y\mid X; \theta)].  
        ]
        
    - The first term does not depend on (\theta). Therefore:  
        [  
        \theta_0  
        = \arg\max_{\theta \in \Theta} E_{f_0}[\log f(Y\mid X; \theta)].  
        ]
        
    - In words: **the pseudo-true parameter maximises the expected log-likelihood under the true model**.
        
- **Why MLE?**
    
    - By the law of large numbers:  
        [  
        \frac{1}{n}\ell(\theta)  
        = \frac{1}{n}\sum_{i=1}^n \log f(y_i \mid x_i; \theta)  
        \xrightarrow{p} E_{f_0}[\log f(Y\mid X; \theta)].  
        ]
        
    - Maximising sample log-likelihood approximates maximising the expected log-likelihood.
        
    - Therefore, MLE targets the **KL-optimal** parameter (\theta_0).
        
    - If the model is correctly specified, (\theta_0) coincides with the true parameter; otherwise, it is the best approximation.
        

---

### 8.5 Asymptotic properties of MLE

Under standard regularity conditions (as in Wooldridge Ch. 13):

1. **Consistency**
    
    - Theorem (Consistency of CMLE):  
        If:
        
        - The model is identified (unique maximiser (\theta_0) of the expected log-likelihood),
            
        - The parameter space (\Theta) is compact / suitably defined,
            
        - The log-likelihood satisfies technical measurability, continuity, and dominance conditions,  
            then:  
            [  
            \hat{\theta} \xrightarrow{p} \theta_0.  
            ]
            
    - If the model is correctly specified, (\theta_0) is the true parameter → **consistent estimator of the truth**.
        
2. **Asymptotic normality**
    
    - Score has mean zero at (\theta_0) and variance equal to the information matrix.
        
    - Define:
        
        - **Expected (Fisher) information matrix**:  
            [  
            I(\theta_0) = - E\left[  
            \frac{\partial^2}{\partial \theta \partial \theta'}  
            \log f(Y\mid X; \theta_0)  
            \right].  
            ]
            
    - Under regularity conditions:  
        [  
        \sqrt{n}(\hat{\theta} - \theta_0)  
        \xrightarrow{d} N(0, I(\theta_0)^{-1}).  
        ]
        
    - This yields **asymptotic standard errors**:
        
        - Estimate (I(\theta_0)) by either:
            
            - The negative average Hessian (observed information), or
                
            - The outer product of scores.
                
        - In practice, software provides a robustified version (sandwich estimator) in many models.
            
3. **Asymptotic efficiency**
    
    - Among a broad class of consistent, asymptotically normal estimators based on the same model:
        
        - MLE attains the **Cramér–Rao lower bound**, i.e. has the smallest asymptotic variance (under correct specification).
            
    - This generalises the Gauss–Markov idea:
        
        - LS is most efficient in the linear–normal setup;
            
        - MLE is most efficient in the full parametric family.
            

---

### 8.6 ML-based tests: LR, score, and Wald (overview)

In the ML block, inference is framed in terms of three asymptotically equivalent test types:

- **Likelihood-Ratio (LR) test**
    
    - Compare unrestricted model (parameter vector (\hat{\theta}_U)) to restricted model (parameter vector (\hat{\theta}_R)):  
        [  
        LR = 2[\ell(\hat{\theta}_U) - \ell(\hat{\theta}_R)].  
        ]
        
    - Under the null (restrictions true), (LR \xrightarrow{d} \chi^2_q), where (q) is the number of restrictions.
        
- **Lagrange Multiplier (LM) / Score test**
    
    - Only estimate the **restricted** model.
        
    - Use the **score** of the unrestricted model evaluated at (\hat{\theta}_R), together with an information matrix estimate.
        
    - Asymptotically (\chi^2_q) under the null.
        
- **Wald test**
    
    - Estimate the **unrestricted** model.
        
    - Let restrictions be written as (c(\theta) = 0) (linear or non-linear).
        
    - Construct:  
        [  
        W = c(\hat{\theta}_U)'  
        [\widehat{\text{Var}}(c(\hat{\theta}_U))]^{-1}  
        c(\hat{\theta}_U)  
        \xrightarrow{d} \chi^2_q.  
        ]
        
    - For linear restrictions (R\theta = r), this simplifies to the familiar quadratic form.
        
- **Key exam message**
    
    - All three are **asymptotically equivalent**:  
        [  
        LR \approx LM \approx W \quad (\text{large } n).  
        ]
        
    - Packages typically default to Wald tests; LR tests show up via log-likelihood comparisons.
        

---

### 8.7 Pseudo-R² in ML settings (McFadden’s measure)

For non-linear ML models (particularly logit/probit), the usual (R^2) based on sums of squares is not meaningful. The slides introduce **McFadden’s pseudo-R²**:

- **Definition (McFadden, 1974)**
    
    - Let:
        
        - (\ell(\hat{\theta})): log-likelihood of the estimated model.
            
        - (\ell(\hat{\theta}_0)): log-likelihood of a **null** model with only an intercept (no covariates).
            
    - McFadden’s pseudo-R²:  
        [  
        R^2_{McF} = 1 - \frac{\ell(\hat{\theta})}{\ell(\hat{\theta}_0)}.  
        ]
        
- **Rationale**
    
    - Since (\ell(\hat{\theta}) \le \ell(\hat{\theta}_0)) (because the null model is nested and the full model fits at least as well), the ratio (\ell(\hat{\theta}) / \ell(\hat{\theta}_0)) is between 0 and 1.
        
    - Then:
        
        - If covariates have **no explanatory power**, (\ell(\hat{\theta}) \approx \ell(\hat{\theta}_0)) →  
            [  
            R^2_{McF} \approx 0.  
            ]
            
        - If the model fits much better than the intercept-only model, (\ell(\hat{\theta})) is much larger (less negative), so the ratio is small and (R^2_{McF}) approaches 1.
            
- **Interpretation**
    
    - Measures the **relative improvement** in log-likelihood compared to an intercept-only model.
        
    - Not directly comparable to classical (R^2), but:
        
        - (R^2_{McF} = 0) corresponds to “no gain over intercept-only”, analogous to (R^2=0).
            
        - Higher pseudo-R² indicates better fit, but typical values are much lower than OLS (R^2) (e.g. 0.2–0.4 can already indicate reasonable fit in discrete choice models).
            
- **Usage**
    
    - Reported alongside log-likelihood and information criteria (AIC/BIC).
        
    - Common in logit/probit, count models, and other ML-estimated discrete outcome models.
        

---

### 8.8 What you need to master under “MLE” in this course

Under **8. Maximum Likelihood Estimation (MLE)** you should be able to:

- Specify a **conditional density/mass** (f(y\mid x;\theta)) for cross-section data and write down the corresponding (log-)likelihood.
    
- Define the **MLE** as the maximiser of the sample log-likelihood and understand the role of **score** and **Hessian** in computing it.
    
- Demonstrate that OLS is the **ML estimator** of (\beta) in the normal linear regression model.
    
- Explain the **Kullback–Leibler** motivation:
    
    - Define the KL divergence and the pseudo-true parameter as the maximiser of expected log-likelihood.
        
    - Link MLE consistency to convergence of average log-likelihood to its expectation.
        
- State the **asymptotic properties** of MLE:
    
    - Consistency.
        
    - Asymptotic normality and information matrix.
        
    - Asymptotic efficiency.
        
- Recognise and outline **ML-based tests** (LR, LM/score, Wald) and their roles.
    
- Define and interpret **pseudo-R² (McFadden)** as a measure of fit in ML settings, especially for binary choice models.

**9. Limited Dependent Variable Models (Binary Choice) – Detailed Breakdown**

### 9.1 Limited and binary dependent variables

- **Limited dependent variable (LDV)**
    
    - Dependent variable’s support is restricted, not ((-\infty,\infty)).
        
    - Examples:
        
        - Binary outcome (y \in {0,1}) (labour force participation, default, adoption).
            
        - Proportions/percentages in ([0,1]) or ([0,100]).
            
        - Counts (0,1,2,…) such as number of arrests.
            
- **Binary dependent variable / binary choice**
    
    - Focus of this block: (y \in {0,1}), where:
        
        - (y=1) = “success” (event occurs),
            
        - (y=0) = “failure” (event does not occur).
            
    - Main object of interest is the **conditional response probability**:  
        [  
        p(x) = P(y=1\mid x).  
        ]
        

---

### 9.2 Linear Probability Model (LPM)

#### 9.2.1 Specification

- Standard linear regression applied to a binary outcome:  
    [  
    y_i = \beta_0 + \beta_1 x_{1i} + \dots + \beta_k x_{ki} + u_i,\quad y_i\in{0,1}.  
    ]
    
- Conditional mean:  
    [  
    E(y_i\mid x_i) = P(y_i=1\mid x_i) = p(x_i) = \beta_0 + x_i'\beta.  
    ]
    
- Interpretation of coefficients:
    
    - For a **continuous** regressor (x_j): (\beta_j) is the change in the probability of success when (x_j) increases by one unit, ceteris paribus.
        
    - For a **binary** regressor (x_j): (\beta_j) approximates the change in success probability when (x_j) switches from 0 to 1, holding other regressors fixed.
        

#### 9.2.2 Estimation and basic interpretation

- Estimation via OLS exactly as in standard multiple regression.
    
- Example (MROZ labour-force participation):
    
    - (y) = in labour force (inlf = 1) vs not.
        
    - Regressors: husband’s income (nwifeinc), education (educ), experience (exper), age, kidslt6, kidsge6.
        
    - Coefficient on education (e.g. (\hat{\beta}_{educ}=0.038)):
        
        - 10 more years of education → predicted probability of being in the labour force increases by (0.038\times 10 = 0.38) (38 percentage points), holding other variables fixed.
            

---

### 9.3 Issues with the LPM

#### 9.3.1 Predicted probabilities outside ([0,1])

- Because (p(x)=\beta_0 + x'\beta) is linear, it can be:
    
    - Negative for some (x),
        
    - Greater than 1 for other (x).
        
- In MROZ example:
    
    - For a specific combination of covariates, predicted probability is negative for low education and >1 for some values.
        
    - If such combinations are far outside the observed support of covariates, practical concern may be limited; but in principle this is a specification problem.
        

#### 9.3.2 Heteroskedasticity

- For binary (y),  
    [  
    \text{Var}(y\mid x)  
    = p(x)[1-p(x)].  
    ]
    
- In the LPM, (p(x) = \beta_0 + x'\beta) → variance **depends on (x)**:
    
    - Heteroskedasticity is inherent; MLR.5 (homoskedasticity) is violated.
        
- Consequences:
    
    - OLS (\hat{\beta}) is still **unbiased and consistent** (if exogeneity holds).
        
    - Usual homoskedastic variance formula is invalid; t and F based on it are not valid.
        
    - Must use **heteroskedasticity-robust standard errors** or WLS/FGLS for valid inference.
        
    - Empirical work suggests homoskedastic t/F are often not dramatically wrong, but formally they are not justified.
        

#### 9.3.3 Goodness-of-fit in LPM

- **Percent correctly predicted**:
    
    - Define predicted probability (\hat{p}_i = \hat{y}_i).
        
    - Classification rule:
        
        - (\tilde{y}_i = 1) if (\hat{p}_i \ge 0.5),
            
        - (\tilde{y}_i = 0) if (\hat{p}_i < 0.5).
            
    - Goodness-of-fit measure = share of observations where (\tilde{y}_i = y_i).
        
    - Same classification logic is later used for logit/probit.
        

#### 9.3.4 Overall role of the LPM

- Pros:
    
    - Extremely simple to estimate and interpret.
        
    - Direct “probability change in percentage points” interpretation.
        
    - Often provides a reasonable estimate of **average partial effects**.
        
- Cons:
    
    - No built-in restriction to ([0,1]).
        
    - Built-in heteroskedasticity; must correct inference.
        
- Optional reading referenced: Battey et al. (2019) for theoretical justification of LPM as an approximation for binary data.
    

---

### 9.4 Logit and probit models

#### 9.4.1 General binary response specification

- Replace linear probability function with a nonlinear CDF (G) mapping (\mathbb{R}\to(0,1)):  
    [  
    p(x) = P(y=1\mid x) = G(\beta_0 + x'\beta).  
    ]
    
- Requirements for (G):
    
    - Strictly between 0 and 1: (0 < G(z) < 1) for all (z).
        
    - Monotone increasing in (z).
        

#### 9.4.2 Logit model

- Use logistic CDF:  
    [  
    G(z) = \Lambda(z) = \frac{\exp(z)}{1 + \exp(z)}.  
    ]
    
- Properties:
    
    - (G(z)\to 0) as (z\to -\infty), (G(z)\to 1) as (z\to +\infty).
        
    - Symmetric around 0: (\Lambda(z) = 1 - \Lambda(-z)).
        
    - Slope (density) function:  
        [  
        g(z) = \Lambda(z)[1-\Lambda(z)].  
        ]
        

#### 9.4.3 Probit model

- Use standard normal CDF:  
    [  
    G(z) = \Phi(z) = \int_{-\infty}^z \phi(\nu), d\nu,  
    ]  
    where (\phi(\nu) = (2\pi)^{-1/2}\exp(-\nu^2/2)).
    
- Properties:
    
    - (\Phi(z)\to 0) as (z\to -\infty), (\Phi(z)\to 1) as (z\to +\infty).
        
    - Symmetric: (\Phi(z) = 1 - \Phi(-z)).
        
    - Slope (density) function:  
        [  
        g(z) = \phi(z).  
        ]
        
- Both logit and probit fix LPM’s main issues:
    
    - Predicted probabilities are automatically in ((0,1)).
        
    - Model explicitly accounts for heteroskedasticity of binary outcomes via the Bernoulli likelihood.
        

---

### 9.5 Latent-variable interpretation

- Assume an unobserved latent index (y^_):  
    [  
    y_i^_ = \beta_0 + x_i'\beta + e_i,  
    ]  
    and define observed binary outcome:  
    [  
    y_i = 1[y_i^* > 0] =  
    \begin{cases}  
    1 & \text{if } y_i^*>0,\  
    0 & \text{otherwise}.  
    \end{cases}  
    ]
    
- Error term (e_i) is independent of (x_i) and:
    
    - Logistic distribution → logit model.
        
    - Standard normal distribution → probit model.
        
- Then:
    
    - (P(y_i=1\mid x_i) = P(e_i > -(\beta_0 + x_i'\beta)) = G(\beta_0 + x_i'\beta)).
        
    - Conditional mean of latent variable: (E[y^*\mid x] = \beta_0 + x'\beta).
        
- Implications:
    
    - Signs of (\beta_j) give the direction of effect on both:
        
        - Latent index (E[y^*\mid x]), and
            
        - Success probability (P(y=1\mid x)).
            
    - But magnitudes of (\beta_j) do not directly represent probability changes.
        

---

### 9.6 Marginal/partial effects and partial relationships

#### 9.6.1 Continuous regressors

- Partial effect on probability:  
    [  
    \frac{\partial p(x)}{\partial x_j}  
    = \frac{\partial}{\partial x_j} G(\beta_0 + x'\beta)  
    = g(\beta_0 + x'\beta),\beta_j,  
    ]  
    where (g(z) = dG(z)/dz) is the density function.
    
- Key properties:
    
    - (g(z) > 0) for all (z) in logit and probit.
        
    - Therefore, the partial effect has the **same sign** as (\beta_j).
        
    - In LPM, marginal effect is just (\beta_j) (constant in (x)).
        
    - In logit/probit, the magnitude of the marginal effect depends on:
        
        - (\beta_j) (slope parameter), and
            
        - (g(\beta_0 + x'\beta)) (location of (x) in the distribution).
            
- Typical values of (g(0)):
    
    - Probit: (g(0)=\phi(0)=0.4).
        
    - Logit: (g(0)=\Lambda(0)[1-\Lambda(0)]=0.25).
        
    - Marginal effects are largest when the index (\beta_0 + x'\beta) is near 0, i.e. when (p(x)) is around 0.5.
        
- Relative effects:
    
    - For regressors (x_j) and (x_h):  
        [  
        \frac{\partial p(x)/\partial x_j}{\partial p(x)/\partial x_h}  
        = \frac{\beta_j}{\beta_h},  
        ]  
        because (g(\beta_0 + x'\beta)) cancels.
        
    - So **ratios of marginal effects** are determined by ratios of (\beta)-coefficients.
        

#### 9.6.2 Discrete/binary regressors

- For binary (x_j) switching from 0 to 1, marginal effect is defined as discrete change:  
    [  
    \Delta p(x)  
    = G(\beta_0 + \beta_j + x_{-j}'\beta_{-j})
    
    - G(\beta_0 + x_{-j}'\beta_{-j}).  
        ]
        
- More generally, if (x_k) changes from (c) to (c+1) at sample mean values of other covariates:  
    [  
    \Delta p(\bar{x})  
    = G(\hat{\beta}_0 + \hat{\beta}_1 \bar{x}_1 + \dots + \hat{\beta}_{k-1}\bar{x}_{k-1}+ \hat{\beta}_k(c+1))
    
    - G(\hat{\beta}_0 + \hat{\beta}_1 \bar{x}_1 + \dots + \hat{\beta}_{k-1}\bar{x}_{k-1}+ \hat{\beta}_k c).  
        ]
        
- Analogous discrete changes can be computed at any combination of covariate values or averaged across the sample.
    

#### 9.6.3 Evaluation of marginal effects

- Because marginal effects depend on (x), we must specify **where** they are evaluated:
    

1. **At selected, economically interesting x**
    
    - Pick a specific observation or reference profile:
        
        - E.g. set binary covariates to 0 (reference group), continuous covariates to meaningful values (quartiles, median, etc.).
            
2. **Partial effect at the sample mean**
    
    - Plug sample mean (\bar{x}) into derivative:  
        [  
        \frac{\partial \hat{p}(\bar{x})}{\partial x_j}  
        \approx g(\hat{\beta}_0 + \bar{x}'\hat{\beta})\hat{\beta}_j.  
        ]
        
    - This is “effect for the average individual”.
        
3. **Average partial effect (APE)**
    
    - Average across all observations:  
        [  
        \widehat{APE}_j  
        = \frac{1}{n} \sum_{i=1}^n g(\hat{\beta}_0 + x_i'\hat{\beta}),\hat{\beta}_j  
        = \left(\frac{1}{n}\sum_{i=1}^n g(\hat{\beta}_0 + x_i'\hat{\beta})\right)\hat{\beta}_j.  
        ]
        
    - The term in parentheses is a **scale factor** common to all continuous regressors:  
        [  
        \text{scale factor}  
        = \frac{1}{n}\sum_{i=1}^n g(\hat{\beta}_0 + x_i'\hat{\beta}).  
        ]
        
    - APE does not depend on an arbitrary choice of (x) and is often preferred.
        

- Role of **scale factor**:
    
    - Once the scale factor is computed, you can directly map logit/probit coefficients to average partial effects by multiplication.
        
    - This allows comparison of average effects across LPM, logit and probit.
        

---

### 9.7 Estimation of logit and probit

#### 9.7.1 Bernoulli likelihood

- For binary (y_i), conditional density:  
    [  
    f(y_i\mid x_i;\beta)  
    = [G(x_i'\beta)]^{y_i}[1-G(x_i'\beta)]^{1-y_i},  
    \quad y_i\in{0,1}.  
    ]
    
- Log-likelihood contribution of observation (i):  
    [  
    \ell_i(\beta)  
    = y_i \log[G(x_i'\beta)]
    
    - (1-y_i)\log[1-G(x_i'\beta)].  
        ]
        
- Sample log-likelihood:  
    [  
    L(\beta)  
    = \sum_{i=1}^n \ell_i(\beta).  
    ]
    

#### 9.7.2 ML estimator

- MLE:  
    [  
    \hat{\beta} = \arg\max_{\beta} L(\beta).  
    ]
    
- No closed-form solution → numerical optimisation (Newton–Raphson, quasi-Newton, etc.).
    
- Under standard regularity conditions:
    
    - (\hat{\beta}) is **consistent**,
        
    - Asymptotically normal:  
        [  
        \sqrt{n}(\hat{\beta}-\beta_0) \xrightarrow{d} N(0, \Sigma),  
        ]
        
    - Asymptotically efficient within the model class.
        
- Asymptotic variance matrix (Fisher information inverse):  
    [  
    Avar(\hat{\beta})  
    = \left{  
    \sum_{i=1}^n  
    \frac{[g(x_i'\hat{\beta})]^2 x_i x_i'}  
    {G(x_i'\hat{\beta})[1-G(x_i'\hat{\beta})]}  
    \right}^{-1}.  
    ]  
    (software handles the implementation).
    
- Inference:
    
    - Standard errors from the estimated variance matrix.
        
    - Asymptotic t-statistics and confidence intervals constructed as in OLS.
        
    - Joint hypothesis tests use Wald/LR/LM frameworks.
        

---

### 9.8 Model fit measures for binary outcome models

#### 9.8.1 Percent correctly predicted (classification accuracy)

- Same definition as under LPM:
    
    - Classify (\tilde{y}_i = 1) if estimated (p_i \ge 0.5); 0 otherwise.
        
    - Compute fraction of observations where (\tilde{y}_i = y_i).
        
- Can be misleading:
    
    - If outcome is heavily imbalanced (e.g. 95% zeros), always predicting 0 yields 95% “accuracy” but zero predictive content on ones.
        
    - Improved versions relate the classification performance to the share of successes in the sample.
        

#### 9.8.2 Pseudo R-squared (McFadden)

- Defined as:  
    [  
    R^2_{\text{McF}}  
    = 1 - \frac{L_{ur}}{L_0},  
    ]  
    where:
    
    - (L_{ur}) is log-likelihood of the unrestricted model (with covariates).
        
    - (L_0) is log-likelihood of a model with only an intercept.
        
- Properties:
    
    - (|L_{ur}| \le |L_0|) and (|L_{ur}|/|L_0|\le 1).
        
    - If covariates have no power, (L_{ur} \approx L_0) → (R^2_{\text{McF}} \approx 0).
        
    - In principle, (R^2_{\text{McF}}=1) if model fits perfectly (all predicted probabilities near 1 when (y=1) and near 0 when (y=0) – practically never).
        
    - Conceptually analogous to OLS (R^2), but **scale and typical values differ** (values like 0.2–0.4 can already be reasonable).
        

#### 9.8.3 What to report in practice

For logit/probit applications in this course you are expected to report at least:

- Estimated coefficients and robust standard errors.
    
- t-statistics / p-values for significance of coefficients.
    
- Percent correctly predicted.
    
- Log-likelihood value.
    
- Pseudo (R^2) (McFadden).
    
- Selected marginal/partial effects (at selected (x) and/or APEs), especially for economically key variables.
    

---

### 9.9 Core learning targets under “Binary Choice”

Under **9. Limited Dependent Variable Models (Binary Choice)** you should be able to:

- Define limited and binary dependent variables and the object (P(y=1\mid x)).
    
- Specify and interpret the **LPM**, including:
    
    - Coefficients as probability changes,
        
    - Predictions outside ([0,1]),
        
    - Built-in heteroskedasticity and need for robust SE.
        
- Specify **logit** and **probit** models, including:
    
    - Functional forms for (G(z)) and (g(z)),
        
    - Latent-variable interpretation (y^* = \beta_0 + x'\beta + e), (y=1[y^*>0]).
        
- Derive and interpret **marginal/partial effects** for continuous and binary regressors, and understand:
    
    - Why sign of marginal effect equals sign of (\beta_j),
        
    - Evaluation at specific (x), at the mean, and **APE**,
        
    - Role of scale factor in mapping coefficients to average probability effects.
        
- Write down and explain the **Bernoulli likelihood**, log-likelihood and MLE for logit/probit, including asymptotic properties.
    
- Use and interpret **model fit measures**: percent correctly predicted and McFadden pseudo-R², and understand their limitations.