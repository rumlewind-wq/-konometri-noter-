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

