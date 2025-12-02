# Solutions to Problem set 2

rm(list=ls())


library(car)
library(lmtest)
library(sandwich)
library(haven)

###Solution to Problem 4

# load data
hprice1 <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Problem Sets/PS2/hprice1.dta")


#a) 
reg1<-lm(lprice ~ llotsize + lsqrft + bdrms, data=hprice1)
summary(reg1)

# regression with heteroscedasticity robust standard errors:
# hc0 is "White" Standard errors
coeftest(reg1, vcov=hccm(reg1, type="hc0"))


#b)

hprice1$resid<-reg1$residuals

hprice1$residsq<-hprice1$resid^2

hprice1$llotsizesq<-hprice1$llotsize^2
hprice1$lsqrftsq<-hprice1$lsqrft^2
hprice1$bdrmssq<-hprice1$bdrms^2
hprice1$llotsizelsqrft<-hprice1$llotsize*hprice1$lsqrft
hprice1$llotsizebdrms<-hprice1$llotsize*hprice1$bdrms
hprice1$lsqrftbdrms<-hprice1$lsqrft*hprice1$bdrms

reg_White<-lm(residsq ~ llotsize + lsqrft + bdrms + llotsizesq + lsqrftsq + bdrmssq + llotsizelsqrft + llotsizebdrms + lsqrftbdrms, data=hprice1)
output<-summary(reg_White)
output

F_stat<-output$fstatistic[1]

#The p-value is:
p_value<-pf(F_stat,9,78,lower.tail = F)
p_value


rm(list=ls())

###Solution to Problem 5

library(haven)
smoke <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Problem Sets/PS2/SMOKE.DTA")
attach(smoke)

# (a) see lecture example
summary(lm(cigs~ lincome + lcigpric + educ + age + agesq + restaurn))

# (b) 
reg1 <- lm(cigs~ lincome + lcigpric + educ + age + agesq + restaurn)
luhsq <- log(resid(reg1)^2)
reg2 <- lm(luhsq~ lincome + lcigpric + educ + age + agesq + restaurn)
hatg <- fitted.values(reg2)
hath <- exp(hatg)
# run WLS:
w <- 1/exp(fitted(reg2))
WLS <- lm(cigs~ lincome + lcigpric + educ + age + agesq + restaurn, weight=w)
summary(WLS)
uhat <- resid(WLS)
yhat <- fitted.values(WLS)

# (c)
breveusq <- (uhat/sqrt(hath))^2
brevey <- yhat/sqrt(hath)
breveysq <- brevey^2
reg3 <- (lm(breveusq~ brevey + breveysq))
summary(reg3)
library(car)
H0 <- c("brevey", "breveysq")
linearHypothesis(reg3, H0)

detach(smoke)

