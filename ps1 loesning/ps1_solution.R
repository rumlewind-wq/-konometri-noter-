# R Solutions to Problem Set 1

rm(list=ls())

# load data
library(haven)
discrim <- read_dta("Opgaver datasæt/P1/ps1 data (1)/discrim.dta")

#a) 
reg1 <- lm(lpsoda ~ prpblck + lincome + prppov, data=discrim)
summary(reg1)

library(car)
linearHypothesis(reg1, c("prpblck =0"))

#b)
cor(discrim$lincome, discrim$prppov, use="na.or.complete")

reg2 <- lm(lpsoda ~ prpblck + lincome + prppov, data=discrim)
summary(reg2)

#c)
reg3 <- lm(lpsoda ~ prpblck + lincome + prppov + lhseval, data=discrim)
summary(reg3)

#d)
reg4 <- lm(lpsoda ~ prpblck + lincome + prppov + lhseval, data=discrim)
summary(reg4)

urss <- sum(resid(reg4)^2) # sum of squared residuals of the Unrestricted model
# alternative 1: urss <- deviance(reg4)
# alternative 2: urss <- anova(reg4)["Residuals", "Sum Sq"]

reg5 <- lm(lpsoda ~ prpblck + lhseval, data=discrim)
summary(reg5)

rrss <- sum(resid(reg5)^2) # sum of squared residuals of the Restricted model
df<- reg4$df.residual # degrees of freedom of the unrestricted model = n - k_unres - 1

# compute the F statistic
F_stat <- ((rrss - urss)/2)/((urss)/(df))
F_stat

# p-value for F-test
p_value <- pf(F_stat,2,df,lower.tail = F)
p_value

#Note that the code above perform the same test as the following built-in command:
linearHypothesis(reg4, c("lincome=prppov","prppov=0"))


rm(list=ls())

###Solution to Problem 2

kielmc <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2025/Problem Sets/PS1/KIELMC.DTA")

#a)
reg1 <- lm(lprice ~ ldist, data=subset(kielmc, y81==1))
summary(reg1)

#b)
reg2 <- lm(lprice ~ ldist + lintst + larea + lland + rooms + baths + age, data=subset(kielmc, y81==1))
summary(reg2)

#c)
reg3 <- lm(lprice ~ ldist + lintst + larea + lland + rooms + baths + age + lintstsq, data=subset(kielmc, y81==1))
summary(reg3)

#d)
kielmc$ldistsq<-kielmc$ldist^2

reg4 <- lm(lprice ~ ldist + lintst + larea + lland + rooms + baths + age + lintstsq +ldistsq, data=subset(kielmc, y81==1))
summary(reg4)

rm(list=ls())


###Solution to Problem 3

vote1 <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2025/Problem Sets/PS1/VOTE1.DTA")

#c)
reg1 <- lm(voteA ~ lexpendA + lexpendB + prtystrA, data=vote1)
summary(reg1)

#d)

vote1$d_expend<-vote1$lexpendB-vote1$lexpendA

reg2 <- lm(voteA ~ lexpendA + d_expend + prtystrA, data=vote1)
summary(reg2)

linearHypothesis(reg2, c("lexpendA =0"))

rm(list=ls())

