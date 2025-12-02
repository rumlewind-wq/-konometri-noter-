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
