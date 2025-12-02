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
