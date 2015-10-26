setwd("~/Kaggle/Churn")

churn <- read.csv("GramenerTraining.csv", header=T)
head(churn)

prop.table(table(churn$Churn))

#         0         1 
# 0.8593926 0.1406074
wald <- ""

install.packages("aod")
library(aod)

wald <- ""
pval <- ""
for (i in 1:17) {
  lm = glm(Churn ~ churn[,i], family = "binomial", data = churn)
  wald[i]=unlist(wald.test(b = coef(lm), Sigma = vcov(lm), Terms = 2))[11]
  pval[i]=unlist(wald.test(b = coef(lm), Sigma = vcov(lm), Terms = 2))[13]
}

# test: lm = glm(Churn ~ churn[,1], family = "binomial", data = churn)


wald <- as.data.frame(t(wald))
pval <- as.data.frame(t(pval))
name <- as.data.frame(t(names(churn[1:17])))


chisq <- t(rbind(name,wald, pval))

write.csv(chisq, file="wald.csv")

#   var	                chisq	      pval
#   Int.l.Plan	        160.2817977	0.0000
#   Day.Mins	          96.21953446	0.0000
#   Day.Charge	        96.2173975	0.0000
#   CustServ.Calls	    93.13967998	0.0000
#   Message.Plan	      22.610702	0.0000
#   Eve.Mins	          21.47064168	0.0000
#   Eve.Charge	        21.46345684	0.0000
#   Messages	          17.84831392	0.0000
#   Intl.Charge	        13.09225153	0.0003
#   Intl.Mins	          13.04033847	0.0003
#   Intl.Calls	        6.677715527	0.0098
#   Night.Charge        1.800514069	0.1797
#   Night.Mins	        1.798338575	0.1799
#   Eve.Calls	          1.22194361	0.2690
#   Day.Calls	          0.498960721	0.4800
#   Night.Calls	        0.471790828	0.4922
#   Account.Length..Weeks.	0.280287015	0.5965

# CORR MATRIX

ot_cor <- cor(churn[1:17], method="pearson")
corrdf <- as.data.frame(ot_cor)

corr_hm <- heatmap(ot_cor, Colv=NA, scale="none", revC=T,
                   symm=T, keep.dendro=F, col=cm.colors(256), verbose=T)

write.csv(corrdf, file="corrdf.csv")

# FEATURE SELECTION FROM CHISQ AND CORR MATRIX

keepvars <- c(  "Int.l.Plan",
                "Day.Mins",
                "CustServ.Calls",
                "Message.Plan",
                "Eve.Mins",
                "Intl.Charge",
                "Intl.Calls",
                "Churn")

churn1 <- churn[, keepvars]

# STATE VARIABLE MANAGEMENT
library(caret)
state <- predict(dummyVars(~ State, data = churn), newdata = churn)

funto <- function(x, y) {
  chisq.test(df[,x],df[,y])$statistic
}

chisq <- outer(names(df),names(df),FUN=Vectorize(funto))
diag(chisq) <- 1  
rownames(chisq) <- names(df)
colnames(chisq) <- names(df)
chisq



churn2 <- as.data.frame(cbind(churn1, state))
churn2$State <- NULL

write.csv(churn2, file="churn2.csv")
churn2 <- read.csv("churn2.csv", header=T)

# ORDINARY GLM

lm <- glm(Churn ~., data = churn2, family = "binomial")
summary(lm)


keepvars1 <- c( "Churn",
                "Int.l.Plan",
                "Day.Mins",
                "CustServ.Calls",
                "Message.Plan",
                "Eve.Mins",
                "Intl.Charge",
                "Intl.Calls",
                "State.CA",
                "State.KS",
                "State.MA",
                "State.MI",
                "State.NJ",
                "State.NV",
                "State.NY",
                "State.SC",
                "State.TX")

churn3 <- churn2[, keepvars1]

# GBM MODEL


train = churn3[1:nrow(churn3)*0.8,]
label = train$Churn
train$Churn <- NULL


# library(gbm)
# model = gbm.fit(x=train, y=label, distribution="bernoulli", 
#                 n.trees=25000, shrinkage=0.005, keep.data=T, 
#                 interaction.depth = 2, n.minobsinnode = 10,
#                 verbose=T, nTrain = round(nrow(train)*0.8))


library(dismo)
m1 = gbm.step(data = churn3, gbm.x = 2:17, gbm.y = 1, family = "bernoulli", 
              learning.rate = 0.005, tree.complexity = 5, bag.fraction = 0.7)

length(m1$fitted) # 2667

m1$nTrain
m1$n.trees
m1$cv.statistics

summary(m1)

# second run
m2 = gbm.step(data = churn3, gbm.x = 2:17, gbm.y = 1, family = "bernoulli",  
              learning.rate = 0.01, tree.complexity = 5, bag.fraction = 0.7)

par(mfcol=c(4,4))
gbm.plot(m2, n.plots=16, write.title=F)
gbm.plot.fits(m2, v=c(2:10))
gbm.plot.fits(m2, v=c(11:16))

summary(m2)

find.int <- gbm.interactions(m2)
find.int$interactions
find.int$rank.list

dev.off()
gbm.perspec(m2, 2, 4, y.range = c(20, 2770), z.range = c(0, 0.6))

#PREDICTION
library(gbm)
churn3$preds <- predict.gbm(m2, churn3, n.trees=m2$gbm.call$best.trees, type="response")
calc.deviance(obs=churn3$Churn, pred=preds, calc.mean=T)
# 0.2333259

d <- cbind(churn3$Churn, churn3$preds)
pres <- d[d[,1]==1, 2]
abs <- d[d[,1]==0, 2]
e <- evaluate(p=pres, a=abs)
e

# class          : ModelEvaluation 
# n presences    : 375 
# n absences     : 2292 
# AUC            : 0.9731821 
# cor            : 0.8858064 
# max TPR+TNR at : 0.1850436 

library(pROC)
auc <- roc(churn3$Churn, churn3$preds)
auc$auc
plot(auc)
# Area under the curve: 0.9732


# LIFT CHARTS
churn_sor <- churn3[order(-churn3$preds),]

plot(churn_sor$preds, xlab="Ranked Customers", ylab="Predicted Probabilities", 
     main="Distribution of Probabilities of Churn")

churn_sor$nonchurn <- 1 - churn_sor$Churn

decile <- function(x){
  deciles <- vector(length=10)
  for (i in seq(0.1,1,.1)){
    deciles[i*10] <- quantile(x, i)
  }
  return (ifelse(x<deciles[1], 1,
                 ifelse(x<deciles[2], 2,
                        ifelse(x<deciles[3], 3,
                               ifelse(x<deciles[4], 4,
                                      ifelse(x<deciles[5], 5,
                                             ifelse(x<deciles[6], 6,
                                                    ifelse(x<deciles[7], 7,
                                                           ifelse(x<deciles[8], 8,
                                                                  ifelse(x<deciles[9], 9, 10))))))))))
}


churn_sor$scrdec <- 11 - decile(churn_sor$preds)

library(plyr)
library(plyr)

groupd <- ddply(churn_sor, c("scrdec"), summarise,
                N = length(Churn),
                sumesc = sum(Churn),
                sumnesc = sum(nonchurn),
                avg_esc_scr = mean(preds)
                
)
groupd

groupd$cum_esc <- as.numeric(formatC(cumsum(groupd$sumesc)))
groupd$cum_nesc <- as.numeric(formatC(cumsum(groupd$sumnesc), format="f"))
groupd$cum_n <- as.numeric(formatC(cumsum(groupd$N)))

tot_esc <- as.numeric(groupd$cum_esc[10])
tot_nesc <- as.numeric(groupd$cum_nesc[10])

groupd$cum_esc_pct <- groupd$cum_esc / tot_esc
groupd$cum_nesc_pct <- groupd$cum_nesc / tot_nesc

groupd$KS <- abs(groupd$cum_esc_pct - groupd$cum_nesc_pct) * 100
groupd
max(groupd$KS)

plot(groupd$scrdec, groupd$cum_esc_pct, xlab="Deciles", ylab="Ratio of Population Captured",
     yaxt="n", main = "Predicted Churn Lift Chart", type ="l", col="red", lwd=2, xlim=c(1,10),ylim=c(0,1))

axis(side=2, at=c(seq(0,1,0.1)))
lines(groupd$cum_nesc_pct, col="green", lwd=2)

# MAKE PREDICTIONS ON TEST DATASET

test <- read.csv("GramenerTesting.csv", header=T)
head(test)

keepvars <- c(  "Int.l.Plan",
                "Day.Mins",
                "CustServ.Calls",
                "Message.Plan",
                "Evening.Mins",
                "Intl.Charge",
                "Intl.Calls"
                 )

test1 <- test[, keepvars]
test1$Eve.Mins <- test1$Evening.Mins
test1$Evening.Mins <- NULL

str(test1)


library(caret)
state <- predict(dummyVars(~ State, data = test), newdata = test)

test2 <- as.data.frame(cbind(test1, state))
test2$State <- NULL

keepvars1 <- c( 
                "Int.l.Plan",
                "Day.Mins",
                "CustServ.Calls",
                "Message.Plan",
                "Eve.Mins",
                "Intl.Charge",
                "Intl.Calls",
                "State.CA",
                "State.KS",
                "State.MA",
                "State.MI",
                "State.NJ",
                "State.NV",
                "State.NY",
                "State.SC",
                "State.TX")

test3 <- test2[, keepvars1]

#PREDICTION
library(gbm)
preds <- predict.gbm(m2, newdata = test3, n.trees=m2$gbm.call$best.trees, type="response")
test3 <- as.data.frame(cbind(test, preds))

test3 <- test3[order(test3$preds), ]
plot(test3$preds)

#Using threshold of 0.5

test3$pred_class <- ifelse(test3$preds > 0.5, 1, 0)
submit <- test3[, c("Area.Code", "Phone", "pred_class")]
head(submit)

prop.table(table(test3$pred_class))
#         0         1 
# 0.8753754 0.1246246 

write.csv(submit, file="submit.csv")
