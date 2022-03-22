rm(list=ls())

library(data.table)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ranger)
library(gbm)
library(ROCR)
library(magrittr)
library(glmnet)
library(splines)

# set a reproducible seed
set.seed(10)

# DATA READING AND CLEANING ------------------------
dat <- read.table("data/infarcts.txt", header=T) %>% data.table

# Select variables to use
dat <- dat[, .(infarcts, age, educ, income, weight, height,
               packyrs, alcoh, chd, claud, htn, diabetes, ldl, crt)]
dat[, chd := as.factor(chd)]
dat[, claud := as.factor(claud)]
dat[, htn := as.factor(htn)]
dat[, diabetes := as.factor(diabetes)]

# Get rid of rows with missing data
dat <- na.omit(dat)

# Split data into training and testing
train.ind <- sample(1:nrow(dat), floor(nrow(dat)/2))
dat.train <- dat[train.ind,]
dat.test <- dat[-train.ind,]

# MODEL FITTING ------------------------------------

# Function to plot an out of sample ROC curve
make.roc <- function(PREDS, ...){
  prediction.obj <- prediction(PREDS, dat.test$infarcts)

  roc.obj <- performance(prediction.obj, "tpr", "fpr")
  auc.obj <- performance(prediction.obj, measure = "auc")

  auc <- auc.obj@y.values[[1]]

  plot(roc.obj, font.main=1, ...)
  text(x=0.6, y=0.1, paste0("AUC: ", round(auc, 2)))
  abline(a=0, b=1, lty=2)
}

# Function to get mis-classification error
# Rounds probability predictions up if > 0.5
get.misclass.error <- function(preds){
  mean(round(preds) != dat.test$infarcts)
}

# Function to create a calibration plot
calibrate <- function(preds, ...){
  # Use a degree 3 spline, higher degree sometimes breaks
  calibrate.plot(dat.test$infarcts, preds,
                 main="Calibration Plot", font.main=1,
                 df=3)
}

make.plots <- function(preds, ...){
  par(mfrow=c(1, 2))
  make.roc(preds, ...)
  calibrate(preds)
}

# Function to plot marginal effect of polynomial/spline variables
plot.spline <- function(fit, col, knots=NULL){
  cols <- colnames(dat)[!colnames(dat) %in% c(col, "infarcts",
                                              "chd", "claud", "htn", "diabetes")]

  xmat <- dat[, lapply(.SD, mean), .SDcols=cols]

  xmat[, chd := factor(0)]
  xmat[, claud := factor(0)]
  xmat[, htn := factor(0)]
  xmat[, diabetes := factor(0)]

  var_seq <- seq(min(dat[[col]]), max(dat[[col]]), by=0.1)

  xmat <- replicate(length(var_seq), xmat, simplify = F) %>% rbindlist
  xmat <- cbind(xmat, var_seq)
  setnames(xmat, "var_seq", col)

  preds.marg <- predict(fit, newdata=xmat, type="response")
  plot(preds.marg ~ var_seq, main=paste0("Marginal effect of ", col),
       xlab=col, ylab="Probability", type='l')

  if(!is.null(knots)){
    abline(v=knots, lty=3)
  }
}

# (0) LOGISTIC REGRESSION
# -----------------------
fit.lr <- glm(infarcts ~ ., data=dat.train,
              family=binomial(link="logit"))
pred.lr <- predict(fit.lr, newdata=dat.test, type="response")

make.plots(pred.lr, main="Logistic Regression")
get.misclass.error(pred.lr)

# (1) FEATURE ENGINEERING
# --------------------------

# (a) POLYNOMIALS ----------

# Specify some continuous variables as a polynomial rather than
# a single linear term
fit.poly <- glm(infarcts ~ .-(age)-(height)-(weight) +
                          bs(age) + bs(height) + bs(weight), data=dat.train,
                  family=binomial(link="logit"))
pred.poly <- predict(fit.poly, newdata=dat.test, type="response")

# Plot the marginal relationship between predictor and outcome
plot.spline(fit.poly, "age")
plot.spline(fit.poly, "height")
plot.spline(fit.poly, "weight")

make.plots(pred.poly, main="Logistic Regression")
get.misclass.error(pred.poly)

# (b) SPLINES --------------

# Get internal knots for splines
quantiles <- c(0.25, 0.5, 0.75)
age.knots <- quantile(dat.train$age, quantiles)
height.knots <- quantile(dat.train$height, quantiles)
weight.knots <- quantile(dat.train$weight, quantiles)

# Fit logistic regression with spline parameterizations, degree 3,
# with the knots specified above, for age, height, and weight
fit.spline <- glm(infarcts ~ .-(age)-(height)-(weight) +
                          bs(age, knots=age.knots) +
                          bs(height, knots=height.knots) +
                          bs(weight, knots=weight.knots), data=dat.train,
              family=binomial(link="logit"))

pred.spline <- predict(fit.spline, newdata=dat.test, type="response")

# Plot the marginal relationship between predictor and outcome
plot.spline(fit.spline, "age", knots=age.knots)
plot.spline(fit.spline, "height", knots=height.knots)
plot.spline(fit.spline, "weight", knots=weight.knots)

make.plots(pred.spline, main="Logistic Regression")
get.misclass.error(pred.spline)

# (2) RANDOM FOREST
# --------------------------
fit.ranger <- ranger(as.factor(infarcts) ~ .,
                     data = dat.train,
                     mtry = 5,
                     probability = TRUE)

preds.ranger <- predict(fit.ranger, dat.test)$predictions[, 2]

make.plots(preds.ranger, main="Random Forest")

get.misclass.error(preds.ranger)

# (3) LASSO
# ---------

# Create covariates matrix (without intercept) and outcome vector
x.train <- model.matrix(infarcts ~ 0 + ., data=dat.train)
x.test <- model.matrix(infarcts ~ 0 + ., data=dat.test)

y.train <- c(dat.train[, "infarcts"])[[1]] %>% as.factor()
y.test <- c(dat.test[, "infarcts"])[[1]] %>% as.factor()

# Fit a lasso model (alpha = 1 [alpha = 0 corresponds to ridge regression])
fit.lasso <- glmnet(x=x.train, y=y.train,
                    family="binomial", alpha=1)

# Create a coefficient plot
plot(fit.lasso)

# Perform cross-validation to see what the best lambda was
# and plot mean squared error
cv.lasso <- cv.glmnet(x=x.train, y=as.numeric(y.train),
                      alpha=1, nfolds=10, type.measure="deviance")
plot(cv.lasso)

# Get the best lambda, and see the values for the coefficients
# plus which ones were dropped out of the model (denoted with ".")
lambda.min <- cv.lasso$lambda.min
predict(fit.lasso, type="coefficients", s=lambda.min)

# Compare to a larger lambda within 1 SE,
# this is a sparser model
lambda.1se <- cv.lasso$lambda.1se
predict(fit.lasso, type="coefficients", s=lambda.1se)

# Get predictions with both of the lambda values
preds.lasso.best <- predict(fit.lasso, newx=x.test, type="response", s=lambda.min)
preds.lasso.1se <- predict(fit.lasso, newx=x.test, type="response", s=lambda.1se)

make.plots(preds.lasso.best, main="Best Lambda")
make.plots(preds.lasso.1se, main="Sparser Model")

get.misclass.error(preds.lasso.best)
get.misclass.error(preds.lasso.1se)

# (4) BOOSTING
# ------------

fit.gbm.2 <- gbm(infarcts ~ .,
                 data = dat.train,
                 distribution = "bernoulli",
                 n.trees = 20000,
                 interaction.depth = 2,
                 shrinkage = 0.001,
                 cv.folds = 4)

fit.gbm.3 <- gbm(infarcts ~ .,
                 data = dat.train,
                 distribution = "bernoulli",
                 n.trees = 20000,
                 interaction.depth = 3,
                 shrinkage = 0.001,
                 cv.folds = 4)

best.fit.gbm.2 <- gbm.perf(fit.gbm.2, method="cv")
best.fit.gbm.3 <- gbm.perf(fit.gbm.3, method="cv")

preds.gbm.2 <- predict(fit.gbm.2, dat.test,
                       ntree=best.fit.gbm.2, type="response")
preds.gbm.3 <- predict(fit.gbm.3, dat.test,
                       ntree=best.fit.gbm.3, type="response")

make.plots(preds.gbm.2, main="Boosting, Interaction Depth: 2")
make.plots(preds.gbm.3, main="Boosting, Interaction Depth: 3")

get.misclass.error(preds.gbm.2)
get.misclass.error(preds.gbm.3)
