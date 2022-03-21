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

# set a reproducible seed
set.seed(10)

# USER CHOICES -------------------------------------

# Set which outcome you're interested in
# pick one of "seasonal_vaccine" or "h1n1_vaccine"
outcome_name <- "seasonal_vaccine"

# DATA READING AND CLEANING ------------------------

features <- fread("training_set_features.csv")
labels <- fread("training_set_labels.csv")

dat <- merge(features, labels, by="respondent_id")

# Delete columns that shouldn't be used as predictors
dat[, hhs_geo_region := NULL]
dat[, employment_industry := NULL]
dat[, employment_occupation := NULL]
dat[, respondent_id := NULL]

# Create outcome variable and delete the other one
# so we don't accidentally use it
dat[, outcome := get(outcome_name)]
dat[, seasonal_vaccine := NULL]
dat[, h1n1_vaccine := NULL]

# Re-code variables as factors
dat[, age_group := as.factor(age_group)]
dat[, education := as.factor(education)]
dat[, race := as.factor(race)]
dat[, sex := as.factor(sex)]
dat[, income_poverty := as.factor(income_poverty)]
dat[, marital_status := as.factor(marital_status)]
dat[, rent_or_own := as.factor(rent_or_own)]
dat[, employment_status := as.factor(employment_status)]
dat[, census_msa := as.factor(census_msa)]

# Get rid of rows with missing data
dat <- na.omit(dat)

# Split data into training and testing
train.ind <- sample(1:nrow(dat), floor(nrow(dat)/2))
dat.train <- dat[train.ind,]
dat.test <- dat[-train.ind,]

# MODEL FITTING ------------------------------------

# Function to plot an out of sample ROC curve
make.roc <- function(PREDS, ...){
  prediction.obj <- prediction(PREDS, dat.test$outcome)

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
  mean(round(preds) != dat.test$outcome)
}

# Function to create a calibration plot
calibrate <- function(preds, ...){
  # Use a degree 3 spline, higher degree sometimes breaks
  calibrate.plot(dat.test$outcome, preds,
                 main="Calibration Plot", font.main=1,
                 df=3)
}

make.plots <- function(preds, ...){
  par(mfrow=c(1, 2))
  make.roc(preds, ...)
  calibrate(preds)
}

# (0) LOGISTIC REGRESSION
# -----------------------
fit.lr <- glm(outcome ~ ., data=dat.train,
              family=binomial(link="logit"))
pred.lr <- predict(fit.lr, newdata=dat.test, type="response")

make.plots(pred.lr, main="Logistic Regression")
get.misclass.error(pred.lr)

# (1) SINGLE REGRESSION TREE
# --------------------------
fit <- rpart(formula = as.factor(outcome) ~ .,
             data = dat.train,
             control = rpart.control(cp = 0.002))
cpTab <- printcp(fit)
plotcp(fit)

# Get the tree with the minimum error
minErr <- which.min(cpTab[,4])
tree.best <- prune(fit, cp = cpTab[minErr,1])
rpart.plot(tree.best)

# Get a "good" tree w/ minimal complexity (within 1 standard error)
good_inds <- which(cpTab[,4]<cpTab[minErr,4]+1*cpTab[minErr,5])
min_complexity_ind <- good_inds[1]
tree.1se <- prune(fit, cp = cpTab[min_complexity_ind,1])
rpart.plot(tree.1se)

# Make predictions for the test set based on "best" tree
# and minimal complexity, out of sample.
# (extracting the second column b/c it's the probability of being a 1)
preds.1se <- predict(tree.1se, dat.test)[, 2]
preds.best <- predict(tree.best, dat.test)[, 2]

make.plots(preds.1se, main="Min. Complexity Tree")
make.plots(preds.best, main="Best Tree")

get.misclass.error(preds.1se)
get.misclass.error(preds.best)

# (2) RANDOM FOREST
# --------------------------
fit.ranger <- ranger(as.factor(outcome) ~ .,
                     data = dat.train,
                     mtry = 5,
                     probability = TRUE)

preds.ranger <- predict(fit.ranger, dat.test)$predictions[, 2]

make.plots(preds.ranger, main="Random Forest")

get.misclass.error(preds.ranger)

# (3) LASSO
# ---------

# Create covariates matrix (without intercept) and outcome vector
x.train <- model.matrix(outcome ~ 0 + ., data=dat.train)
x.test <- model.matrix(outcome ~ 0 + ., data=dat.test)

y.train <- c(dat.train[, "outcome"])[[1]] %>% as.factor()
y.test <- c(dat.test[, "outcome"])[[1]] %>% as.factor()

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

fit.gbm.1 <- gbm(outcome ~ .,
                 data = dat.train,
                 distribution = "bernoulli",
                 n.trees = 15000,
                 interaction.depth = 1,
                 shrinkage = 0.1,
                 cv.folds = 4)

fit.gbm.2 <- gbm(outcome ~ .,
                 data = dat.train,
                 distribution = "bernoulli",
                 n.trees = 15000,
                 interaction.depth = 2,
                 shrinkage = 0.1,
                 cv.folds = 4)

best.fit.gbm.1 <- gbm.perf(fit.gbm.1, method="cv")
best.fit.gbm.2 <- gbm.perf(fit.gbm.2, method="cv")

preds.gbm.1 <- predict(fit.gbm.1, dat.test,
                       ntree=best.fit.gbm.1, type="response")
preds.gbm.2 <- predict(fit.gbm.2, dat.test,
                       ntree=best.fit.gbm.2, type="response")

make.plots(preds.gbm.1, main="Boosting, Interaction Depth: 1")
make.plots(preds.gbm.2, main="Boosting, Interaction Depth: 2")

get.misclass.error(preds.gbm.1)
get.misclass.error(preds.gbm.2)
