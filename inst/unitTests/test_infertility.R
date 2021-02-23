# from gunther and fritsch,  neuralnet: Training of Neural Networks
#
# This data set contains data of a case-control study that in vestigated infertility after
# spontaneous and induced abortion (Trichopoulos et al., 1976). The data set consists of
#     248 observations
#      83 women, who were infertile (cases)
#     165 women, who were not infertile (controls).
# It includes amongst others the variables age, parity, induced, and spontaneous. The
# variables induced and spontaneous denote the number of prior induced and spontaneous
# abortions, respectively. Both variables take possible values 0, 1, and
# 2 relating to 0, 1, and 2 or more prior abortions. The age in years is given by the
# variable age and the number of births by parity.
#------------------------------------------------------------------------------------------------------------------------
library(RUnit)
library(neuralnet)
library(ggplot2)
library(dplyr)
library(reshape2)
library(caret)  # Classification And Regression Training), provides featurePlot
#------------------------------------------------------------------------------------------------------------------------
runTests <- function()
{
   test_fromPaper()

} # runTests
#------------------------------------------------------------------------------------------------------------------------
# from lm and trena, we see that prior abortions (especially spontaneous) predict fertility somewhat
eda <- function()
{
   infert$eduLevel <- as.integer(infert$education)
   cor(infert$eduLevel, infert$case)
   summary(lm(case ~ eduLevel + parity + age + induced + spontaneous + stratum + pooled.stratum,
              data=infert))
     # Coefficients:
     #                  Estimate Std. Error t value Pr(>|t|)
     # (Intercept)    -0.4319130  0.5930704  -0.728   0.4672
     # eduLevel        0.2253350  0.2574448   0.875   0.3823
     # parity         -0.0835753  0.0495449  -1.687   0.0929 .
     # age             0.0122517  0.0066990   1.829   0.0687 .
     # induced         0.2314131  0.0494210   4.682 4.74e-06 ***
     # spontaneous     0.3884657  0.0468996   8.283 8.48e-15 ***
     # stratum         0.0001898  0.0023980   0.079   0.9370
     # pooled.stratum -0.0105447  0.0074557  -1.414   0.1586
     # ---
     # Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
     #
     # Residual standard error: 0.4229 on 240 degrees of freedom
     # Multiple R-squared:  0.2227,	Adjusted R-squared:    0.2
     # F-statistic: 9.824 on 7 and 240 DF,  p-value: 9.175e-11

   cor(infert$spontaneous, infert$case, method="spearman")   # 0.36
   library(trena)
   target <- "case"
   candidates <- c("induced", "spontaneous", "age", "parity")
   mtx <- as.matrix(infert[, c(target, candidates)])
   rownames(mtx) <- infert$id
   solver <- EnsembleSolver(t(mtx),
                           targetGene="case",
                           candidateRegulators=candidates,
                           solverNames=c("lasso", "Ridge", "Spearman", "Pearson", "RandomForest", "xgboost"))
   tbl.trena <- run(solver)
     # tbl.trena
     #          gene    betaLasso    betaRidge spearmanCoeff pearsonCoeff  rfScore   xgboost
     # 1         age  0.000384421  0.001926305   0.003708937  0.003530451 3.967620 0.3593092
     # 2     induced  0.073214621  0.042642289   0.016577071  0.017112002 1.623818 0.1308087
     # 3      parity -0.040712299 -0.022334094   0.005038094  0.008910762 2.980389 0.2490165
     # 4 spontaneous  0.242211248  0.156262970   0.359094335  0.363997632 7.272539 0.2608656

} # eda
#------------------------------------------------------------------------------------------------------------------------
# a very simple first use.   the setosa species is easily identified by petal width, petal length.
test_fromPaper <- function()
{
   dim(infert)  # 248 8
   head(infert)
      # algorithm defaults to "rprop+" resilient backpropagation with weight backtracking
   nn <- neuralnet(case ~ age+parity+induced+spontaneous,
                   data=infert, hidden=3, err.fct="ce",
                   linear.output=FALSE)
                   #algorithm="backprop")
                   #learningrate=0.01)

   tbl.out <- cbind(nn$covariate, nn$net.result[[1]])
   dimnames(tbl.out) <- list(NULL, c("age","parity","induced", "spontaneous","nn-output"))
   dim(tbl.out)   # 248 5
   head(tbl.out)

   tbl.full <- cbind(infert, nn$net.result[[1]])
   colnames(tbl.full)[11] <- "nn.score"
   head(tbl.full)

   cor(tbl.full$case, tbl.full$nn.score) # hidden=2 cor 0.55
                                         # hidden=3 cor 0.58

      # tbl.pred is the same as nn$net.result[[1]]
      # since there is no division of data into training and test sets
   tbl.pred <- as.data.frame(predict(nn, infert))

} # test_fromPaper
#------------------------------------------------------------------------------------------------------------------------
