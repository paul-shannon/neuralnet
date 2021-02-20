# http://sungsoo.github.io/2018/04/05/building-a-neural-network-using-the-iris-dataset.html
# https://github.com/bips-hb/neuralnet
library(RUnit)
library(neuralnet)
library(ggplot2)
library(dplyr)
library(reshape2)
library(caret)  # Classification And REgression Training), provides featurePlot
#------------------------------------------------------------------------------------------------------------------------
runTests <- function()
{
   test_predictSetosa()
   test_predictAllSpecies()

} # runTests
#------------------------------------------------------------------------------------------------------------------------
# a very simple first use.   the setosa species is easily identified by petal width, petal length.
test_predictSetosa <- function()
{
    message(sprintf("--- test_predictSetosa"))

       #---------------------------------------------------------------------------------------
       # this violin plot shows that setosa, in red, separates nicely by petal length and width
       #---------------------------------------------------------------------------------------

    iris.melt <- melt(iris)
    dim(iris)        # 150 5
    dim(iris.melt)   # 600 3

    ggplot(iris.melt, aes(x = factor(variable), y = value)) +
    geom_violin() +
    geom_jitter(height = 0, width = 0.1, aes(colour = Species), alpha = 0.7) +
    theme_minimal()

     # this multivariate plot shows the same separation, though species
     # are unlabeled - not sure how to add them

    caret::featurePlot(x=iris[, c("Petal.Length", "Petal.Width")], y=iris[, "Species"], plot="ellipse")


     #----------------------------------------
     # create two subsets, train and test
     #----------------------------------------

   set.seed(37)
   indices.train <- sort(sample(seq_len(nrow(iris)), 50))
   indices.test <- sort(setdiff(seq_len(nrow(iris)), indices.train))
   checkEquals(length(unique(c(indices.train, indices.test))), nrow(iris))

     #-----------------------------------------------------------------------
     # build the net with tbl.train, just 1 hidden layer (the default) which
     # is sufficient for identifying yes/no the one species, setosa
     #----------------------------------------------------------------------

   tbl.train <- iris[indices.train,]
   tbl.test <- iris[indices.test,]


   nn <- neuralnet(Species == "setosa" ~ Petal.Length + Petal.Width, tbl.train, linear.output = FALSE)

   lapply(nn, class)
       # $call: [1] "call"
       # $response: [1] "matrix" "array"     # the Species=="setosa" rows
       # $covariate: [1] "matrix" "array"    # Petal.length, .width for each of the 150 samples
       # $model.list: [1] "list"             # the function, re-expressed as response and variables
       # $err.fct: [1] "function"            # error function
       # $act.fct: [1] "function"            # activation function
       # $linear.output: [1] "logical"       # FALSE
       # $data: [1] "data.frame"             # the original iris data.frame 150 5
       # $exclude: [1] "NULL"
       # $net.result: [1] "list"             # dim(nn$net.result[[1]])  150,1:  50==1, 100==0
       # $weights: [1] "list"                # fitted for every repetition
       # $generalized.weights: [1] "list"
       # $startweights: [1] "list"
       # $result.matrix: [1] "matrix" "array"


   plot(nn)
     # uses these values:, from nn$result.matrix
     #  error                             0.012204363
     #  reached.threshold                 0.008586344
     #  steps                            86.000000000
     #  Intercept.to.1layhid1            -6.795639187
     #  Petal.Length.to.1layhid1          0.826751088
     #  Petal.Width.to.1layhid1           6.580317190
     #  Intercept.to.Species == "setosa"  3.750704136
     #  1layhid1.to.Species == "setosa"  -7.929457297

     # not clear how to manually apply these values, but the predict function
     # knows how.   here's the first entry in the training set, which
     # of course it gets right:
     #      Sepal.Length Sepal.Width Petal.Length Petal.Width Species
     #    1          5.1         3.5          1.4         0.2  setosa
    checkTrue(predict(nn, tbl.train[1,])[1] > 0.95)
       # all of iris species != "setosa" should have low scores.
       # check in the full dataset, and (more conventionally) in the held-out test set
    checkTrue(max(as.numeric(predict(nn, subset(iris, Species != "setosa")))) < 0.3)
    checkTrue(min(as.numeric(predict(nn, subset(iris, Species == "setosa")))) > 0.85)
    checkTrue(max(as.numeric(predict(nn, subset(tbl.test, Species != "setosa")))) < 0.3)
    checkTrue(min(as.numeric(predict(nn, subset(tbl.test, Species == "setosa")))) > 0.85)

} # test_predictSetosa
#------------------------------------------------------------------------------------------------------------------------
test_predictAllSpecies <- function()
{
    message(sprintf("--- test_predictAllSpecies"))

      # predict species from petal length and width, sepal length and width

   set.seed(17)
   indices.train <- sort(sample(seq_len(nrow(iris)), 50))
   indices.test <- sort(setdiff(seq_len(nrow(iris)), indices.train))
   checkEquals(length(unique(c(indices.train, indices.test))), nrow(iris))

   tbl.train <- iris[indices.train,]
   tbl.test <- iris[indices.test,]

   nn <- neuralnet(Species ~ Petal.Length + Petal.Width + Sepal.Length + Sepal.Width,
                   tbl.train, linear.output = FALSE,
                   hidden=3)
   nn$result.matrix
   checkTrue(nn$result.matrix["error",1][[1]] < 0.006)

       # use the neural net to predict species

   tbl.pred <- as.data.frame(predict(nn, tbl.test))
   tbl.pred <- cbind(tbl.pred, as.character(tbl.test$Species))
   colnames(tbl.pred) <- c("score.setosa", "score.versicolor", "score.virginica", "species")

      # it appears there is a score for each of the outputs - in this case, species as factors
      # where the scores approach 1 for the properly identified species.

   head(subset(tbl.pred, species=="setosa"))
   head(subset(tbl.pred, species=="virginica"))
   head(subset(tbl.pred, species=="versicolor"))


       #------------------------------------------------------------
       # versicolor: how well is it classified?
       # at least 30/33 correct classifications, no more than 3 failures
       #------------------------------------------------------------

   soi <- "versicolor"
   rounded.scores <- round(subset(tbl.pred, species==soi)$score.versicolor, digits=1)
   table(rounded.scores)
   checkEquals(length(rounded.scores), 36)
   checkTrue(length(which(rounded.scores > 0.85)) >= 30)
   checkTrue(length(which(rounded.scores < 0.25)) <= 5)

       # expect none to be classified as setosa
   rounded.scores <- round(subset(tbl.pred, species==soi)$score.setosa, digits=1)
   table(rounded.scores)
   checkEquals(length(rounded.scores), 36)
   checkTrue(length(which(rounded.scores > 0.85)) < 3)
   checkTrue(length(which(rounded.scores < 0.15)) > 33)

       # expect almost none to be classified as virginica
   rounded.scores <- round(subset(tbl.pred, species==soi)$score.virginica, digits=1)
   table(rounded.scores)
   checkEquals(length(rounded.scores), 36)
   checkTrue(length(which(rounded.scores > 0.85)) <= 5)
   checkTrue(length(which(rounded.scores < 0.15)) >= 30)

       #------------------------------------------------------------
       # setosa: how well is it classified?
       # the tests below are slightly forgiving, allowing for
       # some random imprecision which - for setosa - seems not to
       # arise:  this species separates out easily
       #------------------------------------------------------------

   soi <- "setosa"       # 32 correct indentifications, no failures
   rounded.scores <- round(subset(tbl.pred, species==soi)$score.setosa, digits=1)
   table(rounded.scores)
   checkEquals(length(rounded.scores), 28)
   checkTrue(length(which(rounded.scores > 0.85)) >= 26)
   checkTrue(length(which(rounded.scores < 0.15)) <= 2)

   rounded.scores <- round(subset(tbl.pred, species==soi)$score.versicolor, digits=1)
   table(rounded.scores)
   checkEquals(length(rounded.scores), 28)
   checkTrue(length(which(rounded.scores > 0.85)) <= 2)
   checkTrue(length(which(rounded.scores < 0.15)) >= 26)

   rounded.scores <- round(subset(tbl.pred, species==soi)$score.virginica, digits=1)
   table(rounded.scores)
   checkEquals(length(rounded.scores), 28)
   checkTrue(length(which(rounded.scores > 0.85)) <= 2)
   checkTrue(length(which(rounded.scores < 0.15)) >= 26)

       #------------------------------------------------------------
       # virginica: how well is it classified?
       # the tests below are slightly forgiving, allowing for
       # random noise
       #------------------------------------------------------------

   soi <- "virginica"     # 6 failures, 34 correct identifications

   rounded.scores <- round(subset(tbl.pred, species==soi)$score.setosa, digits=1)
   table(rounded.scores)
   checkEquals(length(rounded.scores), 36)
   checkTrue(length(which(rounded.scores > 0.85)) <= 2)
   checkTrue(length(which(rounded.scores < 0.1)) >= 34)

   rounded.scores <- round(subset(tbl.pred, species==soi)$score.versicolor, digits=1)
   table(rounded.scores)
   checkEquals(length(rounded.scores), 36)
   checkTrue(length(which(rounded.scores > 0.85)) >= 4)
   checkTrue(length(which(rounded.scores < 0.25)) >= 25)

   rounded.scores <- round(subset(tbl.pred, species==soi)$score.virginica, digits=1)
   table(rounded.scores)
   checkEquals(length(rounded.scores), 36)
   checkTrue(length(which(rounded.scores > 0.85)) >= 28)
   checkTrue(length(which(rounded.scores < 0.15)) <= 8)


} # test_predictAllSpecies
#------------------------------------------------------------------------------------------------------------------------
