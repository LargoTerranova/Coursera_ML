
Coursera Machine Learning Course Project
Boban D
3 Juli 2018

Loading Data

We load the data directly from the url and partition the trainingset into a training- and test set.

url_train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url_test  <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training        <- read.csv(url(url_train))
testing_final   <- read.csv(url(url_test))

training_partition  <- createDataPartition(training$classe, p=0.7, list=FALSE)
training_set        <- training[training_partition, ]
testing_set         <- training[-training_partition, ]

Cleaning Data

In this step we identify columns with very low variance and remove them. A variable with a low variance (in the extremest case a constant) does not vary much or at all and might contribute very little to an analysis. We identify 55 columns with low variance and remove them.

no_variance_list <- nearZeroVar(training_set)
training_set     <- training_set[, -no_variance_list]
testing_set      <- testing_set[ , -no_variance_list]
testing_final    <- testing_final[, -no_variance_list] 

Calculate missing values per column using purr

Many Machine Learning Algorithms cannot cope with missing data. They either reach wrong results or the clculation is stoped altogether. In this step we identify 59 columns which have missing values in them and remove them.

training_set %>%
    map_df(function(x) sum(is.na(x))) %>%
    gather(feature, num_nulls) %>%
    filter(num_nulls==0) -> zero_value_columns

training_set <- training_set[, zero_value_columns$feature]
testing_set  <- testing_set[ , zero_value_columns$feature]

training_set <- training_set[, -(1:5)]
testing_set  <- testing_set[ , -(1:5)] 

testing_final %>%
    map_df(function(x) sum(is.na(x))) %>%
    gather(feature, num_nulls) %>%
    filter(num_nulls==0) -> zero_value_columns_final

testing_final <- testing_final[, zero_value_columns_final$feature]
testing_final <- testing_final[, -(1:5)]

Model Fitting

Random Forest

We first fit a Random Forest. We instantly achieve a 99.9% Accuracy with our testset as can be seen in the ConfusionMAtrix. Plotting the results confirms the results.

set.seed(246810)
control_model <- trainControl(method = "cv", number = 2, verboseIter = F)
model_rf <- train(classe~., method="rf", data=training_set, trControl=control_model)
model_rf$finalModel

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.22%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3905    0    0    0    1 0.0002560164
## B    5 2650    3    0    0 0.0030097818
## C    0    7 2388    1    0 0.0033388982
## D    0    0    9 2243    0 0.0039964476
## E    0    1    0    3 2521 0.0015841584
```
#Prediction
model_rf_predict    <- predict(model_rf, newdata = testing_set)
model_rf_predict_cm <- confusionMatrix(model_rf_predict, testing_set$classe)
model_rf_predict_cm

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    5    0    0    0
##          B    0 1134    2    0    0
##          C    0    0 1024    5    0
##          D    0    0    0  958    3
##          E    0    0    0    1 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9973          
##                  95% CI : (0.9956, 0.9984)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9966          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9956   0.9981   0.9938   0.9972
## Specificity            0.9988   0.9996   0.9990   0.9994   0.9998
## Pos Pred Value         0.9970   0.9982   0.9951   0.9969   0.9991
## Neg Pred Value         1.0000   0.9989   0.9996   0.9988   0.9994
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1927   0.1740   0.1628   0.1833
## Detection Prevalence   0.2853   0.1930   0.1749   0.1633   0.1835
## Balanced Accuracy      0.9994   0.9976   0.9985   0.9966   0.9985
```

plot(model_rf_predict_cm$table, main= 'Random Forest Prediction: Accuracy = 0.9985')

Boosted Model

In order to validate our results we try a second type of model, Boosting in this case. We achieve a 98.71 % Accuracy which is only a fraction lower than the one we achieved with a random forest. A plot again confirms the results.

set.seed(246810)
control_model <- trainControl(method = 'repeatedcv', number = 4, repeats = 1)
model_bm      <- train(classe~., method="gbm", verbose=F, trControl=control_model, data = training_set)
model_bm$finalModel

## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 53 predictors of which 42 had non-zero influence.

#Prediction
model_bm_predict    <- predict(model_bm, newdata = testing_set)
model_bm_predict_cm <- confusionMatrix(model_bm_predict, testing_set$classe) 
model_bm_predict_cm

## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    9    0    2    0
##          B    2 1119   12    1    3
##          C    0    9 1012   12    2
##          D    0    2    2  949   17
##          E    0    0    0    0 1060
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9876          
##                  95% CI : (0.9844, 0.9903)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9843          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9988   0.9824   0.9864   0.9844   0.9797
## Specificity            0.9974   0.9962   0.9953   0.9957   1.0000
## Pos Pred Value         0.9935   0.9842   0.9778   0.9784   1.0000
## Neg Pred Value         0.9995   0.9958   0.9971   0.9969   0.9954
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2841   0.1901   0.1720   0.1613   0.1801
## Detection Prevalence   0.2860   0.1932   0.1759   0.1648   0.1801
## Balanced Accuracy      0.9981   0.9893   0.9908   0.9901   0.9898

plot(model_bm_predict_cm$table, main= "Boosted Model: Accuracy = 0.9861")

Quiz prediction

In our final step we use the previously developed models to predict the ‘true’ testset provided. Using both methods we reach identidal results.
Using Random Forest

Quiz_prediction_rf <- predict(model_bm, newdata = testing_final)
Quiz_prediction_rf

##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E

Using a Boosted Model

Quiz_prediction_bm <- predict(model_rf, newdata = testing_final)
Quiz_prediction_bm

##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E

