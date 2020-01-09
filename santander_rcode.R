#LOAD REQUIRED LIBRARIES

library(dplyr)
library(ROSE)
library(randomForest)
library(e1071)
library(caret)
library(naivebayes)
 
library(ggplot2)

#LOAD THE DATA SETS
#set the directory
getwd()
setwd("C:/Users/My guest/Desktop/project 2")
#read the csve file from directory
read.csv('train.csv')->df_train
read.csv('test.csv')->df_test



#SOME DISCRIPTIVE ANALYSIS OF DATA
str(df_train)
str(df_test)
dim(df_train)
dim(df_test)
typeof(df_train)
typeof(df_test)
colnames(df_train)
colnames(df_test)


#first column does not provide any usefull information so lets remove it
df_train<-select (df_train,-c(1))
df_test<-select(df_test,-c(1))


# target column of train data frame is categorical so lets do type cast

#CHANGING THE VARIABLE TYPE OF TARGET COLUMN OF DATA FARAME
df_train$target<-as.factor(df_train$target)


#missing value analysis
sum(is.na(df_train))
sum(is.na(df_test))

#there are no na values in the data

# removing outliers
#outlier analysis
#selecting all numeric variables becaues outliers work on numeric variables only
numeric_index<-sapply(df_train,is.numeric)
numeric_index
numeric_data<-df_train[,numeric_index]
cnames<-colnames(numeric_data)
cnames

#removing outliesr
for (i in cnames) {
  print(i)
  val<-df_train[,i][df_train[,i]%in%boxplot.stats(df_train[,i])$out]
  print(length(val))
  df_train<-df_train[which(!df_train[,i] %in% val),]
}

str(df_train)
dim(df_train)
summary(df_train)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#train data  is free from  outliers
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
table(df_train$target)

data_balanced_under <- ovun.sample(target ~ ., data = df_train, method = "under", N = 34206, seed = 1)$data
table(df_train$target)
table(data_balanced_under$target)

df<-data_balanced_under
df$target<-as.factor(df$target)
summary(df)


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# corelational analysis
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#Correlations in train data
#convert factor to int
df$target<-as.numeric(df$target)
train_correlations<-cor(df[,-c(1)])
train_correlations

# there is very less corelation between the columns so we have to select all columns



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#SPLITTING THE DATA IN TO TRAIN AND TEST 
#@##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#Split the training data using simple random sampling
train_index<-sample(1:nrow(df),0.75*nrow(df))
#train data
train_data<-df[train_index,]
#validation data
test_data<-df[-train_index,]
#dimension of train and validation data
dim(train_data)
dim(test_data)
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# LOGISTIC REGRESSION MODEL
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

model=glm(target~.,family="binomial", data = train_data)
model
lr_prediction<-predict(model,test_data[,-c(1)])
table(lr_prediction,test_data$target)
accuracy=(3412+3312)/(3412+3312+884+944)
accuracy
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#RANDOM FOREST
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#Training the Random forest classifier
set.seed(2732)
#convert to int to factor
train_data$target<-as.factor(train_data$target)
#setting the mtry
mtry<-floor(sqrt(200))
#setting the tunegrid
tuneGrid<-expand.grid(.mtry=mtry)
#fitting the ranndom forest
rf<-randomForest(target~.,train_data,mtry=mtry,ntree=10,importance=TRUE)

print(rf)
rf$confusion
set.seed(8909)

p<-predict(rf,test_data[,-c(1)])
print(head(p,20))
print(head(test_data$target,20))


#confusionmatrix
table(p,test_data$target)

head(p)
head(test_data$target)
table(test_data$target)
table(p)
#error rate of random forest
plot(rf)

rf$err.rate
accuracy=(2862+2532)/(2862+1724+1434+2532)
accuracy
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# NAIVE BAYES
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#Fitting the Naive Bayes model
Naive_Bayes_Model=naiveBayes(target ~., data=train_data)
#What does the model say? Print the model summary
Naive_Bayes_Model

#Prediction on the dataset
NB_Predictions=predict(Naive_Bayes_Model,test_data[,-c(1)])
#Confusion matrix to check accuracy
table(NB_Predictions,test_data$target)
accuracy=(3535+3374)/(3535+3374+761+882)*100
accuracy






#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@ since the acuuracy of random forest is maximum so we select it for predicting target vatiable for test.csv
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


target_pred=predict(Naive_Bayes_Model,df_test)
