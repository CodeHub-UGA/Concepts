# Titanic Dataset
#Read in the data
test = read.csv("~/downloads/test.csv", header=TRUE)
train = read.csv("~/downloads/train.csv", header=TRUE)

boy = cbind(train$PassengerId)
boy <- train[,1, drop=FALSE]

#X<-split(train, train$Survived)
#X
#Exploratory Analysis
#Quant analysis
min(na.omit(train$Age))
min(train$SibSp)
min(train$Parch)
min(train$Fare)

max(na.omit(train$Age))
max(train$SibSp)
max(train$Parch)
max(train$Fare)

mean(na.omit(train$Age))
mean(train$SibSp)
mean(train$Parch)
mean(train$Fare)

sd(na.omit(train$Age))
sd(train$SibSp)
sd(train$Parch)
sd(train$Fare)

#split by died vs. didnt die
min(na.omit(train[train$Survived == 0,]$Age))
min(na.omit(train[train$Survived == 0,]$SibSp))
min(na.omit(train[train$Survived == 0,]$Parch))
min(na.omit(train[train$Survived == 0,]$Fare))

mqx(na.omit(train[train$Survived == 0,]$Age))
max(na.omit(train[train$Survived == 0,]$SibSp))
max(na.omit(train[train$Survived == 0,]$Parch))
max(na.omit(train[train$Survived == 0,]$Fare))

mean(na.omit(train[train$Survived == 0,]$Age))
mean(na.omit(train[train$Survived == 0,]$SibSp))
mean(na.omit(train[train$Survived == 0,]$Parch))
mean(na.omit(train[train$Survived == 0,]$Fare))

sd(na.omit(train[train$Survived == 0,]$Age))
sd(na.omit(train[train$Survived == 0,]$SibSp))
sd(na.omit(train[train$Survived == 0,]$Parch))
sd(na.omit(train[train$Survived == 0,]$Fare))

min(na.omit(train[train$Survived == 1,]$Age))
min(na.omit(train[train$Survived == 1,]$SibSp))
min(na.omit(train[train$Survived == 1,]$Parch))
min(na.omit(train[train$Survived == 1,]$Fare))

max(na.omit(train[train$Survived == 1,]$Age))
max(na.omit(train[train$Survived == 1,]$SibSp))
max(na.omit(train[train$Survived == 1,]$Parch))
max(na.omit(train[train$Survived == 1,]$Fare))

mean(na.omit(train[train$Survived == 1,]$Age))
mean(na.omit(train[train$Survived == 1,]$SibSp))
mean(na.omit(train[train$Survived == 1,]$Parch))
mean(na.omit(train[train$Survived == 1,]$Fare))

sd(na.omit(train[train$Survived == 1,]$Age))
sd(na.omit(train[train$Survived == 1,]$SibSp))
sd(na.omit(train[train$Survived == 1,]$Parch))
sd(na.omit(train[train$Survived == 1,]$Fare))

#histograms and bar plots
hist(train$Survived)
barplot(table(train$Sex))
hist(na.omit(train$Age))
hist(train$SibSp)
hist(train$Parch)
hist(train$Fare)
barplot(table(train$Embarked))
hist(train$Pclass)

#split histograms by death event
barplot(table(train[train$Survived == 1,]$Sex))
hist(na.omit(train[train$Survived == 1,]$Age))
hist(train[train$Survived == 1,]$SibSp)
hist(train[train$Survived == 1,]$Parch)
hist(train[train$Survived == 1,]$Fare)
barplot(table(train[train$Survived == 1,]$Embarked))
hist(train[train$Survived == 1,]$Pclass)

barplot(table(train[train$Survived == 0,]$Sex))
hist(na.omit(train[train$Survived == 0,]$Age))
hist(train[train$Survived == 0,]$SibSp)
hist(train[train$Survived == 0,]$Parch)
hist(train[train$Survived == 0,]$Fare)
barplot(table(train[train$Survived == 0,]$Embarked))
hist(train[train$Survived == 0,]$Pclass)



