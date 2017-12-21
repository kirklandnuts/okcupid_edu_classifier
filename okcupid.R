# GOAL: CAN WE PREDICT EDUCATION LEVEL BASED ON WRITING STYLE?
# DATA: https://github.com/rudeboybert/JSE_OkCupid
# methods used: bag of words analysis


# Libraries ----
library(tidyverse)
library(quanteda)
library(readtext)
library(stringr)
library(topicmodels)
library(tidytext)
library(caret)
library(LiblineaR)
library(randomForest)

# Constants ----
set.seed(1)

DATA_DIR <- '/Users/timmy/Documents/school/year-3/pstat131/okcupid'

# EDUCATION LEVEL BINS
# rules:
#   (1) if it doesnt say "graduated" in it, we assumed its in progress
#   (2) we considered "dropped out" to count as "pursuing" bc the person did some of the work
#   (3) we lumped all of the "two-year college" education levels in with highschool
#       except for "graduated from two-year college" which goes in "pursuing UG"
HS_BIN <- c("graduated from high school", "high school", "working on two-year college",
            "dropped out of high school", "working on high school", "two-year college",
            "dropped out of two-year college")
P_UG_BIN <- c("working on college/university", "college/university",
              "dropped out of college/university", "graduated from two-year college")
R_UG_BIN <- c("graduated from college/university")
P_G_BIN <- c("working on masters program", "working on ph.d program", "working on med school",
             "masters program", "dropped out of med school", "dropped out of ph.d program",
             "working on law school", "law school", "dropped out of masters program",
             "ph.d program", "dropped out of law school", "med school")
R_G_BIN <- c("graduated from masters program", "graduated from ph.d program",
             "graduated from law school", "graduated from med school")
SC_BIN <- c("space camp", "dropped out of space camp", "working on space camp",
            "graduated from space camp")
BINS <- c("high school", "pursuing undergraduate degree", "received undergraduate degree",
          "pursuing graduate degree", "received graduate degree")

# Functions ----

# function for mapping education levels into one of the following bins:
#   [highschool, pursuing undergraduate degree, received undergraduate degree,
#   pursuing graduate degree, received graduate degree, other]
edu_bin <- function(s) {
  if (s %in% HS_BIN) {
    "high school"
  } else if (s %in% P_UG_BIN) {
    "pursuing undergraduate degree"
  } else if (s %in% R_UG_BIN) {
    "received undergraduate degree"
  } else if (s %in% P_G_BIN) {
    "pursuing graduate degree"
  } else if (s %in% R_G_BIN) {
    "received graduate degree"
  }
}

accuracy <- function(cfm, bin) {
  cfm[bin, bin]/sum(cfm[,bin])
}

gen_accuracy <- function(cfm) {
  all_acc <- c()
  for (bin in BINS) {
    acc <- accuracy(cfm, bin)
    all_acc <- c(all_acc, acc)
    cat("Accuracy for ", bin, ": ", acc, "\n", sep="")
  }
  cat("Average accuracy: ", mean(all_acc))
}

top_features <- function(model, edu, n = 20) {
  names(sort(model$W[edu,], T)[1:n])
}

gen_top_features <- function(model, n = 20) {
  for (bin in BINS) {
    cat("Top features for '", bin, "':\n", sep="")
    print(top_features(model, bin, n))
  }
}

# Main ----
# Paths
profiles_path <- file.path(DATA_DIR, 'profiles.csv')

# Loading data
original_profiles <- read_csv(profiles_path)
profiles <- original_profiles

### EDUCATION LEVEL BINNING
# there are 32 levels of education right now, we want fewer levels, so we'll bin them
unique(profiles$education)

# there are 6628 rows with missing values for education
sum(is.na(profiles$education))
# there are 1683 rows with education level involving "space camp", which we doesn't tell us much
sum(profiles$education %in% SC_BIN)
# we remove these rows bc it's only ~15% of our data, still leaving us with 52k rows
profiles <- profiles %>% filter(!education %in% c(NA, SC_BIN))

# now we use previously defined function to put education level into bins
education <- sapply(profiles$education, edu_bin)
# frequency counts for education level
table(education)
# barplot of education level freqs
barplot(table(education))
# notice: lots of ppl with degrees received...
# checking age, makes sense considering median age 30 and mean age 32
summary(profiles$age)


### FORMATTING TEXT
# concatenating answers from all questions to be one body of text
text <- paste(profiles$essay0, profiles$essay1, profiles$essay2,
              profiles$essay3, profiles$essay4, profiles$essay5,
              profiles$essay6, profiles$essay7, profiles$essay8,
              profiles$essay9, sep = " ")

# constructing new data frame with a column denoting sex of profile and
#   another containing their comprehensive writing
df <- data.frame(text, stringsAsFactors = F)
df$doc_id <- seq(dim(df)[1])

# building a corpus
my_corpus <- corpus(df)


### MODELING
# creating document frquency matrix
my_dfm <- dfm(my_corpus, remove = stopwords("english"), stem = TRUE,
              remove_punct = TRUE, remove_numbers = TRUE)
# removing features that occur in fewer than 2 documents
(my_dfm <- dfm_trim(my_dfm, min_docfreq = 1000, verbose=TRUE))

# using text frequency - inverse document frequency weighting
my_tfidf <- tfidf(my_dfm, normalize = TRUE) %>%
  as.matrix()

# sampling (creating vector of indices)
train_ind <- sample(seq_len(nrow(df)), size = nrow(df) * .8)

# using indices to pslit data into train and test
tr_X <- my_tfidf[train_ind,]
tr_Y <- education[train_ind]
te_X <- my_tfidf[-train_ind,]
te_Y <- education[-train_ind]
dim(tr_X)
dim(te_X)

# training first model
tr_X_s <- scale(tr_X, center=TRUE, scale = TRUE)

start.time <- Sys.time()
m <- LiblineaR(data = tr_X_s, target = tr_Y, type = 7, bias = 1, verbose = FALSE)
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

te_X_s <- scale(te_X,attr(tr_X_s,"scaled:center"),attr(tr_X_s,"scaled:scale"))
p <- predict(m, te_X_s, decisionValues = TRUE)

cfm_init_logreg <- table(p$predictions,te_Y)
print(cfm_init_logreg)
gen_accuracy(cfm_init_logreg)

# poor results probably due to imblanced classes
# imbalanced classes:
# https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
# DOWNSAMPLING
df_ds <- downSample(df, as.factor(education), yname ='education')
education_ds <- df_ds$education
df_ds <- df_ds %>% select(-education)

my_corpus_ds <- corpus(df_ds)

# making document frequency matrix from downsampled data
my_dfm_ds <- dfm(my_corpus_ds, remove = stopwords("english"), stem = TRUE,
                 remove_punct = TRUE, remove_numbers = TRUE)
# removing features that occur in fewer than 100 documents
(my_dfm_ds <- dfm_trim(my_dfm_ds, min_docfreq = 100, verbose=TRUE))

# using text frequency - inverse document frequency weighting on downsampled dfm
my_tfidf_ds <- tfidf(my_dfm_ds, normalize = TRUE) %>%
  as.matrix()

# splitting downsampled data
train_ind <- sample(seq_len(nrow(df_ds)), size = nrow(df_ds) * .8)

tr_X_ds <- my_tfidf_ds[train_ind,]
tr_Y_ds <- education_ds[train_ind]
te_X_ds <- my_tfidf_ds[-train_ind,]
te_Y_ds <- education_ds[-train_ind]
dim(tr_X_ds)
dim(te_X_ds)

# scaling downsampled data
tr_X_ds_s <- scale(tr_X_ds, center = TRUE, scale = TRUE)
te_X_ds_s <- scale(te_X_ds, center = TRUE, scale = TRUE)

# training RF
start.time <- Sys.time()
rf1 <- randomForest(tr_X_ds_s, y = tr_Y_ds, ntree = 100)
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

p_rf_ds <- predict(rf1, te_X_ds_s)
# confusion matrix
cfm_rf <- table(p_rf_ds,te_Y_ds)
print(cfm_rf)
# outputting accuracy
gen_accuracy(cfm_rf)

# trying multiple cost values
costs <- c(100, 1, 0.01, 0.001)
best_cost <- NA
best_acc <- 0
for(cost in costs) {
  start.time <- Sys.time()
  acc <- LiblineaR(data=tr_X_ds_s,target=tr_Y_ds,type=7,cost=cost,bias=1,cross=3,verbose=FALSE)
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  cat("time taken:", time.taken, sep = " ")
  cat("Results for C=", cost, " : ", acc, "accuracy.\n", sep="")
  if(acc>best_acc){
    best_cost=cost
    best_acc=acc
  }
}

cat("Best cost is:",best_cost,"\n")
cat("Best accuracy is:",best_acc,"\n")

# Re-train best model with best cost value.
best_lr_model <- LiblineaR(data=tr_X_ds_s,target=tr_Y_ds,type=7,cost=best_cost,bias=1,verbose=FALSE)
try_lr_model <- LiblineaR(data=tr_X_ds_s,target=tr_Y_ds,type=7,cost=0.001,bias=1,verbose=FALSE)
# Scale the test data
te_X_ds_s <- scale(te_X_ds,attr(tr_X_ds_s,"scaled:center"),attr(tr_X_ds_s,"scaled:scale"))
# Make prediction

p_lr_ds=predict(best_lr_model,te_X_ds_s,proba=TRUE,decisionValues=TRUE)
cfm_lr <- table(p_lr_ds$predictions,te_Y_ds)
print(cfm_lr)
# outputting accuracy
gen_accuracy(cfm_lr)
gen_top_features(best_lr_model)

p_lr_ds2=predict(try_lr_model,te_X_ds_s,proba=TRUE,decisionValues=TRUE)
cfm_lr2 <- table(p_lr_ds2$predictions,te_Y_ds)
print(cfm_lr2)
# outputting accuracy
gen_accuracy(cfm_lr2)
gen_top_features(try_lr_model)