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


# Functions ----
# clean_text <- function(txt) {
#   new_text <- txt %>% tolower() %>%
#     str_replace_all("\\n", " ") %>%
#     str_replace_all("<br \\/>", " ") %>%
#     str_replace_all("\\s+", " ")
#   new_text
# }

# function for mapping education levels into one of the following bins:
#   [highschool, pursuing undergraduate degree, received undergraduate degree,
#     pursuing graduate degree, received graduate degree, other]
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
text <- paste('"', profiles$essay0, profiles$essay1, profiles$essay2,
              profiles$essay3, profiles$essay4, profiles$essay5,
              profiles$essay6, profiles$essay7, profiles$essay8,
              profiles$essay9, '"', sep = " ")

# constructing new data frame with a column denoting sex of profile and
#   another containing their comprehensive writing
df <- data.frame(text, stringsAsFactors = F)
df$doc_id <- seq(dim(df)[1])

# building a corpus
my_corpus <- corpus(df)

# training SVM
my_dfm <- dfm(my_corpus, remove = stopwords("english"), stem = TRUE,
              remove_punct = TRUE, remove_numbers = TRUE)
# removing features that occur in fewer than 2 documents
(my_dfm <- dfm_trim(my_dfm, min_docfreq = 1000, verbose=TRUE))

# using text frequency - inverse document frequency weighting
my_tfidf <- tfidf(my_dfm, normalize = TRUE) %>%
  as.matrix()

train_ind <- sample(seq_len(nrow(df)), size = nrow(df) * .8)

tr_X <- my_tfidf[train_ind,]
tr_Y <- education[train_ind]
te_X <- my_tfidf[-train_ind,]
te_Y <- education[-train_ind]
dim(tr_X)
dim(te_X)

library(LiblineaR)
tr_X_s <- scale(tr_X, center=TRUE, scale = TRUE)

start.time <- Sys.time()
m <- LiblineaR(data = tr_X_s, target = tr_Y, type = 1, bias = TRUE, verbose = FALSE)
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

te_X_s <- scale(te_X, center=TRUE, scale = TRUE)
p <- predict(m, te_X_s, decisionValues = TRUE)

res <- table(p$predictions,te_Y)
print(res)
BCR=mean(c(res[1,1]/sum(res[,1]),res[2,2]/sum(res[,2]),res[3,3]/sum(res[,3]), res[4,4]/sum(res[,4]), res[5,5]/sum(res[,5])))
print(BCR)

library(e1071)
svm1 <- svm(x = tr_X, y = as.factor(tr_Y), kernel = 'linear')
tune.out=
  tune(svm,y~.,data=dat,kernel="linear", ranges=list(cost=c(0.001, 0.01, 0.1,1,5,10,100)))


library(randomForest)



# DOWNSAMPLING
df_ds <- downSample(df, as.factor(education), yname ='education')
education_ds <- df_ds$education
df_ds <- df_ds %>% select(-education)

my_corpus_ds <- corpus(df_ds)

# making document frequency matrix
my_dfm_ds <- dfm(my_corpus_ds, remove = stopwords("english"), stem = TRUE,
              remove_punct = TRUE, remove_numbers = TRUE)
# removing features that occur in fewer than 2 documents
(my_dfm_ds <- dfm_trim(my_dfm_ds, min_docfreq = 100, verbose=TRUE))

# using text frequency - inverse document frequency weighting
my_tfidf_ds <- tfidf(my_dfm_ds, normalize = TRUE) %>%
  as.matrix()

train_ind <- sample(seq_len(nrow(df_ds)), size = nrow(df_ds) * .8)

tr_X_ds <- my_tfidf_ds[train_ind,]
tr_Y_ds <- education_ds[train_ind]
te_X_ds <- my_tfidf_ds[-train_ind,]
te_Y_ds <- education_ds[-train_ind]
dim(tr_X_ds)
dim(te_X_ds)

tr_X_ds_s <- scale(tr_X_ds, center = TRUE, scale = TRUE)
te_X_ds_s <- scale(te_X_ds, center = TRUE, scale = TRUE)

start.time <- Sys.time()
rf1 <- randomForest(tr_X_ds_s, y = tr_Y_ds, ntree = 50)
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken


p_ds <- predict(rf1, te_X_ds_s)
res_ds <- table(p_ds,te_Y_ds)
print(res_ds)
BCR_ds <- mean(c(res_ds[1,1]/sum(res_ds[,1]),res_ds[2,2]/sum(res_ds[,2]),res_ds[3,3]/sum(res_ds[,3]), res_ds[4,4]/sum(res_ds[,4]), res_ds[5,5]/sum(res_ds[,5])))
print(BCR_ds)
# imbalanced classes:
# https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/

types <- c(0:7)
costs <- c(1000,1,0.001)
best_cost <- NA
best_acc <- 0
best_type <- NA
start.time <- Sys.time()
for(type in types){
  for(cost in costs){
    start.time <- Sys.time()
    acc <- LiblineaR(data=tir_X_ds_s,target=tr_Y_ds,type=type,cost=cost,bias=1,cross=5,verbose=FALSE)
    end.time <- Sys.time()
    time.taken <- end.time - start.time
    time.taken
    cat("Results for C=", cost, " : ", acc, "accuracy.\n", sep="")
    if(acc>best_acc){
      best_cost=cost
      best_acc=acc
      best_type=type
    }
  }
}
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

cat("Best model type is:",bestTyp fe,"\n")
cat("Best cost is:",bestCost,"\n")
cat("Best accuracy is:",bestAcc,"\n")
# Re-train best model with best cost value.
best_model <- LiblineaR(data=tr_X_ds_s,target=tr_Y_ds,type=best_type,cost=best_cost,bias=1,verbose=FALSE)
# Scale the test data
te_X_ds_s <- scale(te_X_ds,attr(tr_X_ds_s,"scaled:center"),attr(tr_X_ds_s,"scaled:scale"))
# Make prediction
pr=FALSE
if(bestType==0 || bestType==7) pr=TRUE
p=predict(m,s2,proba=pr,decisionValues=TRUE)
