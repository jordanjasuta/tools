##### PMR First Look #####


install.packages('textmineR')
library(rstan)
library(dplyr)
library(tidytext)
library(tidyverse)
library(tm)
library(textmineR)
library(stringr)

getwd()
setwd('Documents/Training/PMR/')
load(file = "findings_orig.lang_delay.Rda")

length(unique(findings_orig.lang_delay$OperNum))
# 779 operations

table(findings_orig.lang_delay$language)
table(findings_orig.lang_delay$Delay_category_95)
hist(findings_orig.lang_delay$months_delay_95perc, breaks = 50)



# NLP steps: 
# 1. pull out spanish
esp <- findings_orig.lang_delay[findings_orig.lang_delay$language == 'spanish',]
length(unique(esp$OperNum))   # 581 operations
table(esp$Delay_category_95, exclude = NULL)
esp <- na.omit(esp)
esp$num_chars <- nchar(esp$finding)
min(esp$num_chars)
max(esp$num_chars)
length(unique(esp$OperNum))   # 334 operations with delay data
hist(esp$months_delay_95perc, breaks = 50)


# 2. pre-processing
#### refer to: https://towardsdatascience.com/beginners-guide-to-lda-topic-modelling-with-r-e57a5a8e7a25

# strip punctuation
head(esp$finding, 1)
esp$finding <- gsub('[[:punct:] ]+', ' ', esp$finding)
head(esp$finding, 1)


# filter stopwords
spanish_stop_words <- bind_rows(stop_words,
                               data_frame(word = tm::stopwords("spanish"),
                                          lexicon = "custom"))

# custom_stop_words <- bind_rows(stop_words,
#                                data_frame(word = quanteda::data_char_stopwords$spanish,
#                                           lexicon = "custom"))


esp$nostop <- removeWords(esp$finding, spanish_stop_words$word)
head(esp$nostop, 1)
esp$nostop <- str_replace(gsub("\\s+", " ", str_trim(esp$nostop)), "B", "b")    # remove extra spaces
head(esp$nostop, 1)
esp$ID <- seq.int(nrow(esp))   # add ID number


# some kind of fuzzy matching spell check layer???
# lemmatization to avoid conjugation problems??





# 2. tf-idf to find keywords
#### also refer to: https://medium.com/swlh/text-classification-using-tf-idf-7404e75565b8 (would have to convert to R) 
#### and/or https://cran.r-project.org/web/packages/superml/vignettes/Guide-to-TfidfVectorizer.html 
#### for direct tf-idf model
get_keywords <- function(df){
  # create document term matrix
  dtm <- CreateDtm(df$nostop, 
                   doc_names = df$ID, 
                   ngram_window = c(1, 2))
  
  #explore the basic frequency
  tf <- TermDocFreq(dtm = dtm)
  original_tf <- tf %>% select(term, term_freq,doc_freq)
  rownames(original_tf) <- 1:nrow(original_tf)
  # Eliminate words appearing less than 2 times or in more than half of the
  # documents
  vocabulary <- tf$term[ tf$term_freq > 1 & tf$doc_freq < nrow(dtm) / 2 ]
  # x <- data.frame(vocabulary)
  dtm = dtm
  return(vocabulary)
}



# 3. LDA to find topics associated with risk

model_topics <- function(vocabulary){
  k_list <- seq(1, 20, by = 1)
  model_dir <- paste0("models_", digest::digest(vocabulary, algo = "sha1"))
  if (!dir.exists(model_dir)) dir.create(model_dir)
  model_list <- TmParallelApply(X = k_list, FUN = function(k){
    filename = file.path(model_dir, paste0(k, "_topics.rda"))
    
    if (!file.exists(filename)) {
      m <- FitLdaModel(dtm = dtm, k = k, iterations = 500)
      m$k <- k
      m$coherence <- CalcProbCoherence(phi = m$phi, dtm = dtm, M = 5)
      save(m, file = filename)
    } else {
      load(filename)
    }
    
    m
  }, export=c("dtm", "model_dir")) # export only needed for Windows machines
  #model tuning
  #choosing the best model
  coherence_mat <- data.frame(k = sapply(model_list, function(x) nrow(x$phi)), 
                              coherence = sapply(model_list, function(x) mean(x$coherence)), 
                              stringsAsFactors = FALSE)
  ggplot(coherence_mat, aes(x = k, y = coherence)) +
    geom_point() +
    geom_line(group = 1)+
    ggtitle("Best Topic by Coherence Score") + theme_minimal() +
    scale_x_continuous(breaks = seq(1,20,1)) + ylab("Coherence")
  
  # best k = 8 so run model with k=8
  model <- model_list[which.max(coherence_mat$coherence)][[ 1 ]]
  model$top_terms <- GetTopTerms(phi = model$phi, M = 20)
  top20_wide <- as.data.frame(model$top_terms)
  
  # dendrogram for overlapping topics
  model$topic_linguistic_dist <- CalcHellingerDist(model$phi)
  model$hclust <- hclust(as.dist(model$topic_linguistic_dist), "ward.D")
  model$hclust$labels <- paste(model$hclust$labels, model$labels[ , 1])
  plot(model$hclust)
  
  to_return <- list(top20_wide, model$hclust)
  return(to_return)
}


#### topics for ALL spanish docs... ####
vocabulary <- get_keywords(esp)
top20_all <- model_topics(vocabulary)
### The words are in ascending order of phi-value. 
### The higher the ranking, the more probable the word will belong to the topic.

model$topic_linguistic_dist <- CalcHellingerDist(model$phi)
model$hclust <- hclust(as.dist(model$topic_linguistic_dist), "ward.D")
model$hclust$labels <- paste(model$hclust$labels, model$labels[ , 1])
plot(model$hclust)

#### compare to delayed docs...  ####
table(esp$Delay_category_95)
esp_delayed <- esp[esp$Delay_category_95 != 'Early/on time',]
vocabulary_delayed <- get_keywords(esp_delayed)
top20_delayed <- model_topics(vocabulary_delayed)


#### compare to early/on time docs, etc...  ####
table(esp$Delay_category_95)
esp_on_time <- esp[esp$Delay_category_95 == 'Early/on time',]
vocabulary_on_time <- get_keywords(esp_on_time)

top20_on_time <- model_topics(vocabulary_on_time)
plot(top20_on_time[[2]])
a <- top20_on_time[[1]]



# for viz ideas: https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0#:~:text=LDA%20is%20a%20generative%20probabilistic,a%20set%20of%20topic%20probabilities.



# 4. build keywords + NN model to see if there's predictive power in these keywords 








#

# myfunction <- function(arg1, arg2, ... ){
#   # statements
#   return(object)
# }