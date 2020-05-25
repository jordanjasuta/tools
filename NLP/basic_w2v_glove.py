"""
Very messy and basic word2vec model using transfer learning from GloVe
* the GloVe model can be downloaded here: https://www.kaggle.com/thanakomsn/glove6b300dtxt *

Model can be trained using command `python word2vec_d2.py train trainingdata.csv`;
model can be tested using command `python word2vec_d2.py test testdata.csv`
"""

# import gensim
from gensim.models import Word2Vec, KeyedVectors
import multiprocessing
import pandas as pd
import numpy as np
import sys
import re
from copy import deepcopy
import pickle


# define vars and functions for data cleanup

contractions = re.compile(r"'|-|\"")
symbols = re.compile(r'(\W+)', re.U)           # all non alphanumeric
singles = re.compile(r'(\s\S\s)', re.I|re.U)    # single character removal
seps = re.compile(r'\s+')                      # separators (any whitespace)

def clean(text):    # cleaner
    text = text.lower()
    text = contractions.sub('', text)
    text = symbols.sub(r' \1 ', text)
    text = singles.sub(' ', text)
    text = seps.sub(' ', text)
    return text

alteos = re.compile(r'([!\?])')

def sentences(l):      # sentence splitter
    l = alteos.sub(r' \1 .', l).rstrip("(\.)*\n")
    return l.split(".")

def setup_data(data):
    for index, row in data.iterrows():
        yield {'y':row['TRUTH'],\
               'x':[clean(s).split() for s in sentences(row['TEXT'])]}

# set categories - obviously these would need to be changed depending on the analysis
categories = ['UNKNOWN','EXAM_ISSUE','DEPENDENCY_REQUEST','PAYMENT_ISSUE','NOTICE_OF_DEATH','DEPENDENCY_REQUEST_REMOVAL']

def GetStrings(text, categories = categories):   # pull out text strings
    for t in text:
        if t['y'] in categories:
            for c in t['x']:
                yield c


# def funciton to get composite probabilities
def docprob(docs, models):
    sen_list = [s for d in docs for s in d]    # format as list of sentences [s]
    log_like = np.array( [ m.score(sen_list, len(sen_list)) for m in models ] )    # log likelihood of each sentence in this doc for each w2v representation
    # exponentiate to get likelihoods
    like = np.exp(log_like - log_like.max(axis=0))    # subtract row max to avoid numeric overload
    prob = pd.DataFrame((like/like.sum(axis=0)).transpose())    # normalize across models to get sentence-category probabilities
    prob["doc"] = [i for i,d in enumerate(docs) for s in d]
    prob = prob.groupby("doc").mean()      # average the sentence probabilities to get the review probability
    return prob



def train_models(data):
    # create a w2v learner

    # basemodel = Word2Vec(iter=3, hs=1, negative=0)
    # print('basemodel:', basemodel)
    # # build vocab
    # basemodel.build_vocab(GetStrings(data))
    # print('basemodel after vocab build:', basemodel)

    basemodel = Word2Vec(size=300, min_count=1,hs=1, negative=0)
    print('basemodel:', basemodel)

    basemodel.build_vocab(GetStrings(train))
    print('basemodel after vocab build:', basemodel)
    total_examples = basemodel.corpus_count
    model = KeyedVectors.load_word2vec_format("glove.6B.300d.txt", binary=False)   # incorporate GloVe into base (transfer learning)
    basemodel.build_vocab([list(model.vocab.keys())], update=True)
    basemodel.intersect_word2vec_format("glove.6B.300d.txt", binary=False, lockf=1.0)
    basemodel.train(GetStrings(train), total_examples=total_examples, epochs=basemodel.epochs)   # incorporate our data into base

    # deep copy for each category and train each
    models = [deepcopy(basemodel) for i in categories]
    cat_num = -1
    for i in categories:
        slist = list(GetStrings(data, i))
        print(i, "category (", len(slist), ")")
        cat_num += 1
        models[cat_num].train(  slist, total_examples=len(slist), epochs=3 )
        print('finished training model for',i)
    pickle.dump(models, open('w2v_models.pkl', 'wb'))



def test_models(data):
    models = pickle.load(open('w2v_models.pkl', 'rb'))
    probs = docprob( [r['x'] for r in data], models )
    probs.columns = categories
    probs['pred'] = probs.idxmax(axis=1)
    probs['pred_prob'] = probs.max(axis=1)
    test_concat = pd.concat([testing, probs], axis=1, sort=False)
    test_concat['correct'] = np.where(test_concat['pred'] == test_concat['TRUTH'], 1, 0)
    print(test_concat['correct'].value_counts())
    return test_concat






if __name__ == '__main__':
    # print('you forgot to fill in the main')

    if sys.argv[1] == 'train':
        # load training dataset (0820 to start with)
        training = pd.read_csv(sys.argv[2])
        train = list(setup_data(training))
        print(len(train), "training examples")
        np.random.shuffle(train)
        models = train_models(train)

    elif sys.argv[1] == 'test':
        testing = pd.read_csv(sys.argv[2])
        test = list(setup_data(testing))
        test_results = test_models(test)
        test_results.to_csv('test_results.csv')




    #
