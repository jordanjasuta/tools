"""
Basic word2vec model using transfer learning from GloVe + MLP classifier
* the GloVe model can be downloaded here: https://www.kaggle.com/thanakomsn/glove6b300dtxt *

Model can be trained using command `python word2vec_d2.py train trainingdata.csv`;
model can be tested using command `python word2vec_d2.py test testdata.csv`
"""

# import gensim
from gensim.models import Word2Vec, KeyedVectors
# import multiprocessing
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import sys
import re
# from copy import deepcopy
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

# set categories - obviously these dummy categories would need to be changed
#                  depending on the categories of each analysis
categories = ['one','two','three','four','five','six','seven','eight','nine','ten','eleven']

def GetStrings(text, categories = categories):   # pull out text strings
    for t in text:
        if t['y'] in categories:
            for c in t['x']:
                yield c


def transform(X, model):
    return np.array([
        np.mean([model.wv[w] for w in words if w in model.wv]
#                 or [np.zeros(self.dim)]
                , axis=0)
        for words in X
    ])


#
# # def funciton to get composite probabilities
# def docprob(docs, models):
#     sen_list = [s for d in docs for s in d]    # format as list of sentences [s]
#     log_like = np.array( [ m.score(sen_list, len(sen_list)) for m in models ] )    # log likelihood of each sentence in this doc for each w2v representation
#     # exponentiate to get likelihoods
#     like = np.exp(log_like - log_like.max(axis=0))    # subtract row max to avoid numeric overload
#     prob = pd.DataFrame((like/like.sum(axis=0)).transpose())    # normalize across models to get sentence-category probabilities
#     prob["doc"] = [i for i,d in enumerate(docs) for s in d]
#     prob = prob.groupby("doc").mean()      # average the sentence probabilities to get the review probability
#     return prob

def get_basemodel(data):
    training = pd.read_csv(data, encoding='latin1')
    training['TEXT'] = training['TEXT'].apply(lambda x: x.lower())
    train_X = training['TEXT']
    train_y = training['TRUTH']
    print('training data stats:')
    print(train_y.value_counts())
    print('total training records: ', len(train_y))

    basemodel = Word2Vec(size=300, min_count=1,hs=1, negative=0)
    print('basemodel:', basemodel)

    train = list(setup_data(training))
    basemodel.build_vocab(GetStrings(train))
    print('basemodel after vocab build:', basemodel)
    total_examples = basemodel.corpus_count
    model = KeyedVectors.load_word2vec_format("glove.6B.300d.txt", binary=False)   # incorporate GloVe into base (transfer learning)
    basemodel.build_vocab([list(model.vocab.keys())], update=True)
    basemodel.intersect_word2vec_format("glove.6B.300d.txt", binary=False, lockf=1.0)
    basemodel.train(GetStrings(train), total_examples=total_examples, epochs=basemodel.epochs)   # incorporate our data into base
    print('basemodel with GloVe and vectorized training data build: ', basemodel)

    return basemodel



def train_model(data):
    #read csv
    training = pd.read_csv(data, encoding='latin1')
    training['TEXT'] = training['TEXT'].apply(lambda x: x.lower())
    train_X = training['TEXT']
    train_y = training['TRUTH']
    # print('training data stats:')
    # print(train_y.value_counts())
    # print('total training records: ', len(train_y))
    #
    # basemodel = Word2Vec(size=300, min_count=1,hs=1, negative=0)
    # print('basemodel:', basemodel)
    #
    # train = list(setup_data(training))
    # basemodel.build_vocab(GetStrings(train))
    # print('basemodel after vocab build:', basemodel)
    # total_examples = basemodel.corpus_count
    # model = KeyedVectors.load_word2vec_format("glove.6B.300d.txt", binary=False)   # incorporate GloVe into base (transfer learning)
    # basemodel.build_vocab([list(model.vocab.keys())], update=True)
    # basemodel.intersect_word2vec_format("glove.6B.300d.txt", binary=False, lockf=1.0)
    # basemodel.train(GetStrings(train), total_examples=total_examples, epochs=basemodel.epochs)   # incorporate our data into base
    # print('basemodel with GloVe and vectorized training data build: ', basemodel)

    basemodel = get_basemodel(data)
    pickle.dump(basemodel, open('w2v_glove_basemodel.pkl', 'wb'))
    print('basemodel saved to working directory')

    transformed_X = transform(train_X, basemodel)
    #import NN classifier
    clf = MLPClassifier(max_iter=200, verbose=True)
    model = clf.fit(transformed_X, train_y)
    print('W2V + GloVe + MLP classifier model trained!')

    pickle.dump(model, open('w2v_mlp_model.pkl', 'wb'))
    print('model saved to working directory')


def classify(text):
#     if self.model is None:
#         print('Loading classifier')
#         self.model = self.load_classifier()

#     text = self.fixup(text)
    # with open('w2v_mlp_model.pkl', "rb") as f:
    #     model = pickle.load(f)
    basemodel = pickle.load(open('w2v_glove_basemodel.pkl', 'rb'))
    model = pickle.load(open('w2v_mlp_model.pkl', 'rb'))

    X = transform([text], basemodel)

#     if self.verbose:
#         print(X[0,])

    guesses = model.predict_proba(X)
    # print(guesses)

    top_score = 0
    top_category = None
    for i, guess in enumerate(guesses[0]):
        print(guess, model.classes_[i])
        if guess > top_score:
            top_score = guess
            top_category = model.classes_[i]

    #get keywords
#     keywords = sorted(list(set(self.get_keyword_counts(text).keys())))

    matches = list()
    match = dict()
#     if len(keywords) == 0:
#         match["top_score"] = 0.99
#         match["top_category"] = "UNKNOWN"
#         match["keyword_counts"] = keywords
#     else:
    match["top_score"] = round(top_score,2)
    match["top_category"] = top_category
#         match["keyword_counts"] = keywords

    matches.append(match)

    return matches








def test_model(data):
    testing = pd.read_csv(sys.argv[2])
    test = list(setup_data(testing))
    # models = pickle.load(open('w2v_models.pkl', 'rb'))
    # probs = docprob( [r['x'] for r in data], models )
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
        training_data = sys.argv[2]
        # train = list(setup_data(training))
        # print(len(train), "training examples")
        # np.random.shuffle(train)
        models = train_model(training_data)

    elif sys.argv[1] == 'test':
        test_data = sys.argv[2]
        test_results = classify(test_data)
        print(test_results)
        # test_results.to_csv('test_results.csv')




    #
