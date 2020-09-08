"""
Basic doc2vec model

Model can be trained using command `python doc2vec.py train trainingdata.csv`;
model can be tested using command `python doc2vec.py test testdata.csv`;
single line testing can be executed with command `python doc2vec "text"`

Author: Jordan Jasuta Fischer
"""

import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import numpy as np

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from sklearn.linear_model import LogisticRegression
from sklearn import utils

import sys
import os
import pickle

from datetime import datetime

# Needed to find subdirs
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from CI.form_4138_classifier import Form4138Classifier


class Doc2VecModel:
    def __init__(self, threshold=0.9):

        # define vars and functions for data cleanup
        # self.contractions = re.compile(r"'|-|\"")
        # self.symbols = re.compile(r'(\W+)', re.U)           # all non alphanumeric
        # self.singles = re.compile(r'(\s\S\s)', re.I|re.U)    # single character removal
        # self.seps = re.compile(r'\s+')                      # separators (any whitespace)
        # self.alteos = re.compile(r'([!\?])')

        # set categories - obviously these would need to be changed depending on the analysis
        self.categories = ['CATEGORY1',
                           'CATEGORY2',
                           'CATEGORY3',
                           'CATEGORY4',
                           'CATEGORY5',
                           'CATEGORY6',
                           'CATEGORY7',
                           'CATEGORY8',
                           'CATEGORY9',
                           'UNKNOWN']


    # def clean(self, text):    # cleaner
    #     text = text.lower()
    #     text = self.contractions.sub('', text)
    #     text = self.symbols.sub(r' \1 ', text)
    #     text = self.singles.sub(' ', text)
    #     text = self.seps.sub(' ', text)
    #     return text

    def get_vectors(self, model, tagged_docs):
        sents = tagged_docs.values
        targets, regressors = zip(*[(doc.tags, model.infer_vector(doc.words, steps=20)) for doc in sents])
        return targets, regressors

    def tag_doc(self, doc):
        preprocessed = gensim.utils.simple_preprocess(doc["TEXT"].strip())
        tagged = gensim.models.doc2vec.TaggedDocument(preprocessed, doc["TRUTH"])
        return tagged

    def train_model(self, training_file):
        training_data = pd.read_csv(training_file,
                                    encoding='latin1',
                                    keep_default_na=False)
        train_tagged = training_data.apply(lambda r: self.tag_doc(r), axis=1)
        print(train_tagged.values[30])
        # build vocab
        model = Doc2Vec(dm=1, vector_size=500, negative=5, hs=0, min_count=1, sample = 0)
        model.build_vocab([x for x in tqdm(train_tagged.values)])

        for epoch in range(30):
            model.train(utils.shuffle([x for x in tqdm(train_tagged.values)]),
                        total_examples=len(train_tagged.values),
                        epochs=1)
            model.alpha -= 0.001
            model.min_alpha = model.alpha

        print('finished training doc2vec model on ', training_file)
        pickle.dump(model, open('d2v_model.pkl', 'wb'))
        print('doc2vec model saved as pickle file')

        y_train, X_train = self.get_vectors(model, train_tagged)
        logreg = LogisticRegression(n_jobs=1, C=1e5, max_iter=150)
        logreg.fit(X_train, y_train)

        print('finished training LR classifier on doc2vec regressors')
        pickle.dump(logreg, open('LR_d2v_classifier.pkl', 'wb'))
        print('LR classifier saved as pickle file')

    def classify(self, doc):
        # load d2v model
        model = pickle.load(open('d2v_model.pkl', 'rb'))
        preprocessed = gensim.utils.simple_preprocess(doc.strip())
        regressors = model.infer_vector(preprocessed, steps=20)
        #load LR model
        logreg = pickle.load(open('LR_d2v_classifier.pkl', 'rb'))
        # precict
        y_pred = logreg.predict((regressors,))
        # print(y_pred)

        return y_pred




if __name__ == '__main__':
    # print('you forgot to fill in the main')
    D = Doc2VecModel()

    if sys.argv[1] == 'train':
        start=datetime.now()
        training = sys.argv[2]
        print(len(training), "training examples")
        model = D.train_model(training)

        print("Total training runtime: ", datetime.now()-start)

    elif sys.argv[1] == 'test':
        testing = pd.read_csv(sys.argv[2], encoding = 'Latin-1', keep_default_na=False)
        print("testing on ", sys.argv[2])
        # y_test, X_test = get_vectors(model_dbow, test_tagged)
        # y_pred = logreg.predict(X_test)

        results = list()
        correct = 0
        wrong = 0

        for index, row in testing.iterrows():
            match = D.classify(row['TEXT'])
            # print('match', match[0])
            result = dict()
            result['DOC_ID'] = row['DOC_ID']
            result['TEXT'] = row['TEXT'].strip()
            result['TRUTH'] = row['TRUTH']
            result['PREDICTION'] = match[0]
            results.append(result)

            if result['TRUTH'] == result['PREDICTION']:
                correct += 1
            elif result['TRUTH'] != result['PREDICTION']:
                wrong += 1

        test_results = pd.DataFrame(results)
        test_results.to_csv('test_results.csv')

        print()
        print(correct, 'correct classifications (', correct/len(test_results), '% )' )
        print(wrong, 'incorrect classifications')
        print(len(test_results), 'total cases' )
        print()

        print('testing completed..............................................')


    else:
        doc = sys.argv[1]
        result = D.classify(doc)
        print(result)





    #
