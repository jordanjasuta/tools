# -*- coding: utf-8 -*-
"""
Created on Thu Aug  13 11:30:05 2020

@author: Jordan Jastua Fischer
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import datefinder
from datetime import datetime
import sys

class DateFeatureExtractor:
    def __init__(self, dictionary_path = 'dictionaries', verbose=False):
        self.features = list()
        dictionary_path=dictionary_path

        df = pd.read_csv(dictionary_path + '/doc_types.csv', header=0)
        self.forms = set(list(df['DOC_TYPE']))


    def extract_dates(self, doc):
        dates = list()
        text = ''
        # print(self.forms)
        for word in doc.split():
            # print(word)
            if word not in self.forms:
                text += (word + ' ')
        # print('xxxxxxxxxxxxx', text)
        try:                     #### errors should just be skipped, package doesn't undertand that some things are just not dates and that's ok
            matches = datefinder.find_dates(text)
        except:
            pass

        for match in matches:
            dates.append(match)

        return dates


    def fit(self, docs):
        indptr = [0]
        indices = list()
        data = list()
        feats = dict()
        for doc in docs:
            # print(doc)
            index = feats.setdefault('dates', len(feats))
            indices.append(index)
            data.append(self.extract_dates(doc))
            indptr.append(len(indices))

        csr = csr_matrix((data, indices, indptr), dtype=int)
        print(csr)
        return csr


    def transform(self, docs):
        # X = np.zeros((len(docs), len(self.features)), dtype=int)


        for y, doc in enumerate(docs):
            print('........DOC:', doc)
            dates = self.extract_dates(doc)
            X = np.zeros((len(docs), len(dates)), dtype=int)
            print('........DATES:', dates)
            position = -1

            for date in dates:
                position += 1
                months_ago = ((date-datetime.now()).days)/30
                x = int(months_ago)
                print('......',y,x)
                # print(type(x))

                # X[y, 0] = x
                X[y, 0+position] = x
                print(X[0,])
                # X[y, x] += 1  # To match fit

                # if self.verbose:
                print('months_ago', x)




            # X[y, 0] = len(doc)
            # X[y, 0] = self.extract_dates(doc)

        return X



if __name__ == "__main__":
    fe = DateFeatureExtractor()
    dates = fe.extract_dates([sys.argv[1]])
    print(dates)
    # toda = datetime.now()
    # print(type(toda))
