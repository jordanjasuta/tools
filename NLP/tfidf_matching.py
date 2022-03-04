"""
tf-idf for doc/string matching

...........

Author: Jordan Jasuta Fischer
"""

# import necessary packages and tools
from sklearn.feature_extraction.text import TfidfVectorizer


class docMatcher:

    def __init__(self):
        # load vectorizer
        self.vectorizer = TfidfVectorizer()

        # load dict for term normalization
        self.term_normalization = []
        with open('term_normalization.csv', 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                self.term_normalization.append(line.split(','))

    def preprocess(self, text):

        # lower case
        text = text.lower()

        # spell check?

        # replace original terms with normalized terms
        for term_list in self.term_normalization:
            for term in term_list:
                if term in text:
                    text = text.replace(term, term_list[0])

        return text


    def get_sim_score(self, text1, text2):
        text1 = self.preprocess(text1)
        text2 = self.preprocess(text2)
        print('matching: ')
        print(text1)
        print(text2)

        vecs = self.vectorizer.fit_transform([text1, text2])
        corr_matrix = ((vecs * vecs.T).A)

        return corr_matrix[0][1]






if __name__ == '__main__':

    matcher = docMatcher()

    text1 = 'evaluation for ptsd'
    text2 = 'evaluation of posttraumatic stress disorder on March 4th'

    score = matcher.get_sim_score(text1, text2)
    print('match score: ', score)
