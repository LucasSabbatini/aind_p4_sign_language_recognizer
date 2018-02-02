import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """


    def select(self, return_data=False):
        
        components_range = range(self.min_n_components, self.max_n_components+1)
        hmmsize_bic = []

        N = len(self.sequences)

        for hmm_size in components_range:
            # calculating p using the equation p = m^2 + km - 1.
            # sourcce: https://rdrr.io/cran/HMMpa/man/AIC_HMM.html
            # considering the underlying distribuition is normal, k = 2
            p = hmm_size**2 + 2*hmm_size - 1

            # creating model
            try:
                model = self.base_model(hmm_size)
                logL  = model.score(self.X, self.lengths)
            except:
                print('Did not work for word {}'.format(self.this_word))
                return None

            bic = -2 * logL + p * np.log(N)

            hmmsize_bic.append([hmm_size, bic])

        lowest_bic, n_components = float('inf'), 0
        for hmm_bic in hmmsize_bic:
            if hmm_bic[1] < lowest_bic:
                lowest_bic = hmm_bic[1]
                n_components = hmm_bic[0]

        print('Lowest BIC = {}. Best number of states = {}'.format(lowest_bic, n_components))

        self.n_components = n_components

        model = self.base_model(n_components)
        
        if model is not None:
            if return_data:
                return model, hmmsize_bic, (n_components, lowest_bic)
            else:
                return model

        else:
            print('model is not working')


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self, return_data=False):
        """
        This selector takes into account the whole task, and not just the recognition
        of one pattern. It is a statiscal model on all classes, measuring the difference
        between the probability of regocnizing the right word and the average probability
        of recognizing the other words.

        It does not select the model with highest probability of recognizing the right
        word, but the one with the largest difference between the right and the other.
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # DIC CALCULATION:
        # DIC = logL - sum(logL_for_other_words)/(len(self.words)-1)

        word_list = self.words.keys()

        components_range = range(self.min_n_components, self.max_n_components+1)

        DIC_values = []
        models = {}

        for hmm_size in components_range:
            try:
                models[hmm_size] = self.base_model(hmm_size)
                this_word_score = models[hmm_size].score(self.X, self.lengths)
            except:
                print('Model did not score for own word: {}'.format(self.this_word))
                models[hmm_size] = None

        # print(models)
        for hmm_size in models:
            other_words_score = []
            if models[hmm_size] is not None:
                for word in word_list:
                    if word != self.this_word:
                        word_X, word_lengths = self.hwords[word]
                        logL = models[hmm_size].score(word_X, word_lengths)
                        other_words_score.append(logL)
                        # not storing data about which word
                average_other_words = sum(other_words_score)/(len(word_list)-1)
                DIC = this_word_score - average_other_words
                DIC_values.append([hmm_size, DIC, this_word_score, other_words_score, average_other_words]) 
                # print('n_components = {}. this_word_score = {}. Other words average: {}. DIC = {}'.format(hmm_size, this_word_score, average_other_words, DIC))                                             
                # print("other word scores: {}".format(other_words_score))
                # print('Model could not compute score for own word: {}'.format(self.this_word))

        # now we have the values of DIC for every topology
        highest_dic = float('-inf')
        for dic in DIC_values:
            if dic[1] > highest_dic:
                highest_dic = dic[1]
                best_hmm = dic[0]

        if return_data:
            return models[best_hmm], DIC_values
        else:
            return models[best_hmm]


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def base_model(self, num_states, X, lengths):
        """
        Over ride ModelSelector base_model method
        """
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X, lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


    def select(self):
        # warnings.filterwarnings("ignore", category=DeprecationWarning)

        
        """Model Selector recieves all_sequences dict, all_Xlengths sequences dict, 
        specific word and other parameters not relevant for CV
        """
        components_range = range(self.min_n_components, self.max_n_components+1)
        hmmsize_cross_mean = []

        split_method = KFold(random_state=self.random_state)

        for hmm_size in components_range:
            logl_kfold = []

            # cross folding validation with current hmm size
            if len(self.sequences) > 2:
                for train_idxs, test_idxs in split_method.split(self.sequences):
                    # sequences is a list of examples
                    train_sequences = [self.sequences[idx] for idx in train_idxs]
                    test_sequences = [self.sequences[idx] for idx in test_idxs]
                    train_lengths = [len(sequence) for sequence in train_sequences]
                    test_lengths = [len(sequence) for sequence in test_sequences]

                    # concatenating sequences
                    train_sequences = [features for sequence in train_sequences for features in sequence]
                    test_sequences = [features for sequence in test_sequences for features in sequence]

                    # creating model
                    model = self.base_model(hmm_size, train_sequences, train_lengths)
                    logL  = model.score(test_sequences, test_lengths)

                    # appending logL to list
                    logl_kfold.append(logL)
            else:
                print('word {} has lass than 3 examples. Cannot split'.format(self.this_word))
                return None
            hmmsize_cross_mean.append([hmm_size, np.array(logl_kfold).mean()])

        highest_mean, n_components = float('-inf'), 0
        for hmm_size in hmmsize_cross_mean:
            if hmm_size[1] > highest_mean:
                highest_mean = hmm_size[1]
                n_components = hmm_size[0]

        print('Highest mean = {}. Best number of states = {}'.format(highest_mean, n_components))

        self.n_components = n_components

        model = self.base_model(n_components, self.X, self.lengths)
        
        if model is not None:
            return model

        else:
            print('model is not working')






