import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class TrainingDataCV(object):
    """
    Object for storing data from training using SelectorCV

    Attributes:
    self.hmmsize_cv : list, cross_val mean values for every number of states
    self.best_hmm : int, best number of states
    self.highest_mean : float, cross_val mean associated with beat_hmm
    """
    def __init__(self, hmmsize_cv):
        self.hmmsize_cv = hmmsize_cv
        highest_mean, best_hmm = float('-inf'), 0
        for hmm_size in self.hmmsize_cv:
            if hmm_size[1] > highest_mean:
                highest_mean = hmm_size[1]
                best_hmm = hmm_size[0]
        self.best_hmm = best_hmm
        self.highest_mean = highest_mean

    def get_best_hmm_size(self):
        return self.best_hmm

class TrainingDataBIC(object):
        """
    Object for storing data from training using SelectorCV

    Attributes:
    self.hmmsize_cv : list, bic values for every number of states
    self.best_hmm : int, best number of states
    self.lowesT_bic : float, bic value associated with beat_hmm
    """
    def __init__(self, hmmsize_bic):
        self.hmmsize_bic = hmmsize_bic
        lowest_bic, best_hmm = float('inf'), 0
        for hmm_size in self.hmmsize_bic:
            if hmm_size[1] < lowest_bic:
                lowest_bic = hmm_size[1]
                best_hmm = hmm_size[0]
        self.best_hmm = best_hmm
        self.lowest_bic = lowest_bic     

    def get_best_hmm_size(self):
        return self.best_hmm

    def get_lowest_bic(self):
        return self.lowest_bic

class TrainingDataDIC(object):
    """
    Object for storing data from training using SelectorCV

    Attributes:
    self.hmmsize_dic : list, dic values for every number of states
    self.best_hmm : int, best number of states
    self.highest_dic : float, dic value associated with beat_hmm
    """
    def __init__(self, DIC_values):
        self.hmmsize_dic = DIC_values
        highest_dic, best_hmm = float('-inf'), 0
        for dic in DIC_values:
            if dic[1] > highest_dic:
                highest_dic = dic[1]
                best_hmm = dic[0]
        self.best_hmm = best_hmm
        self.highest_dic = highest_dic

    def get_best_hmm_size(self):
        return self.best_hmm

    def get_highest_dic(self):
        return self.highest_mean




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
        """
        Model selector method for Bayesian Information Criterion. for each hmm_size
        a model is created, score and BIC computed. The model with the lowest value for the BIC
        is then returned.

        Param
        -----
        return_data : bool for returning data

        Return
        ______
        if return_false:
            model : 
        else:
            (model, data) : 


        """
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
                bic = -2 * logL + p * np.log(N)
                hmmsize_bic.append([hmm_size, bic])
            except:
                # print('Model {} did not score for own word: {}'.format(hmm_size, self.this_word))
                pass

        bic_data = TrainingDataBIC(hmmsize_bic)
        model = self.base_model(bic_data.best_hmm)

        if model is not None:
            if return_data:
                return model, bic_data
            else:
                return model

        else:
            print('word {} with {} examples has no model'.format(self.this_word, len(self.sequences)))  
            return model


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
        

        Return
        ______
        if return_false:
            model : 
        else:
            (model, data) : 
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        word_list = self.words.keys()
        components_range = range(self.min_n_components, self.max_n_components+1)
        DIC_values = []
        models = {}

        for hmm_size in components_range:
            try:
                models[hmm_size] = self.base_model(hmm_size)
                this_word_score = models[hmm_size].score(self.X, self.lengths)
            except:
                models[hmm_size] = None

        for hmm_size in models:
            other_words_score = []
            if models[hmm_size] is not None:
                for word in word_list:
                    if word != self.this_word:
                        word_X, word_lengths = self.hwords[word]
                        logL = models[hmm_size].score(word_X, word_lengths)
                        other_words_score.append(logL)
                average_other_words = sum(other_words_score)/float((len(word_list)-1))
                DIC = this_word_score - average_other_words
                DIC_values.append([hmm_size, DIC, this_word_score, other_words_score, average_other_words]) 

        dic_data = TrainingDataDIC(DIC_values)

        if return_data:
            return models[dic_data.best_hmm], dic_data
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

    def select(self, return_data=False):
        """
        For each word and number states (hmm_size), this method will compute the logL
        value as the average of the cross validation values, and then return the model with 
        the highest logL mean value.

        Return
        ______
        if return_false:
            model : 
        else:
            model, cv_data: 

        """
        components_range = range(self.min_n_components, self.max_n_components+1)
        hmmsize_cv = []

        non_cross_val = False

        for hmm_size in components_range:
            logL_kfold = []

            if len(self.lengths) >= 3:
                split_method = KFold(random_state=self.random_state)
                # try:

                split = 0

                for train_idxs, test_idxs in split_method.split(self.sequences):
                    # retrieving sequences from idxs
                    train_sequences = [self.sequences[idx] for idx in train_idxs]
                    test_sequences = [self.sequences[idx] for idx in test_idxs]
                    # creating lengths again
                    train_lengths = [len(sequence) for sequence in train_sequences]
                    test_lengths = [len(sequence) for sequence in test_sequences]

                    train_sequences = [features for sequence in train_sequences for features in sequence]
                    test_sequences = [features for sequence in test_sequences for features in sequence]

                    try:
                        model = self.base_model(hmm_size, train_sequences, train_lengths)
                        logL  = model.score(test_sequences, test_lengths)
                        logL_kfold.append(logL)
                        
                    except:
                        pass

                    split += 1
                if len(logL_kfold) == 3:
                    hmmsize_cv.append([hmm_size, np.array(logL_kfold).mean()])
                
            else:
                non_cross_val = True
                try:
                    model = self.base_model(hmm_size, self.X, self.lengths)
                    logL = model.score(self.X, self.lengths)
                    hmmsize_cv.append([hmm_size, logL])
                except:
                    pass

        cv_data = TrainingDataCV(hmmsize_cv) 
        model = self.base_model(cv_data.best_hmm, self.X, self.lengths)
        
        if model is not None:
            if return_data:
                return model, cv_data
            else:
                return model

        else:
            print('No model for word {} was creates'.format(self.this_word))
            print('Number of examples for word {}'.format(len(self.sequences)))
            if return_data:
                return None, None
            else:
                return None



