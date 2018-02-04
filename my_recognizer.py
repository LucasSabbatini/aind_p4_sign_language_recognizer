import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # PROBABILITIES
    for i in range(len(test_set.wordlist)):
      sample_frames , sample_length = test_set._hmm_data[i][0], test_set._hmm_data[i][1]
      sample_dict = {}
      # scoring for every model
      for word in models:
        try:
          score = models[word].score(sample_frames, sample_length)
          sample_dict[word] = models[word].score(sample_frames, sample_length)
        except:
          sample_dict[word] = float('-inf')
      probabilities.append(sample_dict)

    # GUESSES
    for sample in probabilities:
      biggest_prob = float('-inf')
      best_guess = ''
      for word, prob in sample.items():
        if prob > biggest_prob:
          biggest_prob = prob
          best_guess = word
      guesses.append(best_guess)
    return probabilities, guesses


