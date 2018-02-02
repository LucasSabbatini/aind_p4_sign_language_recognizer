import warnings
from hmmlearn.hmm import GaussianHMM

def train_a_word(word, num_hidden_states, features):
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    training = asl.build_training(features)  # WordsData object
    X, lengths = training.get_word_Xlengths(word)
    model = GaussianHMM(n_components=num_hidden_states, n_iter=1000).fit(X, lengths)
    logL = model.score(X, lengths)
    return model, logL

def show_model_stats(word, model):
    print("Number of states trained in model for {} is {}".format(word, model.n_components))    
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])
    for i in range(model.n_components):  # for each hidden state
        print("hidden state #{}".format(i))
        print("mean = ", model.means_[i])
        print("variance = ", variance[i])
        print()


# lucas

def build_models(word_list, hmm_hidden_states, features):
    logls = []
    for word in word_list:
        model, logL = train_a_word(word, hmm_hidden_states, features) # Experiment here with different parameters
        print("logL = {} for word {} and {} hidden states".format(logL, word, hmm_hidden_states))
        logls.append(logL)
    mean_logl_state = np.array(logls).mean()
    std_logl_state = np.array(logls).std()
    return mean_logl_state, std_logl_state

def varying_hmm_states(word_list, hmm_states_list, features):
    data = []
    for hmm_state in hmm_states_list:
        print("Models with {} hidden states".format(hmm_state))
        mean_logl, std_logl = build_models(word_list, hmm_state, features)
        data.append([mean_logl, std_logl, hmm_state])
    return data

def run_simulation(word_list, hmm_hidden_list, features_lists):
    data = []
    for features in features_lists:
        print("Starting with features: {}".format(features))
        features_data = varying_hmm_states(word_list, hmm_hidden_list, features)
        data.append([features_data, features])
    return data

def simulation_stats(sim_data):
    for features in sim_data:
        print("features tested: {}".format(features[1]))
        for hidden_states in features[0]:
            print("hidden states: {}".format(hidden_states[-1]))
            print("mean and std: {} and {}".format(hidden_states[0], hidden_states[1]))
        means = [features[0][i][0] for i in range(len(features[0]))]
        stds = [features[0][i][1] for i in range(len(features[0]))] 
        total_mean = np.array(means).mean()
        total_std = np.array(stds).mean()
        print("Overall mean and deviation for this set of features: {} and {}".format(total_mean, total_std))
        print('\n')





