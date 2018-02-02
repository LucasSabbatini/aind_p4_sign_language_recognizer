"""
"""

import numpy as np 
import pandas as pd 
from asl_data import AsLDb
import warnings
from hmmlearn.hmm import GaussianHMM
import math
from matplotlib import (cm, pyplot as plt, mlab)

asl = AslDb() # initialize the database
asl.df.head() # Displays the first five rows of the asl database, indexes by video and frame
asl.df.ix[98,1] # look at the data available for an individual frame

# FEATURE SELECION FOR TRAINING THE MODEL

# add ground features 
features_ground = ['grnd-ry', 'grnd-rx', 'grnd-ly', 'grnd-lx']
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

# means and std
df_means = asl.df.gourpby('speaker').mean()
df_stds = asl.df.groupby('speaker').std()

# add means and std to asl dataframe
means = ['left-x-mean', 'left-y-mean', 'right-x-mean', 'right-y-mean']
stds = ['left-x-std', 'left-y-std', 'right-x-std', 'right-y-std']
original_features = ['left-x', 'left-y', 'right-x', 'right-y']

for i in range(len(means)):
    asl.df[means[i]] = asl.df['speaker'].map(df_means[original_features[i]])
    
for i in range(len(stds)):
    asl.df[stds[i]] = asl.df['speaker'].map(df_std[original_features[i]])

# add normalized features
features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']
features_to_norm = ['right-x', 'right-y', 'left-x', 'left-y']
means_norm = ['right-x-mean', 'right-y-mean', 'left-x-mean', 'left-y-mean']
stds_norm = ['right-x-std', 'right-y-std', 'left-x-std', 'left-y-std']

for i in range(len(features_norm)):
	asl.df[features_norm[i]] = (asl.df[features_to_norm[i]] - asl.df[means_norm[i]])/asl.df[stds_norm[i]]

# add polar features
features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']

right_x = asl.df['grnd-rx']
right_y = asl.df['grnd-ry']
asl.df['polar-rr'] = np.sqrt(right_x**2 + right_y**2)
asl.df['polar-rtheta'] = np.arctan2(right_x, right_y)

left_x = asl.df['grnd-lx']
left_y = asl.df['grnd-ly']
asl.df['polar-lr'] = np.sqrt(left_x**2 + left_y**2)
asl.df['polar-ltheta'] = np.arctan2(left_x, left_y)

# add delta features
features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']

for i in range(len(features_delta)):
    asl.df[features_delta[i]] = asl.df[features_to_norm[i]].diff()
    asl.df[features_delta[i]].fillna(0, inplace=True)

# update means and std
df_means = asl.df.groupby('speaker').mean()
df_stds = asl.df.groupby('speaker').std()

# add normalized polar
means = ['polar-lr-mean', 'polar-ltheta-mean', 'polar-rr-mean', 'polar-rtheta-mean']
stds = ['polar-lr-std', 'polar-ltheta-std', 'polar-rr-std', 'polar-rtheta-std']
polar_features = ['polar-lr', 'polar-ltheta', 'polar-rr', 'polar-rtheta']
polar_norm = ['norm-lr', 'norm-ltheta', 'norm-rr', 'norm-rtheta']

for i in range(len(means)):
    asl.df[means[i]] = asl.df['speaker'].map(df_means[polar_features[i]])
    
for i in range(len(stds)):
    asl.df[stds[i]] = asl.df['speaker'].map(df_stds[polar_features[i]])

for i in range(len(polar_norm)):
    asl.df[polar_norm[i]] = (asl.df[polar_features[i]] - asl.df[means[i]])

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


# features lists
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']
features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']
polar_features = ['polar-lr', 'polar-ltheta', 'polar-rr', 'polar-rtheta']
features_custom = ['norm-lr', 'norm-ltheta', 'norm-rr', 'norm-rtheta']
features_custom2 = ['norm-lr', 'norm-rr']

wordlist = ['CAN', 'WRITE', 'BLAME', 'YESTERDAY', 'CHOCOLATE'] #, 'HERE', 'THINK', 'OLD', 'SHOULD', 'FUTURE']       
hmm_list = range(4,7)
features_lists = [features_ground, features_delta, features_norm, polar_features, features_custom, features_custom2]

# visualization
def visualize(word, model):
    """ visualize the input model for a particular word """
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)]) # num_state X num_features
    figures = []
    # iterating over individual feature
    for parm_idx in range(len(model.means_[0])):
    	# model._means is of dimension num_state X num_features
        xmin = int(min(model.means_[:,parm_idx]) - max(variance[:,parm_idx]))
        xmax = int(max(model.means_[:,parm_idx]) + max(variance[:,parm_idx]))
        fig, axs = plt.subplots(model.n_components, sharex=True, sharey=False)
        colours = cm.rainbow(np.linspace(0, 1, model.n_components))
        # iterating over states
        for i, (ax, colour) in enumerate(zip(axs, colours)):
            x = np.linspace(xmin, xmax, 100)
            mu = model.means_[i,parm_idx]
            sigma = math.sqrt(np.diag(model.covars_[i])[parm_idx])
            ax.plot(x, mlab.normpdf(x, mu, sigma), c=colour)
            ax.set_title("{} feature {} hidden state #{}".format(word, parm_idx, i))

            ax.grid(True)
        figures.append(plt)
    for p in figures:
        p.show()













