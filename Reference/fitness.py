"""
This is the main file that executes the flow of DeepFault
"""
from test_nn import test_model
from os import path
from spectrum_analysis import *
from utils import save_perturbed_test_groups, load_perturbed_test_groups
from utils import load_suspicious_neurons, save_suspicious_neurons
from utils import create_experiment_dir, get_trainable_layers
from utils import load_classifications, save_classifications
from utils import save_layer_outs, load_layer_outs, construct_spectrum_matrices
from utils import load_MNIST, load_CIFAR, load_model
from utils import filter_val_set, save_original_inputs
from input_synthesis import synthesize
from sklearn.model_selection import train_test_split
import datetime
import argparse
import random

from keras import backend as K
from collections import defaultdict
import numpy as np
from utils import get_layer_outs
import math





def get_fitness(individual,model):
    
    experiment_path = "experiment_results"
    model_path = "neural_networks"
    group_index = 1
    __version__ = "v1.0"
    args = parse_arguments()
    model_name     = model
    dataset        = 'mnist'
    selected_class = 0
    step_size      = 1
    distance       = 0.1
    
    susp_num       = 1
    repeat         = 1
    seed           = random.randint(0,10)
    star           = 3
    

    ####################
    # 0) Load MNIST or CIFAR10 data
    if dataset == 'mnist':
        X_train, Y_train, X_test, Y_test = load_MNIST(one_hot=True)
    else:
        X_train, Y_train, X_test, Y_test = load_CIFAR()


    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                      test_size=1/6.0,
                                                      random_state=seed)

    logfile = open(logfile_name, 'a')


    ####################
    # 1) Load the pretrained network.
    try:
        model = load_model(path.join(model_path, model_name))
    except:
        logfile.write("Model not found! Provide a pre-trained model as input.")
        exit(1)


    #Fault localization is done per class.
    X_val, Y_val = filter_val_set(selected_class, X_test, Y_test)




    correct_classifications, misclassifications, layer_outs, predictions =\
            test_model(model, X_val, Y_val)



    trainable_layers = get_trainable_layers(model)
    scores, num_cf, num_uf, num_cs, num_us = construct_spectrum_matrices(model,
                                                                        trainable_layers,
                                                                        correct_classifications,
                                                                        misclassifications,
                                                                        layer_outs)



 
        
    suspicious_neuron_idx = individual_analysis(individual,trainable_layers, scores,
                                                 num_cf, num_uf, num_cs, num_us,
                                                 susp_num)



    perturbed_xs = []
    perturbed_ys = []

    # select 10 inputs randomly from the correct classification set.
    selected = np.random.choice(list(correct_classifications), 10)

    # zipped_data = zip(, )
    x_original = list(np.array(X_val)[selected])
    y_original = list(np.array(Y_val)[selected])

    # save_original_inputs(x_original, filename, group_index)

    syn_start = datetime.datetime.now()
    x_perturbed = synthesize(model, x_original, suspicious_neuron_idx, step_size, distance)
    syn_end = datetime.datetime.now()


    # 5) Test if the mutated inputs are adversarial
    score = model.evaluate([x_perturbed], [y_original], verbose=0)
    return score[0]


def individual_analysis(individual,trainable_layers, scores, num_cf, num_uf, num_cs, num_us, suspicious_num):
    '''
    More information on Tarantula fault localization technique can be found in
    [1]
    '''
    suspicious_neuron_idx = [[] for i in range(1, len(trainable_layers))]


    for i in range(len(scores)):
        for j in range(len(scores[i])):
            score = individual(num_cf[i][j],num_uf[i][j],num_cs[i][j],num_us[i][j])
            if np.isnan(score):
                score = 0
            scores[i][j] = score

    flat_scores = [float(item) for sublist in scores for item in sublist if not
               math.isnan(float(item))]

    relevant_vals = sorted(flat_scores, reverse=True)[:suspicious_num]

    suspicious_neuron_idx = []
    for i in range(len(scores)):
        for j in range(len(scores[i])):
            if scores[i][j] in relevant_vals:
                
                if trainable_layers == None:
                    suspicious_neuron_idx.append((i,j))
                else:
                    suspicious_neuron_idx.append((trainable_layers[i],j))
            if len(suspicious_neuron_idx) == suspicious_num:
                break

    return suspicious_neuron_idx

