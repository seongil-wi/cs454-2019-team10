# Spectrum based fault localization formulae for Neural Network 

We will propose SBFL formulae for Neural Network. We refer to FASE'2019 paper [DeepFault: Fault Localization for Deep Neural Networks](https://arxiv.org/abs/1902.05974).

## Abstract

Neural networks are a technology of great interest recently. In particular, it is applied to safety-critical technologies such as autonomous driving. In the meantime, neural network debugging issues have a great impact on people's safety and security.

While SBFL has shown good results in general code debugging, research has been conducted to apply SBFL to Neural network.

The study used fixed novel formuleas such as Tarantula and Ochiai, but we are actively working to make formuleas optimized for problems in the field of SBFL.

We will propose the formuleas by Genetic algorithm.

## Usage

Version: Python 3.6, Keras (v2.2.2) with Tensorflow(v1.10.1), deap(v1.3)

Test Model: mnist_test_model_8_20_relu, mnist_test_model_5_30_relu, mnist_test_model_3_50_relu
Test the number of suspicious neuron: 1, 2, 3, 4

GeneticProgramming.py --model [Model] -num_suspicious [the number of suspicious neuron]

For example, GeneticProgramming.py --model mnist_test_model_8_20_relu --num_suspicious 1
