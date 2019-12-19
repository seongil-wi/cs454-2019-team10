import random, operator, math, numpy
from deap import creator, base, tools, algorithms, gp
from fitness import get_fitness
import csv
import numpy as np
from utils import load_MNIST, load_CIFAR, load_model
from os import path
from utils import filter_val_set
from test_nn import test_model
from utils import construct_spectrum_matrices
from utils import get_trainable_layers
import argparse
from tqdm import tqdm


def protectedDiv(left, right):
    if (right == 0):
        return 1
    else:
        return left / right


def getFitness(individual, m, X, Y, c, t, s, n_cf, n_uf, n_cs, n_us, sel, susp):
    func = toolbox.compile(expr=individual)
    # print(individual)

    return get_fitness(individual, m, X, Y, c, t, s, n_cf, n_uf, n_cs, n_us, sel, susp, func),


# def getFitness(individual):
#     func = toolbox.compile(expr=individual)
#     #print(individual)
#
#     return get_fitness(individual,func),

text = 'Spectrum Based Fault Localization for Deep Neural Networks'
parser = argparse.ArgumentParser(description=text)

# add new command-line arguments
parser.add_argument("--num_suspicious", help="number of suspicious neuron",
                    required=True)
parser.add_argument("--model", help="The model to be loaded. The \
                    specified model will be analyzed.", required=True)

# parse command-line arguments
args = parser.parse_args()

selected_class = 0
model_path = "neural_networks"
model_name = vars(args)['model']
model = load_model(path.join(model_path, model_name))
X_train, Y_train, X_test, Y_test = load_MNIST(one_hot=True)
X_val, Y_val = filter_val_set(selected_class, X_test, Y_test)
correct_classifications, misclassifications, layer_outs, predictions = \
    test_model(model, X_val, Y_val)
trainable_layers = get_trainable_layers(model)
scores, num_cf, num_uf, num_cs, num_us = construct_spectrum_matrices(model,
                                                                     trainable_layers,
                                                                     correct_classifications,
                                                                     misclassifications,
                                                                     layer_outs)

a = np.array(correct_classifications)
selected = a[:10]
# Operator #
pset = gp.PrimitiveSet("main", arity=4)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)

# variable #
pset.renameArguments(ARG0='attr_n_af')
pset.renameArguments(ARG1='attr_n_nf')
pset.renameArguments(ARG2='attr_n_as')
pset.renameArguments(ARG3='attr_n_ns')

# Creator #
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=4)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", getFitness, m=model, X=X_val, Y=Y_val, c=correct_classifications, t=trainable_layers,
                 s=scores, n_cf=num_cf, n_uf=num_uf, n_cs=num_cs, n_us=num_us, sel=selected,
                 susp=vars(args)['num_suspicious'])
# toolbox.register("evaluate", getFitness)
toolbox.register("select", tools.selTournament, tournsize=4)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


def main():
    random.seed(64)
    pop = toolbox.population(n=30)
    CXPB, MUTPB, NGEN = 0.9, 0.1, 1

    print("Start of evolution")

    f = open("output.csv", 'w')
    wr = csv.writer(f)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in tqdm(zip(pop, fitnesses), desc = "evalute the entire population"):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Begin the evolution
    for g in range(NGEN):
        print("-- Generation %i --" % g)

        currentBest = tools.selBest(pop, 1)[0]
        BestScore = currentBest.fitness.values
        print("Current Best, Current BestScore are %s, %s", currentBest, BestScore)


        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))


        wr.writerow(tools.selBest(pop, 1)[0])
        wr.writerow(BestScore)

        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in tqdm(offspring, desc="mutating..."):

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    print("-- End of (successful) evolution --")

    x_perturbed = list(np.array(X_val)[selected])
    y_original = list(np.array(Y_val)[selected])

    fig = plt.figure()
    rows = 2
    cols = 5

    for i in range(1, 11):
        image = np.reshape(x_perturbed[i-1][:][:], [28,28])
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(image, cmap='Greys')
        ax.set_xlabel(str(what_number(y_original[i-1])))
        ax.set_xticks([]), ax.set_yticks([])
    plt.show()

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))


if __name__ == "__main__":
    main()
