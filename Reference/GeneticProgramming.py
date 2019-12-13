import random, operator, math, numpy
from deap import creator, base, tools, algorithms, gp
from fitness import get_fitness
from utils import load_MNIST, load_CIFAR, load_model
from os import path
from utils import filter_val_set
from test_nn import test_model
from utils import construct_spectrum_matrices
from utils import get_trainable_layers
def protectedDiv(left, right):
    if (right == 0):
        return 1
    else:
        return left / right


# def getFitness(individual,m,X,Y,c,t,s,n_cf,n_uf,n_cs,n_us):
#     func = toolbox.compile(expr=individual)
#     #print(individual)
#     
#     return get_fitness(individual, m,X,Y,c,t,s,n_cf,n_uf,n_cs,n_us, func),

def getFitness(individual):
    func = toolbox.compile(expr=individual)
    #print(individual)
    
    return get_fitness(individual,func),


       
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
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=4, max_=4)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# toolbox.register("evaluate", getFitness, m = model, X=X_val,Y=Y_val,c=correct_classifications,t=trainable_layers,s=scores,n_cf=num_cf,n_uf=num_uf,n_cs=num_cs,n_us=num_us)
toolbox.register("evaluate", getFitness)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=2, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


def main():
    random.seed(64)
    pop = toolbox.population(n=40)
    CXPB, MUTPB, NGEN = 1.0, 0.1, 40

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Begin the evolution
    for g in range(NGEN):
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        print("Current Best", tools.selBest(offspring, 1)[0])

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

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

        # # Gather all the fitnesses in one list and print the stats
        # fits = [ind.fitness.values[0] for ind in pop]
        #
        # length = len(pop)
        # mean = sum(fits) / length
        # sum2 = sum(x * x for x in fits)
        # std = abs(sum2 / length - mean ** 2) ** 0.5
        #
        # print("  Min %s" % min(fits))
        # print("  Max %s" % max(fits))
        # print("  Avg %s" % mean)
        # print("  Std %s" % std)

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))



if __name__ == "__main__":
    main()
