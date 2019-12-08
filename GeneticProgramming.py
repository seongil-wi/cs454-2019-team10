import random,operator,math,numpy
from deap import creator, base, tools, algorithms,gp
from Reference.fitness import get_fitness

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def getFitness(individual,points): # I dont know what point is. And please tell me how to make function(individual) as input parameter
    func = toolbox.compile(expr=individual)
    
    model = "mnist_test_model_8_20_relu"  # We can choose models in DeepFault-Reference/neural_networks
    get_fitness(individual,model)
    
    #sqerrors = ((func(x) - x ** 4 - x ** 3 - x ** 2 - x) ** 2 for x in points)
    #return math.fsum(sqerrors) / len(points),

# Operator #
pset = gp.PrimitiveSet("main", 4)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)


# variable #
pset.renameArguments(ARG0='attr_n_as')
pset.renameArguments(ARG1='attr_n_af')
pset.renameArguments(ARG2='attr_n_ns')
pset.renameArguments(ARG3='attr_n_nf')

# Creator #
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)


toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=4, max_=6)


toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", getFitness, points=[x / 10. for x in range(-10, 10)])
toolbox.register("select", tools.selTournament, toursize=4)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=2, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    random.seed(40)

    pop = toolbox.population(n=40)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats, halloffame=hof, verbose=True)
    # print log
    return pop, log, hof


if __name__ == "__main__":
    main()


