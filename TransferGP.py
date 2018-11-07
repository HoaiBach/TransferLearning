'''
Created on 21/09/2018

@author: nguyenhoai2
'''
import operator
import random
from deap import gp
from deap import creator, base, tools
import time
import FitnessFunction, GPUtility, Core


# GP parameters
POP_SIZE = 512
NGEN = 50

NO_SUB_IND = 1
INIT_MAX_DEPTH = 3
INIT_MIN_DEPTH = 2
MAX_DEPTH = 7
MIN_DEPTH = 2

GP_CXPB = 0.8
GP_MUTBP = 0.2
GP_CXSUB = 1.0
GP_MUTSUB = 1.0

GP_ELI = 1.0-GP_CXPB-GP_MUTBP

# Define new functions
def protectedDiv(left, right):
    return left/(right+0.0) if right != 0 else 1

def maxOperator(left, right):
    return left if left > right else right

def ifOperator(condition, left, right):
    return left if condition else right

def squareRoot(value):
    if value > 0:
        return value**0.5
    else:
        return 0

# Supervised - use labeled target instances
# Unsupervised - do not use labeled target instances
SUPERVISED = True

# Set up the terminal sets
pset = gp.PrimitiveSet("MAIN", Core.no_features)
pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1))

# Set up the operation
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)

# Fitness function: to minimize
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# Each individual is a list of tree
# Each individual has a fitness value
creator.create("Individual", list, fitness=creator.FitnessMin)

# Now defining the toolbox
toolbox = base.Toolbox()
# Each expr is one tree, it is initialized by halfAndHalf strategy
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=INIT_MIN_DEPTH, max_=INIT_MAX_DEPTH)
toolbox.register("sub_individual", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
toolbox.register("compile", gp.compile, pset=pset)
# Register genetic operators:
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Evaluate fitness of each individual
def evaluate(individual):
    for sub_ind in individual:
        if sub_ind.height > MAX_DEPTH:
            print("Too high in GP: "+individual.height)

    funcs = [toolbox.compile(expr=tree) for tree in individual]
    src_feature  = GPUtility.buildNewFeatures(Core.src_feature, funcs)
    tarU_feature = GPUtility.buildNewFeatures(Core.tarU_feature, funcs)
    tarL_feature = GPUtility.buildNewFeatures(Core.tarL_feature, funcs)

    if SUPERVISED:
        return FitnessFunction.fitness_function(src_feature=src_feature, src_label=Core.src_label,
                                                tarU_feature=tarU_feature, tarU_soft_label= Core.tarU_soft_label,
                                                classifier=Core.classifier,
                                                tarL_feature=tarL_feature, tarL_label=Core.tarL_label),
    else:
        return FitnessFunction.fitness_function(src_feature=src_feature, src_label=Core.src_label,
                                                tarU_feature=tarU_feature, tarU_soft_label=Core.tarU_soft_label,
                                                classifier=Core.classifier),

toolbox.register("evaluate", evaluate)

# to avoid floaing
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH))

def setWeight():
    if SUPERVISED:
        src_err, diff_marg, tar_err = FitnessFunction.domain_differece(src_feature=Core.src_feature, src_label=Core.src_label,
                                                                       classifier=Core.classifier,
                                                                       tarU_feature=Core.tarU_feature, tarU_soft_label=Core.tarU_soft_label,
                                                                       tarL_feature=Core.tarL_feature, tarL_label=Core.tarL_label)
    else:
        src_err, diff_marg, tar_err = FitnessFunction.domain_differece(src_feature=Core.src_feature, src_label=Core.src_label,
                                                                       classifier=Core.classifier,
                                                                       tarU_feature=Core.tarU_feature, tarU_soft_label=Core.tarU_soft_label)

    FitnessFunction.margWeight= 1.0 / diff_marg
    FitnessFunction.tarWeight = 1.0 / tar_err
    FitnessFunction.srcWeight = 1.0/src_err

def main(args):
    global SUPERVISED, NO_SUB_IND, toolbox

    #For elitism, at least the best individual
    #is recorded
    NO_ELI = (int)(POP_SIZE * GP_ELI)
    if NO_ELI < 10:
        NO_ELI = 10

    filename = "iteration"+str(args[0])+".txt"
    file = open(filename,'w+')

    run_index = int(args[0])
    supervised = int(args[1])

    if supervised == 0:
        SUPERVISED = False
    else:
        SUPERVISED = True

    #setWeight()

    NO_SUB_IND = int(args[2])

    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.sub_individual, n=NO_SUB_IND)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=POP_SIZE)

    random.seed(1617**2*run_index)

    #FitnessFunction.setWeight(src_feature=Core.src_feature, src_label=Core.src_label,
    #                          tarU_feature=Core.tarU_feature, tarU_label=Core.tarU_soft_label)
    time_start = time.clock()
    pop = toolbox.population()
    hof = tools.HallOfFame(NO_ELI)

    #evaluate the population
    fitness = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitness):
        ind.fitness.values = fit

    #Update the HoF
    hof.update(pop)

    towrite = "Supervised: %r \n" \
              "Number of sub tree: %d\n" \
              "Source weight: %f\n" \
              "Diff source and target weight: %f\n" \
              "Target weight: %g \n" % (SUPERVISED, NO_SUB_IND,
                                        FitnessFunction.srcWeight,
                                        FitnessFunction.margWeight,
                                        FitnessFunction.tarWeight)

    for gen in range(NGEN):
        print(gen)

        towrite = towrite + ("----Generation %i -----\n" %gen)

        #Select the next generation individuals
        #Leave space for elitism
        offspringS = toolbox.select(pop, len(pop)-NO_ELI)
        # Clone the selected individuals
        offspring = [toolbox.clone(ind) for ind in offspringS]

        #go through each individual
        for i in range(1, len(offspring), 2):
            if random.random() < GP_CXPB:
                #perform crossover for all the features
                first = offspring[i-1]
                second = offspring[i]
                first, second = crossoverEach(first, second)
                del first.fitness.values
                del second.fitness.values

        for i in range(len(offspring)):
            if random.random() < GP_MUTBP:
                parent = pop[i]
                for j in range(1, len(parent)):
                    if random.random() < GP_MUTSUB:
                        parent[j] = toolbox.mutate(parent[j])
                del parent.fitness.values

        #Now put HOF back to offspring
        for ind in hof:
            offspring.append(toolbox.clone(ind))

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        #Now update the hof for the next iteration
        hof.update(offspring)

        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        towrite = towrite + ("  Min %s\n" % min(fits))
        towrite = towrite + ("  Max %s\n" % max(fits))
        towrite = towrite + ("  Avg %s\n" % mean)
        towrite = towrite + ("  Std %s\n" % std)

        bestInd = hof[0]

        funcs = [toolbox.compile(expr=tree) for tree in bestInd]
        src_feature = GPUtility.buildNewFeatures(Core.src_feature, funcs)
        tarU_feature = GPUtility.buildNewFeatures(Core.tarU_feature, funcs)
        tarL_feature = GPUtility.buildNewFeatures(Core.tarL_feature, funcs)

        if SUPERVISED:
            src_err, diff_marg, tar_err = FitnessFunction.domain_differece(src_feature=src_feature, src_label=Core.src_label,
                                                                           classifier=Core.classifier,
                                                                           tarU_feature=tarU_feature, tarU_soft_label=Core.tarU_soft_label,
                                                                           tarL_feature=tarL_feature, tarL_label=Core.tarL_label)
        else:
            src_err, diff_marg, tar_err = FitnessFunction.domain_differece(src_feature=src_feature, src_label=Core.src_label,
                                                                           classifier=Core.classifier,
                                                                           tarU_feature=tarU_feature, tarU_soft_label=Core.tarU_soft_label)

        towrite = towrite + ("  Source Error: %f \n  Diff Marg: %f \n  Target Error: %f \n" %(src_err, diff_marg, tar_err))

        acc = 1.0 - FitnessFunction.classification_error(training_feature=src_feature, training_label=Core.src_label,
                                                         classifier=Core.classifier,
                                                         testing_feature=tarU_feature, testing_label=Core.tarU_label)
        towrite = towrite + ("  Accuracy on unlabel target: "+str(acc) + "\n")

        # Update the pseudo label and weight
        Core.classifier.fit(src_feature, Core.src_label)
        Core.tarU_soft_label = Core.classifier.predict(tarU_feature)
        #FitnessFunction.setWeight(Core.src_feature, Core.src_label, Core.tarU_feature, Core.tarU_SoftLabel)

    time_elapsed = (time.clock() - time_start)

    #process the result
    bestInd = hof[0]
    towrite = towrite + "----Final -----\n"

    funcs = [toolbox.compile(expr=tree) for tree in bestInd]
    src_feature = GPUtility.buildNewFeatures(Core.src_feature, funcs)
    tarU_feature = GPUtility.buildNewFeatures(Core.tarU_feature, funcs)
    acc = 1.0 - FitnessFunction.classification_error(training_feature=src_feature, training_label=Core.src_label,
                                                     classifier=Core.classifier,
                                                     testing_feature=tarU_feature, testing_label=Core.tarU_label)
    towrite = towrite + ("Accuracy on the target (TL): %f\n" % acc)
    towrite = towrite + "Accuracy on the target (No TL): %f\n" % (
                    1.0 - FitnessFunction.classification_error(training_feature=Core.src_feature, training_label=Core.src_label,
                                                               classifier=Core.classifier,
                                                               testing_feature=Core.tarU_feature, testing_label=Core.tarU_label))

    towrite = towrite + ("Computation time: %f\n" % time_elapsed)
    towrite = towrite + ("Number of features: %d\n" % len(bestInd))

    file.write(towrite)
    file.close()

def crossoverEach(first, second):
    for j in range(1,len(first)):
        if random.random() < GP_CXSUB:
            first[j], second[j] = toolbox.mate(first[j], second[j])
    return first, second

def crossverAll(first, second):
    return tools.cxOnePoint(first, second)

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
