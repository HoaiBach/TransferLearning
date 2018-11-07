'''
Created on 24/09/2018

@author: nguyenhoai2
'''
import operator
import random
from deap import base
from deap import creator
from deap import tools
import Core
import FitnessFunction
import  time
from SpecialFitness import SpecialFitness

NBIT = Core.no_features
rate=1
NGEN = 100*rate
NPART = NBIT if NBIT <100 else 100
NPART = NPART/rate
is_low = 0
is_up  = 10.0/NBIT
ustks_low = NGEN/100.0
ustks_up  = 8*NGEN/100.0
pg_rate = 2.0
threshold = 0.6

i_stick = is_up
i_gbest = (1-i_stick)/(pg_rate+1)
i_pbest = pg_rate * i_gbest

ustks = ustks_low

TEST = False

SUPERVISED = False

creator.create("FitnessMin", SpecialFitness, weights=(-1.0, -1.0, -1.0, -1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, stk=list, best=None)


def generate(size):
    part = creator.Particle(1 if random.uniform(0, 1) > threshold else 0 for _ in range(size))
    part.stk = [1 for _ in range(size)]
    return part


def updateParticle(part, best):
    # find flipping probability
    stick_part = map(lambda x: i_stick*(1-x), part.stk)
    diff_pbest = map(operator.abs, map(operator.sub, part.best, part))
    pbest_part = map(lambda x: i_pbest * x, diff_pbest)
    diff_gbest = map(operator.abs, map(operator.sub, best, part))
    gbest_part = map(lambda x: i_gbest * x, diff_gbest)

    if TEST:
        print("Stick", stick_part)
        print("diff_pbest", pbest_part)
        print("diff_gbest", gbest_part)

    flipping = map(operator.add, stick_part, map(operator.add, pbest_part, gbest_part))

    # update the position -> update the stickiness
    for i, prob in enumerate(flipping):
        if random.random() < prob:
            # flip, change stickness back to 0:
            part[i] = 1 - part[i]
            part.stk[i] = 1
        else:
            # if the bit is not flipped, update stickiness
            part.stk[i] = max(0, part.stk[i] - 1.0/ustks)


def evaluate(particle):
    # Find all selected features
    indices = [index for index, entry in enumerate(particle) if entry == 1.0]
    # Build new dataset with selected features
    src_feature = Core.src_feature[:, indices]
    tarU_feature = Core.tarU_feature[:, indices]
    tarL_feature = Core.tarL_feature[:, indices]

    if SUPERVISED:
        return FitnessFunction.fitness_function(src_feature=src_feature, src_label=Core.src_label,
                                                tarU_feature=tarU_feature,
                                                classifier=Core.classifier,
                                                tarL_feature=tarL_feature, tarL_label=Core.tarL_label)
    else:
        return FitnessFunction.fitness_function(src_feature=src_feature, src_label=Core.src_label,
                                                tarU_feature=tarU_feature,
                                                classifier=Core.classifier)


toolbox = base.Toolbox()
toolbox.register("particle", generate, size=NBIT)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle)
toolbox.register("evaluate", evaluate)


def setWeight():
    if SUPERVISED:
        src_err, diff_marg, tar_err = FitnessFunction.domain_differece(src_feature=Core.src_feature, src_label=Core.src_label,
                                                                       classifier=Core.classifier,
                                                                       tarU_feature=Core.tarU_feature,
                                                                       tarL_feature=Core.tarL_feature, tarL_label=Core.tarL_label)
    else:
        src_err, diff_marg, tar_err = FitnessFunction.domain_differece(src_feature=Core.src_feature, src_label=Core.src_label,
                                                                       classifier=Core.classifier,
                                                                       tarU_feature=Core.tarU_feature)

    print(src_err, diff_marg, tar_err)
    if diff_marg == 0:
        FitnessFunction.margWeight = 0
    else:
        FitnessFunction.margWeight = 1.0/diff_marg

    if tar_err == 0:
        FitnessFunction.tarWeight = 0
    else:
        FitnessFunction.tarWeight = 1.0 / tar_err

    if src_err == 0:
        FitnessFunction.srcWeight = 0
    else:
        FitnessFunction.srcWeight = 1.0/src_err


# args[1] refers to which measure is used for diffCond
#  it defines which diffCond 1-gecco, 2-wrapper, 3-mmd
def main(args):
    global i_stick, i_pbest, i_gbest, ustks, SUPERVISED

    run_index = int(args[0])
    random.seed(1617 ** 2 * run_index)
    filename = "iteration"+str(args[0])+".txt"
    file = open(filename, 'w+')

    time_start = time.clock()

    SUPERVISED = False
    #supervised = int(args[1])

    #if supervised == 0:
    #    SUPERVISED = False
    #else:
    #    SUPERVISED = True

    cond_index = int(args[1])
    FitnessFunction.tarVersion = cond_index

    #setWeight()
    #FitnessFunction.setWeight(Core.src_feature, Core.src_label, Core.tarU_feature, Core.tarU_soft_label)

    # Set the weight for each components in the fitness function
    #FitnessFunction.setWeight(src_feature=Core.src_feature, src_label=Core.src_label,
    #                          tarU_feature=Core.tarU_feature, tarU_label=Core.tarU_soft_label)
    FitnessFunction.srcWeight = 0.0
    FitnessFunction.margWeight = 1.0
    FitnessFunction.tarWeight = 0.0

    # Initialize population and the gbest
    pop = toolbox.population(n=NPART)
    best = None

    toWrite = ("Supervised: %r \n" \
              "Source weight: %f\n" \
              "Diff source and target weight: %f\n" \
              "Target weight: %g\n" \
              "Conditional version: %d\n" % (SUPERVISED,
                                        FitnessFunction.srcWeight,
                                        FitnessFunction.margWeight,
                                        FitnessFunction.tarWeight,
                                        FitnessFunction.tarVersion))

    for g in range(NGEN):
        print(g)
        toWrite += ("=====Gen %d=====\n" % g)

        for part in pop:
            # Evaluate all particles
            part.fitness.values = toolbox.evaluate(part)

            if part.best is None or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values

            # update gbest
            if best is None or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values

        if TEST:
            print("is=", i_stick, "ip=", i_pbest, "ig=", i_gbest, "ustks=", ustks )
            print("best=", best)
            print(best.fitness.values)
            print("\n")
            for i, part in enumerate(pop):
                print("Particle %d: " % i)
                print("Particle position:",part)
                print("Particle pbest:",part.best)
                print("Particle stickiness:",part.stk)
                print("\n")


        # now update the position of each particle
        for part in pop:
            toolbox.update(part, best)

        # Gather all the fitness components of the gbest and print the stats
        indices = [index for index, entry in enumerate(best) if entry == 1.0]
        src_feature = Core.src_feature[:, indices]
        tarU_feature = Core.tarU_feature[:, indices]
        tarL_feature = Core.tarL_feature[:, indices]
        if SUPERVISED:
            src_err, diff_marg, tar_err = FitnessFunction.domain_differece(src_feature=src_feature, src_label=Core.src_label,
                                                                           classifier=Core.classifier,
                                                                           tarU_feature=tarU_feature,
                                                                           tarL_feature=tarL_feature, tarL_label=Core.tarL_label)
        else:
            src_err, diff_marg, tar_err = FitnessFunction.domain_differece(src_feature=src_feature, src_label=Core.src_label,
                                                                           classifier=Core.classifier,
                                                                           tarU_feature=tarU_feature)

        toWrite += ("  Source Error: %f \n  Diff Marg: %f \n  Target Error: %f \n" %(src_err, diff_marg, tar_err))
        toWrite += ("  Fitness function of real best: %f\n" % best.fitness.values[0])
        acc = 1.0 - FitnessFunction.classification_error(training_feature=src_feature, training_label=Core.src_label,
                                                         classifier=Core.classifier,
                                                         testing_feature=tarU_feature, testing_label=Core.tarU_label)
        toWrite += ("  Accuracy on unlabel target: " + str(acc) + "\n")
        toWrite += "  Position:"+str(best)+"\n"


        # update the parameters
        i_stick = is_up - (is_up - is_low)*(g+1)/NGEN
        i_gbest = (1-i_stick)/(pg_rate+1)
        i_pbest = pg_rate*i_gbest
        ustks   = ustks_low + (ustks_up-ustks_low)*(g+1)/NGEN

        # Update the pseudo label (only when the cond_index is equal to 2)
        if cond_index == 3 & g % 10==0:
            Core.classifier.fit(src_feature, Core.src_label)
            Core.tarU_soft_label = Core.classifier.predict(tarU_feature)
            FitnessFunction.set_weight(src_feature, Core.src_label, tarU_feature, Core.tarU_soft_label)
            # Need to update the fitness value of best and pbest again
            best.fitness.values = FitnessFunction.fitness_function(src_feature, Core.src_label,
                                                                   tarU_feature, Core.tarU_soft_label,
                                                                   Core.classifier),
            for part in pop:
                indices = [index for index, entry in enumerate(part.best) if entry == 1.0]
                p_src_feature = Core.src_feature[:, indices]
                p_tarU_feature = Core.tarU_feature[:, indices]
                part.best.fitness.values = FitnessFunction.fitness_function(p_src_feature, Core.src_label,
                                                                            p_tarU_feature, Core.tarU_soft_label,
                                                                            Core.classifier),

    time_elapsed = (time.clock() - time_start)
    toWrite += "----Final -----\n"
    indices = [index for index, entry in enumerate(best) if entry == 1.0]
    src_feature = Core.src_feature[:, indices]
    tarU_feature = Core.tarU_feature[:, indices]
    acc = 1.0 - FitnessFunction.classification_error(training_feature=src_feature, training_label=Core.src_label,
                                                     classifier=Core.classifier,
                                                     testing_feature=tarU_feature, testing_label=Core.tarU_label)
    toWrite += ("Accuracy on unlabel target: " + str(acc) + "\n")
    toWrite += ("Accuracy on the target (No TL): %f\n" % (
                    1.0 - FitnessFunction.classification_error(training_feature=Core.src_feature, training_label=Core.src_label,
                                                               classifier=Core.classifier,
                                                               testing_feature=Core.tarU_feature, testing_label=Core.tarU_label)))
    toWrite += ("Computation time: %f\n" % time_elapsed)
    toWrite += ("Number of features: %d\n" % len(indices))
    toWrite += str(best)

    file.write(toWrite)
    file.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])