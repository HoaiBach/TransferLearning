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
import FitnessFunction as fitness

NBIT = Core.no_features
NGEN = 100
NPART = NBIT if NBIT <100 else 100
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

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, stk=list, best=None)


def generate(size):
    part = creator.Particle(1 if random.uniform(0, 1) > threshold else 0 for _ in range(size))
    part.stk = [1 for _ in range(size)]
    return part


def updateParticle(part, best):

    # find flipping probability
    stick_part = map(lambda x: i_stick*(1-x), part.stk)
    diff_pbest = map(operator.abs, map(operator.sub, part.best, part))
    pbest_part = map(lambda x: i_pbest* x, diff_pbest)
    diff_gbest = map(operator.abs, map(operator.sub, best, part))
    gbest_part = map(lambda x: i_gbest* x, diff_gbest)

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
    indices = [index for index, entry in enumerate(particle) if entry == 1.0]
    src_fea_new = Core.src_feature[:, indices]
    tarU_fea_new = Core.tarU_feature[:, indices]
    tarL_fea_new = Core.tarL_feature[:, indices]

    return fitness.fitnessFunction(src_fea_new, Core.src_label, tarU_fea_new)
    return fitness.fitnessFunction(src_fea_new, Core.src_label, tarU_fea_new, Core.classifier,
                                   tarL_fea_new, Core.tarL_label)


toolbox = base.Toolbox()
toolbox.register("particle", generate, size=NBIT)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle)
toolbox.register("evaluate", evaluate)


def main(args):
    global i_stick, i_pbest, i_gbest, ustks
    random.seed(1617 ** 2 * int(args[0]))

    pop = toolbox.population(n=NPART)

    best = None

    for g in range(NGEN):
        print("=====Gen %d=====\n" % g)
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)

            # update pbest
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values

            # update gbest
            if not best or best.fitness < part.fitness:
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

        # now update the position
        for part in pop:
            toolbox.update(part, best)

        # update the parameters
        i_stick = is_up - (is_up - is_low)*(g+1)/NGEN
        i_gbest = (1-i_stick)/(pg_rate+1)
        i_pbest = pg_rate*i_gbest
        ustks   = ustks_low + (ustks_up-ustks_low)*(g+1)/NGEN

        # Gather all the fitnesses in one list and print the stats
        print("Fitness function of best: %f\n" % best.fitness.values)

    return pop, best


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])