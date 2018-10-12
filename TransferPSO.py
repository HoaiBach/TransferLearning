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
from Core import src_label

w = 0.7298
c1 = c2 = 1.49618
threshold = 0.6

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, 
    smin=None, smax=None, best=None)

def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size)) 
    part.speed = [0 for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part

def updateParticle(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1)*c1 for _ in range(len(part)))
    u2 = (random.uniform(0, phi2)*c2 for _ in range(len(part)))
    
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    v_mo = map(lambda x: x*w,part.speed)
    
    part.speed = list(map(operator.add, v_mo, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if speed < part.smin:
            part.speed[i] = part.smin
        elif speed > part.smax:
            part.speed[i] = part.smax
    part[:] = list(map(operator.add, part, part.speed))

def evaluate(particle):
    indices = [index for index,entry in enumerate(particle) if entry>threshold]
    src_fea_new = Core.src_feature[:, indices]
    tarU_fea_new = Core.tarU_feature[:, indices]
    tarL_fea_new = Core.tarL_feature[:, indices]

    return fitness.fitnessFunction(src_fea_new, Core.src_label, tarU_fea_new, Core.classifier,
                                   tarL_fea_new, Core.tarL_label)


toolbox = base.Toolbox()
toolbox.register("particle", generate, size=Core.no_features, pmin=0.0, pmax=1.0, smin=-2, smax=2)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=1.0, phi2=1.0)
toolbox.register("evaluate", evaluate)

def main(args):
    random.seed(1617**2*args[0])
    
    pop = toolbox.population(n=10)

    GEN = 10
    best = None


    for g in range(GEN):
        print("=====Gen %d=====\n" %g)
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)

        # Gather all the fitnesses in one list and print the stats
        print("Fitness function of best: %f\n" %best.fitness.values)
        print("Accuracy on the target (TL): %f\n" %)
    
    return pop, best

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])