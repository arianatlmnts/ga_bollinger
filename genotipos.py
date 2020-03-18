import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
class Candidate(object):
    def __init__(self, genotype, fitness):
        self.genotype = genotype
        self.fitness = fitness

    def mutation(self):
        r = random.random()
        if r<p_mut:
            x,y = random.sample(range(1, queens), 2)
            self.genotype[x],self.genotype[y] = self.genotype[y],self.genotype[x]
        self.fitness = fitness(self.genotype)


def returns_fitness(returns):
    neg_returns = len(list(filter(lambda x: (x < 0), returns)))
    pos_returns = len(list(filter(lambda x: (x > 0), returns)))

    return pos_returns / (neg_returns+pos_returns)

def fitness(c):

    return(0)


def main():
    population_size = 100
    C = []
    p_mut = 0.1
    g = np.zeros(4)
    # initialize random population
    for i in range(population_size):
        x = random.uniform(1,3) #Valor aleatorio para el alpha de la banda superior
        y = random.uniform(1,3) #Valor aleatorio para la banda inferior
        g[0], g[1] = x, y
        g[2] = random.randint(0,2) #Selecciona el tipo de media a usar
        g[3] = random.randint(20,200) #
        C.append(Candidate(g.tolist(),fitness(g)))

    # best fitness & survival
    ##C.sort(key=lambda x: x.fitness, reverse=True) # ordenar por fitness
    ##C = C[:population_size]



if __name__ == "__main__":
    main()


