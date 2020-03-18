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
    # initialize random population
    for i in range(population_size):
        # permutación
        g = np.random.permutation(range(1,queens+1))

        C.append(Candidate(g.tolist(),fitness(g)))


    # best fitness & survival
    C.sort(key=lambda x: x.fitness, reverse=True) # ordenar por fitness
    C = C[:population_size]



if __name__ == "__main__":
    main()


