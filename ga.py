from matplotlib import pyplot

import numpy as np
import seaborn
from run import Run
from solution import Solution

OFFSPRING_COUNT = 200

class GeneticAlgorithm:
    def __init__(self, run):
        self.crossover_rate = 0.7
        self.mutation_rate = 0.09
        self.population_size = 500

        # self.population = self.populate(run.data)
        self.run = run
        # weight of accuracy score and of feature count; used in fitness evaluation
        self.weight_a = 0.8
        self.weight_fc = 0.2

        self.generations = 0
        self.offspring = []
        self.population = []

        self.fitness_averages = []
        self.best_fitness_values = []

    def select(self):
        """
        Selection operator, prepares returns part of the population selected for mating
        """
        # updates current population from previous population based on some criteria
        ranked_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        self.best_fitness_values.append(ranked_population[0].fitness)
        # return 20 fittest individuals as selected for the reproduction
        return ranked_population[:OFFSPRING_COUNT]

    def updatePopulation(self):
        """
        Add children to population and substitute with lowest fit older individuals
        """
        print "Updating population with initial size:"
        ranked_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        ranked_population = ranked_population[:len(self.population) - OFFSPRING_COUNT]
        ranked_population.extend(self.offspring)
        self.population = ranked_population
        self.offspring = []

    def cost(self, accuracy_score, solution_features):
        """
        Evaluates the population
        Cost function / Objective function to evaluate the population.
        Performs classification using each of the solutions (feature sets) and assigns it a score

        Classification accuracy  /  Number of selected features
        - https://www.researchgate.net/profile/Cheng_Lung_Huang/publication/222536766_A_GA-based_feature_selection_and_parameters_optimization_for_support_vector_machines/links/544e69e60cf29473161bde17.pdf

        """
        # print 'Accuracy score %f ' % accuracy_score
        return self.weight_a * accuracy_score + self.weight_fc * (1/sum(solution_features))

    def crossover(self, parents):
        """
        Exchanges solution information according to self.crossover_rate
        :param parents: List of parents selected based on fitness
        """

        while (len(self.offspring) < len(parents)):
            # randomly select 2 parents
            pair = np.random.choice(a=parents, size=2)

            # 1 point crossover
            crossover_point = np.random.randint(pair[0].size)

            child_1 = np.append(pair[0].features[:crossover_point], pair[1].features[crossover_point:])
            child_2 = np.append(pair[1].features[:crossover_point], pair[0].features[crossover_point:])

            self.offspring.append(Solution(len(child_1), child_1))
            self.offspring.append(Solution(len(child_2), child_2))

    def mutation(self, individual):
        """
        Mutates an individual according to the crossover rate
        :param individual: solution instance
        """
        risks = np.random.random_sample(size=individual.size)
        genes = individual.features
        for gene_idx, gene in enumerate(genes):
            mutation_risk = risks[gene_idx]
            if mutation_risk < self.mutation_rate:
                gene = not gene
                individual.features[gene_idx] = gene
        return individual

    def mutate(self):
        """
        Mutate by flipping a random position
        """
        print "Start mutating..."
        for solution in self.offspring:
            # print solution.to_string()
            solution = self.mutation(individual=solution)
            # print solution.to_string()

    def populate(self, data):
        # solutions array, of size self.population_size
        """
        Creates population of random solutions
        :param data: Data represents entire dataset, including labels
        """
        solutions = []

        # see @param data
        solution_size = data.columns.size - 1
        for i in range(self.population_size):
            solution = Solution(solution_size)
            solutions.append(solution.random())
            # print 'Population Size: %d\n' % len(solutions)
        return solutions

    def iterate(self):
        print 'Generation %d ' % self.generations
        self.generations += 1
        sum =0
        # evaluate
        for solution in self.population:
            # print 'Solution with %d features ' % sum(solution.features)
            features = solution.features
            solution_score = run.knn_sel(selected=features)
            solution.fitness = self.cost(solution_score, features)
            sum += solution.fitness
        self.fitness_averages.append(sum/len(self.population))
        parents = self.select()
        self.crossover(parents)
        self.mutate()
        self.updatePopulation()

        # if (len(self.offspring) > 0):
        #     print 'Offspring not empty, evaluating'
        #     for solution in self.offspring:
        #         solution_score = run.knn_sel(selected=solution.features)
        #         solution.fitness = self.cost(solution_score)

    def print_first_three(self):
        ranked_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        print ranked_population[0].to_string()
        print ranked_population[1].to_string()
        print ranked_population[2].to_string()

    def termination_criteria(self):
        return self.generations < 100


if __name__ == '__main__':
    run = Run(dataset='ionosphere.data')
    ga = GeneticAlgorithm(run=run)
    count = 0

    ga.population = ga.populate(ga.run.data)
    # run generations until termination criteria is met
    while ga.termination_criteria():
        ga.iterate()
        print "Iteration %d " % count
        count += 1
        print 'Best three solutions:'
        ga.print_first_three()

    # average of all solutions' fitness in population at the end of the run
    pyplot.figure(1).suptitle(t="Population fitness across generations.")
    pyplot.plot(ga.fitness_averages)
    pyplot.savefig("woop4-1000-0802.png")

    pyplot.figure(2).suptitle(t="Fittest individuals' accuracy across generations.")
    pyplot.plot(ga.best_fitness_values, c='r')
    pyplot.savefig("woop44-1000-0802.png")
