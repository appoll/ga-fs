import numpy as np

from run import Run


class GeneticAlgorithm:
    def __init__(self, run):
        self.crossover_rate = 0.7
        self.mutation_rate = 0.02
        self.population_size = 5

        self.population = self.populate(run.data)

    def select(self):
        """
        Selection operator
        """
        # updates current population from previous population based on some criteria

    def cost(self):
        """
        Evaluates the population
        Cost function / Objective function to evaluate the population.
        Performs classification using each of the solutions (feature sets) and assigns it a score

        Classification accuracy  /  Number of selected features
        - https://www.researchgate.net/profile/Cheng_Lung_Huang/publication/222536766_A_GA-based_feature_selection_and_parameters_optimization_for_support_vector_machines/links/544e69e60cf29473161bde17.pdf

        """

    def crossover(self, entity_a, entity_b):
        """
        Exchanges solution information according to self.crossover_rate
        """

    def mutation(self, entity):
        """
        Mutates
        :param entity:
        """

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
            print 'Population Size: %d\n' % len(solutions)
        return solutions

    def iterate(self):
        for solution in self.population:
            print 'Solution with %d features ' % sum(solution.features)
            solution_score = run.knn_sel(selected=solution.features)
            print 'Has score %f ' % solution_score
            solution.fitness = solution_score


class Solution:
    def __init__(self, features_count):
        """
        Wraps a binary vector of features' usage.
        Represents feature subset.
        :param features_count:
        """
        self.size = features_count
        # initialize one solution's binary vector as an array of zeros
        self.features = np.zeros(self.size, dtype=bool)
        self.fitness = -1

    def random(self):
        # random binary numpy array
        self.features = np.random.randint(2, size=self.size, dtype=bool)
        # print 'Created solution of size %d' % len(self.features)
        # print 'Out of which true: %d' %sum(self.features)
        print self.features
        return self


if __name__ == '__main__':
    run = Run(dataset='ionosphere.data')
    ga = GeneticAlgorithm(run=run)

    ga.iterate()
