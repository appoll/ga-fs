from matplotlib import pyplot

import numpy as np
import seaborn
from run import Run
from settings import PLOTS_PATH
from solution import Solution
from utils import plot_confusion_matrix

OFFSPRING_COUNT = 100


class GeneticAlgorithm:
    def __init__(self, run):
        self.crossover_rate = 0.7
        self.mutation_rate = 0.09
        # 501 - seed for random train test splits.
        # self.population_size = 501

        # 502 - seed for random cross validation splits. each solution now gets average of 10 performances as fitness
        # ran together with offspring count = 20
        # self.population_size = 502

        # 503 - same seed for all generations, offspring = 200
        #        self.population_size = 503

        # 1000 - same seed for all generations, offspring = 600
        # self.population_size = 1000

        # offspring = 200
        # self.population_size = 505

        # offspring 100
        # self.population_size = 506
        #
        # # offspring 100
        # self.population_size = 507

        # offspring 100, used without duplicate check
        # self.population_size = 508

        # # offspring 100, with duplicate check
        # self.population_size = 509

        # offspring 100 without duplicate check
        # self.population_size = 510

        # # offspring 100, with duplicate check
        self.population_size = 501

        # self.population = self.populate(run.data)
        self.run = run
        # weight of accuracy score and of feature count; used in fitness evaluation
        self.weight_a = 0.7
        self.weight_fc = 0.3

        self.generations = 0
        self.offspring = []
        self.population = []

        self.fitness_averages = []
        self.best_fitness_values = []
        self.similarity_measures = []

        self.full_set_10fold_run_duration = 0
        self.full_set_10fold_run_accuracy = 0
        self.full_set_size = 0

        self.best_solution = Solution(features_count=0)

        # def run_with_full_feature_set(self):
        #   self.full_set_10fold_run_accuracy, self.full_set_10fold_run_duration = self.run.knn_cv(10)

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
        # print "Updating population with initial size:"
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
        return self.weight_a * accuracy_score + self.weight_fc * (1 / sum(solution_features))

    def crossover(self, parents):
        """
        Exchanges solution information according to self.crossover_rate
        :param parents: List of parents selected based on fitness
        """
        print 'Start mating...'
        while (len(self.offspring) < len(parents)):
            # randomly select 2 parents
            pair = np.random.choice(a=parents, size=2)

            # 1 point crossover
            crossover_point = np.random.randint(pair[0].total_size)

            child_1 = np.append(pair[0].features[:crossover_point], pair[1].features[crossover_point:])
            child_2 = np.append(pair[1].features[:crossover_point], pair[0].features[crossover_point:])

            if self.compatible(child=child_1):
                self.offspring.append(Solution(len(child_1), child_1))

            if self.compatible(child=child_2):
                self.offspring.append(Solution(len(child_2), child_2))

    def compatible(self, child):
        indices = np.nonzero(child)
        used_features_indices = np.array_str(indices[0])

        for solution in self.population:
            if solution.used_features_indices == used_features_indices:
                return False
        return True

    def mutation(self, individual):
        """
        Mutates an individual according to the crossover rate
        :param individual: solution instance
        """
        risks = np.random.random_sample(size=individual.total_size)
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
        print sum(sum(offspring.features) for offspring in self.offspring)
        for solution in self.offspring:
            # print solution.to_string()
            solution = self.mutation(individual=solution)
            # print solution.to_string()
        print "Done mutating..."
        print sum(sum(offspring.features) for offspring in self.offspring)

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
        self.full_set_size = len(solutions[0].features)
        return solutions

    def iterate(self):
        print 'Generation %d ' % self.generations
        self.generations += 1
        sum = 0
        count = 0
        skipped = 0
        used_feature_indices = []
        # evaluate
        for solution in self.population:
            # print 'Solution with %d features ' % sum(solution.features)
            # print solution.fitness
            if solution.fitness != -1:
                # print 'Skipping, solution fitness %f ' % solution.fitness
                skipped += 1

                used_feature_indices.append(solution.used_features_indices)
                sum += solution.fitness
                continue
            features = solution.features
            count += 1
            # 502
            # solution_score, solution_duration = run.knn_sel_cv(selected=features, seed=self.generations)

            # 503 seed  = 3, offspring 200
            # solution_score, solution_duration = run.knn_sel(selected=features, seed=3)

            # 506 cross validation seed = 7, offspring = 100, average over cross-validation folds
            # solution_score, solution_duration = run.knn_sel_cv(selected=features, seed=7)

            # # 507 & 500 cross validation seed = 8, offspring = 100, average over cross-validation folds
            # solution_score, solution_duration = run.knn_sel_cv(selected=features, seed=8)
            solution_score, solution_duration = run.knn_sel(selected=features, seed=4)

            # 508 seed  = 3, offspring 100
            # solution_score, solution_duration = run.knn_sel(selected=features, seed=3)

            # 509 seed  = 3, offspring 100
            # solution_score, solution_duration = run.knn_sel(selected=features, seed=3)

            # 510 seed  = 3, offspring 100
            # solution_score, solution_duration = run.knn_sel(selected=features, seed=3)

            solution.fitness = self.cost(solution_score, features)
            solution.duration = solution_duration

            if solution.fitness > self.best_solution.fitness:
                self.best_solution = solution

            used_feature_indices.append(solution.used_features_indices)
            sum += solution.fitness

        print 'Evaluated only %d solutions, skipped %d ' % (count, skipped)
        unique_solutions_set = set(used_feature_indices)
        unique_solutions = len(unique_solutions_set)
        similarity_measure = float(unique_solutions) / len(self.population)
        print unique_solutions
        print 'Basic similarity measure %f ' % similarity_measure

        self.fitness_averages.append(sum / len(self.population))
        self.similarity_measures.append(similarity_measure)

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

    def get_first_three(self):
        ranked_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        return ranked_population[0], ranked_population[1], ranked_population[2]

    def termination_criteria(self):
        # comparing to either of max score or mean score of cross validation is pointless
        full_run_cost = self.weight_a * self.full_set_10fold_run_accuracy + self.weight_fc * (1 / self.full_set_size)
        print 'Comparing to: '
        print full_run_cost
        # return self.best_solution.fitness < full_run_cost
        # return self.generations < 2
        return self.generations < 200


if __name__ == '__main__':
    # run = Run(dataset='ionosphere.data')
    run = Run(dataset='ionosphere.data')
    ga = GeneticAlgorithm(run=run)
    # ga.run_with_full_feature_set()
    count = 0

    ga.population = ga.populate(ga.run.data)
    # new generations until termination criteria is met
    ga.iterate()
    while ga.termination_criteria():
        ga.iterate()
        print 'Best three solutions:'
        ga.print_first_three()

    ga.print_first_three()
    ranked_population = sorted(ga.population, key=lambda x: x.fitness, reverse=True)

    #
    # used_feature_indices = []
    # for i in range(ga.population_size):
    #     print ranked_population[i].to_string()
    #     used_feature_indices.append(ranked_population[i].used_features_indices)
    #
    # unique_solutions = len(set(used_feature_indices))
    # similarity_measure = float(unique_solutions) / len(ga.population)
    # print unique_solutions
    # print 'Final Basic similarity measure %f ' % similarity_measure

    first, second, third = ga.get_first_three()
    # # print 'First solution CV: '
    # first_sol_scores = run.knn_sel_cv(first.features, seed=9)
    # # run.knn_sel_cv(first.features, seed=10)
    # # run.knn_sel_cv(first.features, seed=11)
    #
    # all_features_scores = run.knn_cv(splits=10, seed=9)
    # # run.knn_cv(splits=10, seed=10)
    # # run.knn_cv(splits=10, seed=11)


    first_sol_clf_report, first_sol_cnf_m = run.knn_validate_sub(first.features, seed=12)
    all_features_clf_report, all_features_cnf_m = run.knn_validate_full(seed=12)

    print "Classification report first sol"
    print first_sol_clf_report

    print "Clf report full feature set"
    print all_features_clf_report

    # plot_confusion_matrix(input_type="best_%s"%run.dataset, cm=first_sol_cnf_m, classes=[0, 1], normalize=True,
    #                       title="Best features subset")
    #
    # plot_confusion_matrix(input_type="all_%s"%run.dataset, cm=all_features_cnf_m, classes=[0, 1], normalize=True,
    #                       title="All features")

    # run.knn_validate(first.features, 10, seed=8)

    # run knn on the full feature set
    #    all_score, all_duration = run.knn()

    # all_cv, all_cv_duration = run.knn_cv(10)
    # durations = [all_duration, ga.full_set_10fold_run_duration, first.duration, first.duration, first.duration]
    # soft = ('Full', 'Full CV', 'Solution 1', 'Solution 2', 'Solution 3')

    # average of all solutions' fitness in population at the end of the run
    pyplot.figure(1).suptitle(t="Population fitness across generations.")
    pyplot.plot(ga.fitness_averages)
    pyplot.savefig(PLOTS_PATH + "fa-knn-%s-%s-%sc-%sm-%swa%swf%sgen.png" % (
        run.dataset, ga.population_size, ga.crossover_rate, ga.mutation_rate, ga.weight_a, ga.weight_fc,
        ga.generations))

    pyplot.figure(2).suptitle(t="Fittest individuals' fitness across generations.")
    pyplot.plot(ga.best_fitness_values, c='r')
    pyplot.savefig(PLOTS_PATH + "bf-knn-%s-%s-%sc-%sm-%swa%swf%sgen.png" % (
        run.dataset, ga.population_size, ga.crossover_rate, ga.mutation_rate, ga.weight_a, ga.weight_fc,
        ga.generations))

    fig, ax1 = pyplot.subplots()
    ax2 = ax1.twinx()
    ax1.plot(ga.fitness_averages, c='g')
    ax2.plot(ga.similarity_measures, c='r')
    ax1.set_title('Population similarity measure and fitness across generations.')
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Population fitness', color='g')
    ax2.set_ylabel('Diversity measure', color='r')
    ax2.set_ylim([0.2, 1])
    pyplot.savefig(PLOTS_PATH + "sm-knn-%s-%s-%sc-%sm-%swa%swf%sgen.png" % (
        run.dataset, ga.population_size, ga.crossover_rate, ga.mutation_rate, ga.weight_a, ga.weight_fc,
        ga.generations))

    fig, ax1 = pyplot.subplots()
    ax2 = ax1.twinx()
    ax1.plot(first_sol_scores, c='g')
    ax2.plot(all_features_scores, c='r')
    ax1.set_title('Cross validation scores of best feature subset and full feature set.')
    ax1.set_xlabel('CV folds')
    ax1.set_ylabel('Best feature subset', color='g')
    ax2.set_ylabel('Full feature set', color='r')
    ax2.set_ylim([0.5, 1])
    ax1.set_ylim([0.5, 1])
    pyplot.savefig(PLOTS_PATH + "full-first-knn-%s-%s-%sc-%sm-%swa%swf%sgen.png" % (
        run.dataset, ga.population_size, ga.crossover_rate, ga.mutation_rate, ga.weight_a, ga.weight_fc,
        ga.generations))


    # pyplot.figure(3).suptitle(t='Time needed to perform train + predict (smaller is better)')
    # pyplot.bar(np.arange(0, 3 * len(durations), 3), durations, color='b', label='Ionosphere dataset')
    # pyplot.legend()
    # pyplot.xticks(np.arange(0, 3 * len(durations), 3) + 1,
    #               soft)
    # pyplot.savefig(PLOTS_PATH + "bf-knn-%s-%s-%sc-%sm-%swa%swf%sgen-durations.png" % (
    #     run.dataset, ga.population_size, ga.crossover_rate, ga.mutation_rate, ga.weight_a, ga.weight_fc,
    #     ga.generations))
