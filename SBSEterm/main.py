import math

import common
from GeneticAlgorithm import GA
from CNN_Classification import SentimentAnalysisByCNN

import random

def generate_init_population(popul_size):
    population = []
    while True:
        idx = []
        for j in range(7):
            idx.append(random.randrange(3))
        chromosome = [common.character_count[idx[0]], common.embedding_dimension[idx[1]], common.num_filters[idx[2]],
                      common.filter_size[idx[3]], common.batch_size[idx[4]], common.learning_rate[idx[5]],
                      common.weight_decay[idx[6]]]
        if chromosome not in population:
            population.append(chromosome)

        if len(population) == popul_size:
            break

    return population

if __name__ == '__main__':
    population = generate_init_population(common.population_size)
    ga = GA()

    evals = common.fitness_eval
    min_score = math.inf
    best_chrom = []

    while evals > 0:
        population = ga.get_next_generation(population)
        scores = {}
        for chromosome in population:
            sentiCNN = SentimentAnalysisByCNN(character_count=chromosome[0],
                                              embedding_dimension=chromosome[1],
                                              num_filters=chromosome[2],
                                              num_filter_size=chromosome[3],
                                              batch_size=chromosome[4],
                                              learning_rate=chromosome[5],
                                              weight_decay=chromosome[6]
                                              )
            sentiCNN.preprocess_sentiment_data()
            score = sentiCNN.train_eval_model()
            scores[score] = chromosome

        sorted_population = sorted(scores.items())

        if min_score > sorted_population[0][0]:
            min_score = sorted_population[0][0]
            best_chrom = sorted_population[0][1]

        evals -= 1

    print(min_score)
    print(best_chrom)