import random
import common

from CNN_Classification import SentimentAnalysisByCNN


class GA:
    def __init__(self):
        self.next_pop_size = int(common.population_size / 2)

    def get_parents_pool(self, population):
        parents_pool = []

        # for _ in range(common.population_size / 2):
        # random_sampled_population = random.sample(population, int(common.population_size / 2))

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
        if len(sorted_population) > 1:
            for i in range(self.next_pop_size):
                parents_pool.append(sorted_population[i][1])
        else:
            parents_pool.extend(population[self.next_pop_size:])

        return parents_pool

    def get_cross_over_child(self, parent1, parent2):
        front_idx = random.randint(0, int(len(parent1) / 2))
        end_idx = front_idx + int(len(parent1) / 2)

        child = parent1[:]

        for i in range(0, len(parent1)):
            if front_idx > i or end_idx < i:
                child[i] = None

        for i in range(0, len(parent2)):
            if child[i] == None:
                child[i] = parent2[i]

        return child

    def get_random_mutate(self, child):
        mutate_rate = 1 / len(child)
        if random.random() < mutate_rate:
            idx = random.randrange(3)
            if child[0] != common.character_count[idx]:
                child[0] = common.character_count[idx]
        if random.random() < mutate_rate:
            idx = random.randrange(3)
            if child[1] != common.embedding_dimension[idx]:
                child[1] = common.embedding_dimension[idx]
        if random.random() < mutate_rate:
            idx = random.randrange(3)
            if child[2] != common.num_filters[idx]:
                child[2] = common.num_filters[idx]
        if random.random() < mutate_rate:
            idx = random.randrange(3)
            if child[3] != common.filter_size[idx]:
                child[3] = common.filter_size[idx]
        if random.random() < mutate_rate:
            idx = random.randrange(3)
            if child[4] != common.batch_size[idx]:
                child[4] = common.batch_size[idx]
        if random.random() < mutate_rate:
            idx = random.randrange(3)
            if child[5] != common.learning_rate[idx]:
                child[5] = common.learning_rate[idx]
        if random.random() < mutate_rate:
            idx = random.randrange(3)
            if child[6] != common.weight_decay[idx]:
                child[6] = common.weight_decay[idx]

        return child

    def get_next_generation(self, population):
        next_generation = []
        next_generation.extend(population[:self.next_pop_size])

        parents_pool = self.get_parents_pool(population)
        for _ in range(self.next_pop_size):
            parent1, parent2 = random.sample(parents_pool, 2)
            child = self.get_cross_over_child(parent1, parent2)
            mutate = self.get_random_mutate(child)
            next_generation.append(mutate)

        return next_generation
