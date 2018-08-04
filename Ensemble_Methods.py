import copy
import math
import random
from enum import Enum

from anytree import Node, RenderTree

GA_STEP_LIMITATION = 1000
NUMBER_OF_POPULATION = 1000
PRECISION_SOLVING_EQUATION = 0.1
PARENT_SELECTION_RATE = 10

SCORES_VECTOR = dict()
VALIDATION_SET = list()
OBJECTIVE_THRESHOLD = 0.90


class Func(Enum):
    Avg = 'Avg'
    Max = 'Max'
    Product = 'Product'
    Square = 'Square'


class Algo(Enum):
    Algo1 = "k_means"
    Algo2 = "SVM"
    Algo3 = "DB_Scan"
    Algo4 = "fastVOA"
    Algo5 = "LOF"
    Algo6 = "loOP"


class Individual:
    fitness = 0
    depth = 0
    parent = None
    """
    This is root of tree
    """
    state = None

    def __init__(self, state):
        self.state = state


def is_goal(population):
    for individual in population:
        if individual.fitness > OBJECTIVE_THRESHOLD:
            return True
    return False


def get_random_func():
    x = random.randint(0, 3)
    if x == 0:
        return Func.Avg
    elif x == 1:
        return Func.Max
    elif x == 2:
        return Func.Product
    else:
        return Func.Square


def get_random_algo():
    x = random.randint(0, 5)
    if x == 0:
        return Algo.Algo1
    elif x == 1:
        return Algo.Algo2
    elif x == 2:
        return Algo.Algo3
    elif x == 3:
        return Algo.Algo4
    elif x == 4:
        return Algo.Algo5
    else:
        return Algo.Algo6


def make_random_tree():
    target = list()
    target.append(Node(Algo.Algo1))
    target.append(Node(Algo.Algo2))
    target.append(Node(Algo.Algo3))
    target.append(Node(Algo.Algo4))
    target.append(Node(Algo.Algo5))
    target.append(Node(Algo.Algo6))

    while len(target) > 1:
        func = get_random_func()
        root = Node(func)
        if func == Func.Square:
            x = 1
        else:
            x = random.randint(3, 5)

        if x > len(target):
            x = len(target)

        selected_node = random.sample(range(0, len(target)), x)
        temp = set()
        for index in selected_node:
            target[index].parent = root
            temp.add(target[index])

        for a in temp:
            target.remove(a)

        target.append(root)

    return target[0]


def get_random_initial_state():
    root = make_random_tree()
    return Individual(root)


def generate_initial_population():
    result = list()
    i = 0
    while i < NUMBER_OF_POPULATION:
        result.append(get_random_initial_state())
        i += 1
    return result


def select_parents_with_tournament_selection(population):
    initialNumber = PARENT_SELECTION_RATE * 2
    if initialNumber > NUMBER_OF_POPULATION :
        initialNumber = NUMBER_OF_POPULATION
    tempResult = list()
    temp = copy.deepcopy(population)
    for i in range(initialNumber):
        individual = temp[random.randint(0, len(temp))]
        tempResult.append(individual)
        temp.remove(individual)

    tempResult = sorted(tempResult, key=lambda x: x.fitness, reverse=True)
    # select best PARENT_SELECTION_RATE number of temp result for final result
    result = list()
    for i in range(PARENT_SELECTION_RATE):
        result.append(tempResult[i])
    return result


def crossover(parent1, parent2):
    offspring = Individual((parent1.getState() + parent2.getState()) / 2)
    return offspring


def objective_function(individual):
    # TODO : compute AUC
    # 1. compute scores based on individual with iterate on root of tree
    # 2. compute ROC or AUC based on validation set
    # 3. compare AUC score with threshold
    root = individual.state
    # iterate on root
    return 1.0


def crossover_and_offspring(parents):
    offspring = list()
    i = 0
    while i < len(parents):
        # problem object generates child from parents
        child = crossover(parents[i], parents[i + 1])
        child.fitness = objective_function(child)
        offspring.append(child)
        i += 2
    return offspring


def mutation(children):
    for i in range(len(children)):
        x = children[i].getState()
        gaussianValue = math.gaussianValue(x)
        if 0.2 < gaussianValue < 3.14:
            children[i].setState(gaussianValue)
    return children


def remaining_selection(children, population):
    # Adds K/2 individual to population
    for i in range(len(children)):
        population.add(children[i])

    # Deletes K/2 individual from population based on fitness
    temp = list()
    for i in range(PARENT_SELECTION_RATE):
        x = math.getIntegerRandNum(population.size())
        temp.append(population[x])

    temp.sort(math.getComparator(problem))
    i = PARENT_SELECTION_RATE-1
    while i >= PARENT_SELECTION_RATE/2:
        population.remove(temp[i])
        i -= 1
    return population


def best_individual(population):
    max = population[0]
    for i in range(len(population)):
        if objective_function(population[i]) > objective_function(max):
            max = population[i]
    return max


def ensemble():
    population = generate_initial_population()
    stepLimit = GA_STEP_LIMITATION
    while stepLimit > 0 and not is_goal(population):
        for individual in population:
            individual.fitness = objective_function(individual)
        parents = select_parents_with_tournament_selection(population)
        children = crossover_and_offspring(parents)
        children = mutation(children)
        population = remaining_selection(children, population)
        stepLimit -= 1

    print(best_individual(population))


