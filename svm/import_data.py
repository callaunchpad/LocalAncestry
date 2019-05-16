import numpy as np
from peekable import Peekable

DATASET_ROOT = "../dataset/"
FILE_NAME = "simulation"

global train_i
global test_i

def get_genotypes():
    genotype_file_loc = DATASET_ROOT + FILE_NAME + ".phgeno"
    genotype_file = Peekable(filename=genotype_file_loc)

    first_line = genotype_file.peek().strip()

    indiviudals = [[] for _ in first_line]

    for line in genotype_file:
        line = line.strip()

        for i in range(len(line)):
            indiviudals[i].append(int(line[i]))

    arr = np.array(indiviudals)
    # mapper = np.vectorize(lambda x: x * 2 - 1)
    # return mapper(arr)  # Super duper efficient way of converting 0 -> -1
    # print("ARR TESTING!" , arr[:10,])
    training_i = np.random.randint(arr.shape[0], size=int(len(arr)*2/3))
    # print(arr.shape, int(len(arr)*2/3))
    testing_i = np.random.randint(arr.shape[0], size=int(len(arr)/3))
    # print("TRAIN I:" , training_i)
    global train_i, test_i
    train_i = training_i
    test_i = testing_i
    training, test = arr[training_i,:], arr[test_i,:]
    # print("TRAIN:", training[:10,], "TEST:", test[:10,])
    return training, test

def get_ancestry():
    ancestry_file_loc = DATASET_ROOT + FILE_NAME + ".ancestry"
    ancestry_file = Peekable(filename=ancestry_file_loc)

    first_line = ancestry_file.peek().replace('-','').strip()

    individuals = [[] for _ in first_line]

    # print(indiviudals)
    for line in ancestry_file:
        line = line.strip().replace('-', '').replace('A', '0').replace('B', '1')

        for i in range(len(line)):
            individuals[i].append(int(line[i]))

    # return np.array(indiviudals)
    arr = np.array(individuals)
    # print("ANCESTRY ARR TESTING!" , arr[:10,], "ARR SHAPE", arr.shape)
    # print("TRAIN I:" , train_i)
    training, test = arr[train_i,:], arr[test_i,:]
    # print("TRAINING", training.shape)
    return training, test
