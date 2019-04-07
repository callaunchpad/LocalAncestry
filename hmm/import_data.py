import numpy as np
from peekable import Peekable

DATASET_ROOT = "../dataset/"
FILE_NAME = "simulation"

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
    mapper = np.vectorize(lambda x: x * 2 - 1)
    return mapper(arr)  # Super duper efficient way of converting 0 -> -1

def get_ancestry():
    ancestry_file_loc = DATASET_ROOT + FILE_NAME + ".ancestry"
    ancestry_file = Peekable(filename=ancestry_file_loc)

    first_line = ancestry_file.peek().replace('-','').strip()

    indiviudals = [[] for _ in first_line]

    # print(indiviudals)
    for line in ancestry_file:
        line = line.strip().replace('-', '').replace('A', '0').replace('B', '1')

        for i in range(len(line)):
            indiviudals[i].append(int(line[i]))

    return np.array(indiviudals)