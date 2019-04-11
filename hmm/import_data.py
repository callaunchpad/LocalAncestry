import numpy as np
from peekable import Peekable

DATASET_ROOT = "dataset/"

def get_genotypes(filename):
    genotype_file_loc = DATASET_ROOT + filename + ".phgeno"
    genotype_file = Peekable(filename=genotype_file_loc)

    first_line = genotype_file.peek().strip()

    individuals = [[] for _ in first_line]
    for line in genotype_file:
        line = line.strip()

        for i in range(len(line)):
            individuals[i].append(int(line[i]))
        
    arr = np.array(individuals) # have matrix of individuals [sites x individuals]
    print('done loading', filename)
    return arr  

def get_gen_distances(filename):
    snptype_file_loc = DATASET_ROOT + filename + ".phsnp"
    snptype_file = Peekable(filename=snptype_file_loc)

    gen_distances = []
    for line in snptype_file:
        line = ' '.join(line.split())
        line = line.split()
        gen_distances.append(float(line[2]))

    print('done parsing genetic distances')
    return gen_distances

def get_ancestry():
    ancestry_file_loc = DATASET_ROOT + FILE_NAME + ".ancestry"
    ancestry_file = Peekable(filename=ancestry_file_loc)

    first_line = ancestry_file.peek().replace('-','').strip()

    individuals = [[] for _ in first_line]

    # print(individuals)
    for line in ancestry_file:
        line = line.strip().replace('-', '').replace('A', '0').replace('B', '1')

        for i in range(len(line)):
            individuals[i].append(int(line[i]))

    return np.array(individuals)