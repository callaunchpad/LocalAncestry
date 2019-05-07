import numpy as np
from peekable import Peekable

DATASET_ROOT = "dataset/"

def get_genotypes(filename, data_inds):
    genotype_file_loc = DATASET_ROOT + filename + ".phgeno"
    genotype_file = Peekable(filename=genotype_file_loc)

    first_line = genotype_file.peek().strip()

    individuals = [[] for _ in first_line]

    ind = 0
    for line in genotype_file:
        if ind > max(data_inds):
            break
        if ind in data_inds:
            line = line.strip()

            for i in range(len(line)):
                individuals[i].append(int(line[i]))
        ind += 1
        
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

def get_ancestry(filename, data_inds):
    ancestry_file_loc = DATASET_ROOT + filename + ".ancestry"
    ancestry_file = Peekable(filename=ancestry_file_loc)

    first_line = ancestry_file.peek().replace('-','').strip()

    individuals = [[] for _ in first_line]

    # print(individuals)
    ind = 0
    for line in ancestry_file:
        if ind > max(data_inds):
            break
        if ind in data_inds:
            line = line.strip().replace('-', '').replace('A', '0').replace('B', '1')

            for i in range(len(line)):
                individuals[i].append(int(line[i]))
        ind += 1

    return np.array(individuals)