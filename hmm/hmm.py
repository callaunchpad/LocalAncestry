import numpy as np
import time
from import_data import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from peekable import Peekable
from sklearn.metrics import accuracy_score

data_inds = list(range(100))
num_obs = len(data_inds)

num_inds_pop1 = 10
num_inds_pop2 = 10

# Get reference population data
pop1 = get_genotypes("CEU", data_inds)[:num_inds_pop1, :]
pop2 = get_genotypes("YRI", data_inds)[:num_inds_pop2, :]

# Get genetic distance data - CEU & YRI snp data files are the same
gen_dists = get_gen_distances("CEU")

# Define hyperparameters (param1 for European, param2 for African)
n1 = pop1.shape[0]
n2 = pop2.shape[0]

T = 100
mu1 = 0.2
mu2 = 1 - mu1

# Parameters set from HAPMIX paper
p1 = 0.05
p2 = 0.05
ro1 = 60000/n1
ro2 = 90000/n2
theta1 = 0.2/(0.2 + n1)
theta2 = 0.2/(0.2 + n2)
theta3 = 0.01

transition_prob_time = 0

# Helper functions
def pois_T(r_s):
	return np.exp(-r_s * T)


def pois_ro(r_s, ro):
	return np.exp(-r_s * ro)


def indicator(pop, indiv, value, site):
	data = (pop1 if indiv < n1 else pop2)
	# print(pop, indiv, value)
	if indiv < n1:
		return data[indiv][site] == value
	return data[indiv - n1][site] == value

# Helper for generating hidden states
def gen_hidden_states():
	# zero index population individuals
	hidden_states = []
	for i in range(n1):
		hidden_states.append((1, 1, i))
		hidden_states.append((1, 2, i))
	for j in range(n2):
		hidden_states.append((2, 2, n1 + j))
		hidden_states.append((2, 1, n1 + j))
	return hidden_states

# Helper for getting relevant variables
def get_relevant_vars(hidden_state):
	(l, m, n) = hidden_state
	(mu_l, ro_l, p_l) = (mu1, ro1, p1) if l == 1 else (mu2, ro2, p2)
	n_m = n1 if m == 1 else n2
	return (mu_l, ro_l, p_l, n_m)

def emm_prob(curr_state, site, snptype):
	# population j is 1 for european, 2 for african 
	(i, j, k) = curr_state
	thetai = (theta1 if i == 1 else theta2)
	if snptype == 1:
		if i == j: 
			return thetai * indicator(j, k, 0, site) + (1 - thetai) * indicator(j, k, 1, site)
		else: 
			return theta3 * indicator(j, k, 0, site) + (1 - theta3) * indicator(j, k, 1, site)
	else:
		if i == j: 
			return thetai * indicator(j, k, 1, site) + (1 - thetai) * indicator(j, k, 0, site)
		else: 
			return theta3 * indicator(j, k, 1, site) + (1 - theta3) * indicator(j, k, 0, site)

def generate_transition_prob(r_s):
	transition_dict = {}
	given_states = [(1, 1, 1), (1, 2, 1), (2, 1, 1), (2, 2, 1)]
	for given_state in given_states:
		mu_l, ro_l, p_l, n_m = get_relevant_vars(given_state)
		p1 = (1 - pois_T(r_s))*mu_l * (1 - p_l)/n_m
		p2 = (1 - pois_T(r_s))*mu_l * p_l/n_m
		p3 = pois_T(r_s) * (1 - pois_ro(r_s, ro_l)) * (1 - p_l)/n_m + (1 - pois_T(r_s))*mu_l * (1 - p_l)/n_m
		p4 = pois_T(r_s) * pois_ro(r_s, ro_l) + pois_T(r_s) * (1 - pois_ro(r_s, ro_l)) * (1 - p_l)/n_m + (1 - pois_T(r_s))*mu_l * (1 - p_l)/n_m
		p5 = pois_T(r_s) * (1 - pois_ro(r_s, ro_l)) * p_l/n_m + (1 - pois_T(r_s))*mu_l * p_l/n_m
		p6 = pois_T(r_s) * pois_ro(r_s, ro_l) + pois_T(r_s) * (1 - pois_ro(r_s, ro_l)) * p_l/n_m + (1 - pois_T(r_s))*mu_l * p_l/n_m
		transition_dict[given_state[:2]] = [p1, p2, p3, p4, p5, p6]

	def transition_prob(curr_state, given_state):
		(i, j, k) = curr_state
		(l, m, n) = given_state
		transitions = transition_dict[(l, m)]
		if l != i and m == l:
			return transitions[0]
		elif l != i and m != l:
			return transitions[1]
		elif l == i and m == l and (j != m or k != n):
			return transitions[2]
		elif l == i and m == l and j == m and k == n:
			return transitions[3]
		elif l == i and m != l and (j != m or k != n):
			return transitions[4]
		elif l == i and m != l and j == m and k == n:
			return transitions[5]
	return transition_prob


# Adapted from Wikipedia: Forward-Backward Algorithm
def fwd_bkw(observations, snpindices, gen_distances, states):
	"""
	snpindices: array of indices of the sites to be looked at (zero indexed)
	states: array of states
	start_prob: dictionary of key: states, value: probability
	trans_prob(curr_state, next_state): function which gives probability between two states, returns float
	emm_prob(curr_state, obs): function which gives probability of emissions, returns float
	"""
	# forward part of the algorithm
	forward_pass_time = 0
	fwd = []
	f_prev = {}
	for i, snp_ind in enumerate(snpindices):
		transition_prob = generate_transition_prob(gen_distances[i-1])
		f_curr = {}
		for st in states:
			if i == 0:
				# base case for the forward part
				prev_f_sum = 1 / len(states)
			else:
				prev_f_sum = sum(f_prev[k]*transition_prob(k, st) for k in states)
			f_curr[st] = emm_prob(st, snp_ind, observations[i]) * prev_f_sum
		fwd.append(f_curr)
		f_prev = f_curr

		if i % 1000 == 0:
			print("CURR FORWARD IND:", i)

	p_fwd = sum(f_curr[k] * 1 / len(states) for k in states)

	# backward part of the algorithm
	bkw = []
	b_prev = {}
	for i, snp_ind_plus in enumerate(reversed(snpindices[1:]+[None])):
		transition_prob = generate_transition_prob(gen_distances[-i])
		b_curr = {}
		for st in states:
			if i == 0:
				# base case for backward part
				b_curr[st] = 1 / len(states)
			else:
				b_curr[st] = sum(transition_prob(st, l) * emm_prob(l, snp_ind_plus, observations[-i]) * b_prev[l] for l in states)
		bkw.insert(0, b_curr)
		b_prev = b_curr

		if i % 1000 == 0:
			print("CURR BACKWARD IND:", i)

	p_bkw = sum(1/len(states) * emm_prob(l, snpindices[0], observations[0]) * b_curr[l] for l in states)

	# merging the two parts
	posterior = []
	for i in range(len(snpindices)):
		posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states})

	return fwd, posterior

# Adapted from Wikipedia: Viterbi Algorithm
def viterbi(observations, snpindices, gen_distances, states):
	V = [{}]  # sites x states where each matrix space is a dictionary {max_prob:, prev:}
	initial_state_prob = 1/(n1 + n2)
	for st in states:
		V[0][st] = {'prob': initial_state_prob, 'prev': None} # initialize first SNP

	# Run Viterbi when t > 0
	for i in range(1, len(snpindices)):
		transition_prob = generate_transition_prob(gen_distances[i-1])
		snp_ind = snpindices[i]
		V.append({})
		for st in states:
			# finding the max probable path to our state
			max_tr_prob = V[i-1][states[0]]["prob"] + np.log(transition_prob(states[0], st))
			prev_st_selected = states[0]
			previous_states = [prev_st_selected] # need to keep track in case we have multiple max probs
			for prev_st in states[1:]:
				tr_prob = V[i-1][prev_st]["prob"] + np.log(transition_prob(prev_st, st))
				if tr_prob > max_tr_prob:
					max_tr_prob = tr_prob
					prev_st_selected = prev_st
					previous_states = [prev_st_selected] # reintialize the states with the same max probability
				if tr_prob == max_tr_prob: # keep track of multiple max previous states
					previous_states.append(prev_st)

			prev_st_index = np.random.choice(len(previous_states)) # pick the previous state of multiple uniformly at random
			prev_st_selected = previous_states[prev_st_index]
			# fill the SNP x site matrix with max probability of getting to that point (path included in prev)
			max_prob = max_tr_prob + np.log(emm_prob(st, snp_ind, observations[i])) 
			V[i][st] = {"prob": max_prob, "prev": prev_st_selected}
 
	opt_path = []
	# Get most probable state
	last_site_probs = V[-1]
	st = max(last_site_probs, key=lambda x: last_site_probs[x]["prob"])
	opt_path.append(st)

	# Iterate backwards till the first observation
	previous = last_site_probs[st]["prev"]
	for t in range(len(V) - 2, -1, -1): 
		opt_path.insert(0, V[t + 1][previous]["prev"]) # inserting at the front of where you are
		previous = V[t + 1][previous]["prev"]

	return opt_path, max_prob

def phase_observations(observations, snpindices, gen_distances, states):
	fwd, posterior = fwd_bkw(observations, snpindices, gen_distances, states)
	most_probable_states = []
	highest_probs = []
	for hs in posterior:
		max_states = {}
		max_prob = 0
		for state, prob in hs.items():
			if(prob >= max_prob):
				max_prob = prob
		for state, prob in hs.items():
			if(prob == max_prob):
				max_states[state] = prob

		normalized_probs = np.array(list(max_states.values()))/sum(list(max_states.values()))
		state_ind = np.random.choice(list(range(len(max_states))), 1, p=normalized_probs)[0]
		state = list(max_states.keys())[state_ind]

def classify_and_val_inds(indiv_inds, sim_geno_file_loc, sim_ancestry_file_loc, snpindices, gen_distances, states, save_vals, alg='fwd_bkw'):
	outputs = get_ancestry(sim_ancestry_file_loc, snpindices)
	genotypes = get_genotypes(sim_geno_file_loc, snpindices)
	ind_accuracies = []
	paths = []

	for i in indiv_inds:	#for each individual
		preds, prob = classify_new_ind(genotypes[i], snpindices, gen_distances, states, alg)	#get out prediction for individual i

		if save_vals:
			print("********************* SAVING PREDICTIONS *********************")
			np.save('preds_ind_' + str(i) + '.npy', np.array(preds))
			
			print("********************* SAVING TRUE VALS *********************")
			np.save('true_ind_' + str(i) + '.npy', np.array(outputs[i]))

			print("********************* SAVING POSTERIOR *********************")
			np.save('prob_ind_' + str(i) + '.npy', np.array(prob))

		acc = 100 * accuracy_score(outputs[i], preds)

		#print("Ind", i, ":", acc)
		paths.append(preds)
		ind_accuracies.append(acc)
	print("-----------------------------------------------")
	print("Final accuracy: ",  sum(ind_accuracies)/len(ind_accuracies))
	return ind_accuracies, paths

def classify_new_ind(ind_observations, snpindices, gen_distances, states, alg):
	classes = []
	if alg == 'fwd_bkw':
		fwd, posterior = fwd_bkw(ind_observations, snpindices, gen_distances, states)
	
		for hs in posterior:
			population_1 = 0
			population_2 = 0
			for state, prob in hs.items():
				if(state[0] == 1):
					population_1 += prob
				else:
					population_2 += prob
			if population_1 > population_2:
				classes.append(0)
			else:
				classes.append(1)
		return classes, posterior
	elif alg == 'viterbi':
		path, prob = viterbi(ind_observations, snpindices, gen_distances, states)
		classes = [] 
		for st in path: 
			classes.append(st[1] - 1) #States are one of (1, 2) originally for population, need to convert to (0, 1) for validation.
		return classes, prob


# st = time.time()
# # m, h = classify_and_val_inds(load_new_ind_snps(), list(range(num_obs)), gen_dists, gen_hidden_states())
# acc, paths = classify_and_val_inds(indiv_inds=list(range(100)),
# 							sim_geno_file_loc='simulated/simulation',
# 							sim_ancestry_file_loc='simulated/simulation',
# 							snpindices=data_inds,
# 							gen_distances=gen_dists,
# 							states=gen_hidden_states(),
# 							save_vals=False,
# 							alg='viterbi')
# print("TIME", time.time() - st)

# only for simulation purposes
def animate(i, x, y):
    graph.set_data(x[:i+1], y[:i+1])
    return graph

def get_viterbi_paths(indiv_inds, sim_geno_file_loc, sim_ancestry_file_loc, snpindices, gen_distances, states, save_vals, alg='fwd_bkw'):
	outputs = get_ancestry(sim_ancestry_file_loc, snpindices)
	genotypes = get_genotypes(sim_geno_file_loc, snpindices)
	paths = []
	for i in indiv_inds:	#for each individual
		preds, prob = viterbi(genotypes[i], snpindices, gen_distances, states)	#get out prediction for individual i
		paths.append(preds)	
	return paths

paths = get_viterbi_paths(indiv_inds=list(range(100)),
							sim_geno_file_loc='simulated/simulation',
							sim_ancestry_file_loc='simulated/simulation',
							snpindices=data_inds,
							gen_distances=gen_dists,
							states=gen_hidden_states(),
							save_vals=False)

# path is always the same length
path = paths[0]
num_individuals = n1 + n2
x = np.arange(len(path)) # snp on x axis
y = np.arange(num_individuals) # individuals on y axis
c = ['blue' if i < n1 else 'red' for i in range(num_individuals)]

# get the individual who has different ancestries
for i in range(len(paths)):
	print('step', i)
	indivs_from = [st[1] for st in paths[i]] # we need a k > n1 and < n1
	mask = np.array(indivs_from) > n1
	if len(set(mask)) > 1: 
		print('switch at', i)
		break

# background plot
fig = plt.figure()
for i in range(num_individuals):
	plt.axhline(y=i, color=c[i])
plt.title("Ancestry")
plt.xlabel("SNP Site")
plt.ylabel("Individual Index")
plt.xlim(0, len(x))
plt.ylim(0, len(y))

# animation
graph, = plt.plot([], [], 'o-', c='black')
ani = animation.FuncAnimation(fig, animate, frames=len(indivs_from), fargs=(x, indivs_from), interval=400, repeat=False) #set repeat = True to keep going
plt.show()


# curr_state = (1, 1, 10)
# st = time.time()
# for i in range(10000):
# 	next_state = transition(curr_state, np.random.uniform(), 1)
# 	emission = emm_prob(curr_state, i)
# 	if i % 500 == 0:
# 		print('next_state', next_state)
# 		print('emission', emission)
# 	curr_state = next_state

# print(time.time() - st)

# def sample_path():
# 	curr_state = (1, 1, 10)
# 	st = time.time()
# 	for i in range(10000):
# 		next_state = transition(curr_state, np.random.uniform(), 1)
# 		em_prob = emission(curr_state, i)
# 		if i % 500 == 0:
# 			print('next_state', next_state)
# 			print('emission', em_prob)
# 		curr_state = next_state
