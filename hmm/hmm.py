import numpy as np
import time
from import_data import *

# Get reference population data 
pop1 = get_genotypes("CEU") 
pop2 = get_genotypes("YRI") 
print('lengths', len(pop1), len(pop2))

# Define hyperparameters (param1 for European, param2 for African)
n1 = 100
n2 = 100

T = 10
mu1 = 0.5
mu2 = 1 - mu1

# Parameters set from HAPMIX paper
p1 = 0.05
p2 = 0.05
ro1 = 60000/n1
ro2 = 90000/n2
theta1 = 0.2/(0.2 + n1)
theta2 = 0.2/(0.2 + n2)
theta3 = 0.01

# Helper functions
def pois_T(r_s):
	return np.exp(-r_s * T)

def pois_ro(r_s, ro):
	return np.exp(-r_s * ro)

def indicator(pop, indiv, value, site):
	data = (pop1 if pop == 1 else pop2)	
	return data[indiv][site] == value

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

# Transition state function
def transition(curr_state, r_s, obs):
	"""
	curr_state: (i, j, k) triple of values denoting current hidden state
	r_s: genetic distance between current pair of SNP sites
	obs: observed value

	returns: (l, m, n) triple of values corresponding to next hidden state
	"""
	(i, j, k) = curr_state
	trans_mat = dict()
	poss_hidden_states = gen_hidden_states()

	counts = [0 for i in range(6)]
	for hidden_state in poss_hidden_states:
		mu_l, ro_l, p_l, n_m = get_relevant_vars(hidden_state)
		(l, m, n) = hidden_state

		if l != i and m == l:
			trans_mat[hidden_state] = (1 - pois_T(r_s))*mu_l * (1 - p_l)/n_m
			counts[0] += 1
		elif l != i and m != l:
			trans_mat[hidden_state] = (1 - pois_T(r_s))*mu_l * p_l/n_m
			counts[1] += 1
		elif l == i and m == l and (j != m or k != n):
			trans_mat[hidden_state] = pois_T(r_s) * (1 - pois_ro(r_s, ro_l)) * (1 - p_l)/n_m + (1 - pois_T(r_s))*mu_l * (1 - p_l)/n_m
			counts[2] += 1
		elif l == i and m == l and j == m and k == n:
			trans_mat[hidden_state] = pois_T(r_s) * pois_ro(r_s, ro_l) + pois_T(r_s) * (1 - pois_ro(r_s, ro_l)) * (1 - p_l)/n_m + (1 - pois_T(r_s))*mu_l * (1 - p_l)/n_m
			counts[3] += 1
		elif l == i and m != l and (j != m or k != n):
			trans_mat[hidden_state] = pois_T(r_s) * (1 - pois_ro(r_s, ro_l)) * p_l/n_m + (1 - pois_T(r_s))*mu_l * p_l/n_m
			counts[4] += 1
		elif l == i and m != l and j == m and k == n:
			trans_mat[hidden_state] = pois_T(r_s) * pois_ro(r_s, ro_l) + pois_T(r_s) * (1 - pois_ro(r_s, ro_l)) * p_l/n_m + (1 - pois_T(r_s))*mu_l * p_l/n_m
			counts[5] += 1

	# print(set(trans_mat.values()))

	normalized_probs = np.array(list(trans_mat.values()))/sum(list(trans_mat.values()))
	next_hidden_state_ind = np.random.choice(list(range(len(trans_mat))), 1, p=normalized_probs)[0]
	next_hidden_state = list(trans_mat.keys())[next_hidden_state_ind]

	return next_hidden_state

def emission(curr_state, site):
	# population j is 1 for european, 2 for african 
	(i, j, k) = curr_state
	thetai = (theta1 if i == 1 else theta2)
	if i == j: 
		return thetai * indicator(j, k, 0, site) + (1 - thetai) * indicator(j, k, 1, site)
	else: 
		return theta3 * indicator(j, k, 0, site) + (1 - theta3) * indicator(j, k, 1, site)

# Transition state function
def transition_prob(curr_state, given_state, r_s):
	"""
	curr_state: (i, j, k) triple of values denoting current hidden state
	r_s: genetic distance between current pair of SNP sites
	obs: observed value

	returns: probability of transitioning from curr_state to given_state
	"""
	(i, j, k) = curr_state
	(l, m, n) = given_state

	mu_l, ro_l, p_l, n_m = get_relevant_vars(given_state)

	if l != i and m == l:
		return (1 - pois_T(r_s))*mu_l * (1 - p_l)/n_m
	elif l != i and m != l:
		return (1 - pois_T(r_s))*mu_l * p_l/n_m
	elif l == i and m == l and (j != m or k != n):
		return pois_T(r_s) * (1 - pois_ro(r_s, ro_l)) * (1 - p_l)/n_m + (1 - pois_T(r_s))*mu_l * (1 - p_l)/n_m
	elif l == i and m == l and j == m and k == n:
		return pois_T(r_s) * pois_ro(r_s, ro_l) + pois_T(r_s) * (1 - pois_ro(r_s, ro_l)) * (1 - p_l)/n_m + (1 - pois_T(r_s))*mu_l * (1 - p_l)/n_m
	elif l == i and m != l and (j != m or k != n):
		return pois_T(r_s) * (1 - pois_ro(r_s, ro_l)) * p_l/n_m + (1 - pois_T(r_s))*mu_l * p_l/n_m
	elif l == i and m != l and j == m and k == n:
		return pois_T(r_s) * pois_ro(r_s, ro_l) + pois_T(r_s) * (1 - pois_ro(r_s, ro_l)) * p_l/n_m + (1 - pois_T(r_s))*mu_l * p_l/n_m

			
# Adapted from Wikipedia: Forward-Backward Algorithm
def fwd_bkw(observations, gen_distances, states, transition_prob, emm_prob):
	"""
	observations: array of 1/0s
	states: array of states
	start_prob: dictionary of key: states, value: probability
	trans_prob(curr_state, next_state): function which gives probability between two states, returns float
	emm_prob(curr_state, obs): function which gives probability of emissions, returns float	
	"""
	#observations ---> [(r_s, emssions)]
	# forward part of the algorithm

	fwd = []
	f_prev = {}
	for i, observation_i in enumerate(observations):
		f_curr = {}
		for st in states:
			if i == 0:
				# base case for the forward part
				prev_f_sum = 1/len(states)
			else:
				prev_f_sum = sum(f_prev[k]*transition_prob(k, st, gen_distances[i-1]) for k in states)
			f_curr[st] = emm_prob(st, observation_i) * prev_f_sum

		fwd.append(f_curr)
		f_prev = f_curr

	p_fwd = sum(f_curr[k] * 1 / len(states) for k in states)

	# backward part of the algorithm
	bkw = []
	b_prev = {}
	for i, observation_i_plus in enumerate(reversed(observations[1:]+[None])):
		b_curr = {}
		for st in states:
			if i == 0:
				# base case for backward part
				b_curr[st] = 1 / len(states)
			else:
				b_curr[st] = sum(transition_prob(st, l, gen_distances[-i]) * emm_prob(l, observation_i_plus) * b_prev[l] for l in states)

		bkw.insert(0, b_curr)
		b_prev = b_curr

	p_bkw = sum(1/len(states) * emm_prob(l, observations[0]) * b_curr[l] for l in states)

	# merging the two parts
	posterior = []
	for i in range(len(observations)):
		posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states})

	print(p_fwd, p_bkw)
	return fwd, posterior

def phase_observations(observations, gen_distances, states, transition_prob, emm_prob):
	fwd, posterior = fwd_bkw(observations, gen_distances, states, transition_prob, emm_prob)
	most_probable_states = []
	for hs in posterior:
		max_state = None
		max_prob = 0
		for state, prob in hs.items():
			if(prob > max_prob):
				max_prob = prob
				max_state = state
		most_probable_states.append(max_state)

def sample_path():
	curr_state = (1, 1, 10)
	st = time.time()
	for i in range(10000):
		next_state = transition(curr_state, np.random.uniform(), 1)
		em_prob = emission(curr_state, i)
		if i % 500 == 0:
			print('next_state', next_state)
			print('emission', em_prob)
		curr_state = next_state

	print(time.time() - st)

trans_prob_matrix = [[0.05, 0.9, 0.05], [0.9, 0.01, 0.09], [0.1, 0.1, 0.8]]
trans_prob = lambda x, y, _: trans_prob_matrix[x][y]
states = list(range(3))
def emmission_prob(x, y):
	if(y == 0):
		if(x < 2):
			return 0.9
		else:
			return 0.1
	if(y == 1):
		if(x >= 2):
			return 0.9
		else:
			return 0.1

def dict_sum(states):
	total = 0
	for state, prob in states.items():
		total += prob
	return total

observations = [0, 0, 0, 0, 1, 1, 1, 1]
gen_distances = list(range(1000)) #dummy variable
fwd, posterior = fwd_bkw(observations, gen_distances, states, trans_prob, emmission_prob)

for hm in posterior:
	print(hm, dict_sum(hm))

print("printing forward pass")
for hm in fwd:
	print(hm, dict_sum(hm))

most_probable_states = []
for hs in posterior:
	max_state = None
	max_prob = 0
	for state, prob in hs.items():
		if(prob > max_prob):
			max_prob = prob
			max_state = state
	most_probable_states.append(max_state)
print(most_probable_states)



