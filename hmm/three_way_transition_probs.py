import itertools
import numpy as np

n1 = 100
n2 = 100
n3 = 100

T = 10
mu1 = 0.3
mu2 = 0.2
mu3 = 1 - mu1 - mu2

# Parameters set from HAPMIX paper
p1 = 0.05
p2 = 0.05
p3 = 0.05

ro1 = 60000/n1
ro2 = 90000/n2
ro3 = 70000/n3

theta1 = 0.2/(0.2 + n1)
theta2 = 0.2/(0.2 + n2)
theta3 = 0.01

# Helper functions

#Probability of transitioning between populations
def pois_T(r_s):
	return np.exp(-r_s * T)

#Probability of transitioning inside population
def pois_ro(r_s, ro):
	return np.exp(-r_s * ro)


# Helper for getting relevant variables
def get_relevant_vars(hidden_state):
	(l, m, n) = hidden_state
	if(l == 1):
		(mu_l, ro_l, p_l) = (mu1, ro1, p1)
	elif(l == 2):
		(mu_l, ro_l, p_l) = (mu2, ro2, p2)
	else:
		(mu_l, ro_l, p_l) = (mu3, ro3, p3)
	if(m == 1):
		n_m = n1
	elif(m == 2):
		n_m = n2
	else:
		n_m = n3
	return (mu_l, ro_l, p_l, n_m)

def generate_transition_prob(r_s):
	transition_dict = {}
	states = [1, 2, 3]
	given_states = itertools.product(states, repeat=2)
	given_states = [tuple(x + (1,)) for x in given_states]
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

current_state = (1, 1, 1)
all_states = itertools.product([1, 2, 3], [1, 2, 3], range(1, 101))
all_states = list(all_states)
print(len(all_states))

rs = 0.9
transition_prob = generate_transition_prob(rs)

total_prob = 0
for state in all_states:
	trans_prob = transition_prob(current_state, state)
	total_prob += 
	if(state[0] == 1):
		same_pop_prob == 
print(total_prob)