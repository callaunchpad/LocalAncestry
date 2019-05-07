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

	print(trans_mat[1, 1, 10])
    
	normalized_probs = np.array(list(trans_mat.values()))/sum(list(trans_mat.values()))
	next_hidden_state_ind = np.random.choice(list(range(len(trans_mat))), 1, p=normalized_probs)[0]
	next_hidden_state = list(trans_mat.keys())[next_hidden_state_ind]
	print(max(normalized_probs))