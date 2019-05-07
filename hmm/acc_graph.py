import numpy as np 
import matplotlib.pyplot as plt

preds = np.load('preds_ind_0.npy')
probs = np.load('prob_ind_0.npy')
true = np.load('true_ind_0.npy')

def graph_it(preds, true, probs):

	def get_real_probs(probs):
		real_probs = []
		for prob in probs:
			pop1_tot = 0
			pop2_tot = 0
			for key, value in prob.items():
				if key[0] == 1:
					pop1_tot += value
				else:
					pop2_tot += value
			mx = max(pop1_tot, pop2_tot)
			real_probs.append(mx)
		return real_probs

	real_probs = get_real_probs(probs)
	plt.plot(true)
	plt.plot(preds)
	plt.plot(real_probs)
	plt.show()

graph_it(preds, true, probs)