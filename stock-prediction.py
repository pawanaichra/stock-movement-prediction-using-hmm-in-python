"""
Last update on Nov 02 2017

@author: Pawan Kumar Aichra(15HS20026), Balveer Singh(15HS20011), Sourabh Singhal(15MF3IM14)
"""
import numpy as np
import datetime as dt
from nsepy import get_history

class HMM(object):
    # Implements discrete 1-st order Hidden Markov Model 
	def __init__(self):
		pass

	def forward(self, pi, A, O, observations):
	    N = len(observations)
	    S = len(pi)
	    alpha = np.zeros((N, S))

	    # base case
	    alpha[0, :] = pi * O[:,observations[0]]
	    
	    # recursive case
	    for i in range(1, N):
	        for s2 in range(S):
	            for s1 in range(S):
	                alpha[i, s2] += alpha[i-1, s1] * A[s1, s2] * O[s2, observations[i]]
	    
	    return (alpha, np.sum(alpha[N-1,:]))

	def backward(self, pi, A, O, observations):
	    N = len(observations)
	    S = len(pi)
	    beta = np.zeros((N, S))
	    
	    # base case
	    beta[N-1, :] = 1
	    
	    # recursive case
	    for i in range(N-2, -1, -1):
	        for s1 in range(S):
	            for s2 in range(S):
	                beta[i, s1] += beta[i+1, s2] * A[s1, s2] * O[s2, observations[i+1]]
	    
	    return (beta, np.sum(pi * O[:, observations[0]] * beta[0,:]))


	def baum_welch(self, o, N, rand_seed=1):
	    # Implements HMM Baum-Welch algorithm        
	    T = len(o[0])
	    M = int(max(o[0]))+1 # now all hist time-series will contain all observation vals, but we have to provide for all

	    # Initialise A, B and pi randomly, but so that they sum to one
	    np.random.seed(rand_seed)
	        
	    pi_randomizer = np.ndarray.flatten(np.random.dirichlet(np.ones(N),size=1))/100
	    pi=1.0/N*np.ones(N)-pi_randomizer

	    a_randomizer = np.random.dirichlet(np.ones(N),size=N)/100
	    a=1.0/N*np.ones([N,N])-a_randomizer

	    b_randomizer=np.random.dirichlet(np.ones(M),size=N)/100
	    b = 1.0/M*np.ones([N,M])-b_randomizer

	    pi, A, O = np.copy(pi), np.copy(a), np.copy(b) # take copies, as we modify them
	    S = pi.shape[0]
	    iterations = 1000
	    training = o
	    # do several steps of EM hill climbing
	    for it in range(iterations):
	        pi1 = np.zeros_like(pi)
	        A1 = np.zeros_like(A)
	        O1 = np.zeros_like(O)
	        
	        for observations in training:
	            # compute forward-backward matrices
	            alpha, za = self.forward(pi, A, O, observations)
	            beta, zb = self.backward(pi, A, O, observations)
	            assert abs(za - zb) < 1e-6, "it's badness 10000 if the marginals don't agree"
	            
	            # M-step here, calculating the frequency of starting state, transitions and (state, obs) pairs
	            pi1 += alpha[0,:] * beta[0,:] / za
	            for i in range(0, len(observations)):
	                O1[:, observations[i]] += alpha[i,:] * beta[i,:] / za
	            for i in range(1, len(observations)):
	                for s1 in range(S):
	                    for s2 in range(S):
	                        A1[s1, s2] += alpha[i-1,s1] * A[s1, s2] * O[s2, observations[i]] * beta[i,s2] / za
	                                                                    
	        # normalise pi1, A1, O1
	        pi = pi1 / np.sum(pi1)
	        for s in range(S):
	            A[s, :] = A1[s, :] / np.sum(A1[s, :])
	            O[s, :] = O1[s, :] / np.sum(O1[s, :])
	    return pi, A, O

	def predict(self, stock_prices, states=3):
		(pi, A, O) = self.baum_welch(np.array([stock_prices]), states)
		(alpha, c) = self.forward(pi, A, O, stock_prices)
		# normalize alpha
		row_sums = alpha.sum(axis=1)
		matrix_1 = alpha / row_sums[:, np.newaxis]
		# probability distribution of last hidden state given data
		matrix_2 = matrix_1[-1, :]
		# probability distribution of last hidden state given data
		matrix_3 = np.matmul(matrix_2, A)
		# probabilty distribution of predicted observation state given past observations
		matrix_4= np.matmul(matrix_3, O)
		return(np.argmax(matrix_4))

	def get_optimal_states(self, price_movement):
		accuracy = 0
		for j in range(2,6):
			count=0
			total=0
			for i in range(int(3/4*len(price_movement)), len(price_movement)):
				total=total+1
				predicted=self.predict(price_movement[:i], states=j)
				if(predicted==price_movement[i]):
					count=count+1
			accuracy_this = (count/total*100)
			if(accuracy_this>accuracy):
				accuracy = accuracy_this
				optimal_states = j
		return j
        
def get_stock_prices(company_symbol, start_date, end_date):
    # stock price data from nsepy library (closing prices)
    start_date = dt.datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = dt.datetime.strptime(end_date, '%Y-%m-%d').date()
    stock_prices = get_history(symbol=company_symbol, start=start_date, end=end_date)
    # pandas dataframe to numpy array
    stock_prices = stock_prices.values
    # return closing prices
    return stock_prices[:,7]
        
def get_price_movements(stock_prices):
	price_change = stock_prices[1:] - stock_prices[:-1]
	price_movement = np.array(list(map(lambda x: 1 if x>0 else 0, price_change)))
	return price_movement

if __name__ == '__main__':

    hmm = HMM()
    stock_prices = get_stock_prices('SBIN', '2017-01-01', '2017-04-01')
    price_movement = get_price_movements(stock_prices)
    optimal_states = hmm.get_optimal_states(price_movement)
    prediction = hmm.predict(price_movement, optimal_states)
    if prediction==1:
    	print("You should buy stock.")
    else:
    	print("Sell stock if you have.")

