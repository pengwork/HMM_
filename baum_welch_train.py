def baum_welch_train(self, observations, criterion=0.05):
    n_states = self.A.shape[0]
    n_samples = len(observations)
 
    done = False
    while not done:
        # alpha_t(i) = P(O_1 O_2 ... O_t, q_t = S_i | hmm)
        # Initialize alpha
        alpha = self._forward(observations)
 
        # beta_t(i) = P(O_t+1 O_t+2 ... O_T | q_t = S_i , hmm)
        # Initialize beta
        beta = self._backward(observations)
 
        xi = np.zeros((n_states,n_states,n_samples-1))
        for t in range(n_samples-1):
            denom = np.dot(np.dot(alpha[:,t].T, self.A) * self.B[:,observations[t+1]].T, beta[:,t+1])
            for i in range(n_states):
                numer = alpha[i,t] * self.A[i,:] * self.B[:,observations[t+1]].T * beta[:,t+1].T
                xi[i,:,t] = numer / denom
 
        # gamma_t(i) = P(q_t = S_i | O, hmm)
        gamma = np.sum(xi,axis=1)
        # Need final gamma element for new B
        prod =  (alpha[:,n_samples-1] * beta[:,n_samples-1]).reshape((-1,1))
        gamma = np.hstack((gamma,  prod / np.sum(prod))) #append one more to gamma!!!
 
        newpi = gamma[:,0]
        newA = np.sum(xi,2) / np.sum(gamma[:,:-1],axis=1).reshape((-1,1))
        newB = np.copy(self.B)
 
        num_levels = self.B.shape[1]
        sumgamma = np.sum(gamma,axis=1)
        for lev in range(num_levels):
            mask = observations == lev
            newB[:,lev] = np.sum(gamma[:,mask],axis=1) / sumgamma
 
        if np.max(abs(self.pi - newpi)) < criterion and \
                        np.max(abs(self.A - newA)) < criterion and \
                        np.max(abs(self.B - newB)) < criterion:
            done = 1
 
        self.A[:],self.B[:],self.pi[:] = newA,newB,newpi