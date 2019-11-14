def simulate(self, T):
 
    def draw_from(probs):
        return np.where(np.random.multinomial(1,probs) == 1)[0][0]
 
    observations = np.zeros(T, dtype=int)
    states = np.zeros(T, dtype=int)
    states[0] = draw_from(self.pi)
    observations[0] = draw_from(self.B[states[0],:])
    for t in range(1, T):
        states[t] = draw_from(self.A[states[t-1],:])
        observations[t] = draw_from(self.B[states[t],:])
    return observations,states