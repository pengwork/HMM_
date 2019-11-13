import numpy as np


class HMM:

    def __init__(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi

    # 前向算法
    def forward(self, obs_seq):
        # A是状态转移矩阵，0位置是行数
        N = self.A.shape[0]
        # T是观测序列的元素个数
        T = len(obs_seq)
        # 初始化由前向向量组成的矩阵
        alpha_mat = np.zeros((N, T))
        alpha_mat[:, 0] = self.pi * self.B[:, obs_seq[0]]
        # 递归计算
        for t in range(1, T):
            for j in range(N):
                alpha_mat[j, t] = np.dot(
                    alpha_mat[:, t-1], (self.A[:, j])) * self.B[j, obs_seq[t]]
        return alpha_mat

    # 后向算法
    def backward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)

        beta_mat = np.zeros((N, T))
        beta_mat[:, -1:] = 1

        for t in reversed(range(T-1)):
            for j in range(N):
                beta_mat[j, t] = np.sum(
                    beta_mat[:, t+1] * self.A[j, :] * self.B[:, obs_seq[t+1]])

        return beta_mat

    def viterbi(self,observationsSeq):
            T = len(observationsSeq)
            N = len(self.pi)
            prePath = np.zeros((T,N),dtype=int)
            dpMatrix = np.zeros((T,N),dtype=float)
            dpMatrix[0,:] = self.pi * self.B[:,observationsSeq[0]]

            for t in range(1,T):
                for n in range(N):
                    probs = dpMatrix[t-1,:] * self.A[:,n] * self.B[n,observationsSeq[t]]
                    prePath[t,n] = np.argmax(probs)
                    dpMatrix[t,n] = np.max(probs)

            maxProb = np.max(dpMatrix[T-1,:])
            maxIndex = np.argmax(dpMatrix[T-1,:])
            path = [maxIndex]

            for t in reversed(range(1,T)):
                path.append(prePath[t,path[-1]])

            path.reverse()
            return maxProb,path

    def baum_welch_train(self, observationsSeq, criterion=0.001):
        T = len(observationsSeq)
        N = len(self.pi)

        while True:
        # alpha_t(i) = P(O_1 O_2 ... O_t, q_t = S_i | hmm)
        # Initialize alpha
            alpha = self.forward(observationsSeq)

            # beta_t(i) = P(O_t+1 O_t+2 ... O_T | q_t = S_i , hmm)
            # Initialize beta
            beta = self.backward(observationsSeq)
            #根据公式求解XIt(i,j) = P(qt=Si,qt+1=Sj | O,λ)
            xi = np.zeros((T-1, N, N), dtype=float)
            for t in range(T-1):
                denominator = np.sum(
                    np.dot(alpha[t, :], self.A) * self.B[:, observationsSeq[t+1]] * beta[t+1, :])
                for i in range(N):
                    molecular = alpha[t, i] * self.A[i, :] * self.B[:, observationsSeq[t+1]] * beta[t+1, :]
                    xi[t, i, :] = molecular / denominator
            #根据xi就可以求出gamma，注意最后缺了一项要单独补上来
            gamma = np.sum(xi, axis=2)
            prod = (alpha[T-1, :] * beta[T-1, :])
            gamma = np.vstack((gamma, prod / np.sum(prod)))
            newpi = gamma[0, :]
            newA = np.sum(xi, axis=0) / np.sum(gamma[:-1, :], axis=0).reshape(-1, 1)
            newB = np.zeros(self.B.shape, dtype=float)

            for k in range(self.B.shape[1]):
                mask = observationsSeq == k

                newB[:, k] = np.sum(gamma[int(mask), :], axis = 0) / np.sum(gamma, axis=0)
            if np.max(abs(self.pi - newpi)) < criterion and \
                    np.max(abs(self.A - newA)) < criterion and \
                    np.max(abs(self.B - newB)) < criterion:
                break

            self.A, self.B, self.pi = newA, newB, newpi



if __name__ == "__main__":

    # 隐状态{'晴天', '阴天', '雨天'}的初始分布
    pi = np.array([0, 1, 0])
    # 状态转移概率矩阵
    A = np.array([[0.2, 0.5, 0.3],
                  [0.3, 0.2, 0.5],
                  [0.4, 0.3, 0.3]])
    # 发射矩阵
    B = np.array([[0.3, 0.3, 0.3, 0.2],
                  [0.2, 0.3, 0.4, 0.2],
                  [0.1, 0.2, 0.1, 0.6]])
    # 初始化观测值，分别代表观测状态集合中{'看电视', '踢足球', '写作业', '上课'}中位置4、3、2的状态
    obs_seq = ([3, 2, 1])
    hmm = HMM(A, B, pi)
    # # 调用前向算法
    # alpha_mat = hmm.forward(obs_seq)
    # prob = np.sum(alpha_mat[:,-1])
    # print("通过前向算法计算出观测序列出现的概率为: %f" %prob)
    # # 调用后向算法
    # beta_mat = hmm.backward(obs_seq)
    # prob = np.sum(beta_mat[:,0] * pi * B[:, obs_seq[0]])
    # print("通过后向算法计算出观测序列出现的概率为: %f" %prob)
    # 调用维特比算法
    max_pro, path = hmm.viterbi(obs_seq)
    print(max_pro, path)
    # # 调用前向 后向算法
    # hmm.baum_welch_train(obs_seq, 0.05)
    # print(hmm.A)
