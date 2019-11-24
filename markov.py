import numpy as np
def markov():
    init_state = np.array([1, 0, 0])
    transfer_matrix = np.array([[0.2, 0.5, 0.3],
                               [0.3, 0.2, 0.5],
                               [0.4, 0.3, 0.3]])
    start = init_state
    for i in range(9):
        res = np.dot(start, transfer_matrix)
        print("第",i+2, "天的分布为：", res)
        start = res
if __name__ == '__main__':
    markov()