from math import log, exp

def str_to_float_mat(A, m, n):
    A = [float(i) for i in A[2:]]
    return [A[i*n:i*n+n] for i in range(m)]


def str_to_float_vec(A):
    return [float(i) for i in A[2:]]


def create_zero_mat(m, n):
    return [[0.0] * n for _ in range(m)]

def argmax(v):
    return max_argmax(v)[1]


def max_argmax(v):
    idx = 0
    max_ = v[0]
    for i, x in enumerate(v):
        if x > max_:
            idx = i
            max_ = x
    return max_, idx


def repr_vec(v):
    return '{} {}'.format(len(v), ' '.join(str(i) for i in v))


def repr_mat(A):
    return '{} {} {}'.format(len(A), len(A[0]), ' '.join(' '.join(str(v) for v in l) for l in A))


class HMM(object):
    EPS = 1e-3
    MAX_ITER = 500

    def __init__(self, A, B, pi):
        self.N = int(A[0])
        self.K = int(B[1])
        self.A = str_to_float_mat(A, self.N, self.N)
        self.B = str_to_float_mat(B, self.N, self.K)
        self.pi = str_to_float_vec(pi)

    @staticmethod
    def converged_diff(A, B):
        # Difference in any element is greater than EPS
        if isinstance(A[0], list):
            for l1, l2 in zip(A, B):
                for a, b in zip(l1, l2):
                    if abs(a - b) > HMM.EPS:
                        return False
        else:
            for a, b in zip(A, B):
                if abs(a - b) > HMM.EPS:
                    return False
        return True

    def learning(self, O):
        self.T = int(O[0])
        self.O = [int(i) for i in O[1:]]
        old_log_prob = 0

        for iteration in range(HMM.MAX_ITER):
            new_A, new_B, new_pi, c = self.learning_iter()
            log_prob = -sum(log(ci) for ci in c)
            self.A, self.B, self.pi = new_A, new_B, new_pi
            if iteration != 0 and (log_prob <= old_log_prob or abs(log_prob - old_log_prob) <= HMM.EPS):
                break
            old_log_prob = log_prob
        return self.A, self.B, self.pi


    def learning_iter(self):
        # calculate alpha, beta, di_gamma, gamma
        alpha, c = self.get_scaled_alpha()
        beta = self.get_scaled_beta(c)
        # di_gamma and gamma are computed at each time
        # sum of di_gamma and gamma over time 0:T-1
        sum_di_gamma = create_zero_mat(self.N, self.N)
        sum_gamma = [0.0] * self.N
        sum_O_k_count_times_gamma = create_zero_mat(self.N, self.K)
        new_pi = None
        for t in range(self.T-1):
            di_gamma_t = create_zero_mat(self.N, self.N)
            gamma_t = [0.0] * self.N
            O_t = self.O[t]
            O_t_plus_1 = self.O[t+1]
            for i in range(self.N):
                for j in range(self.N):
                    di_gamma_t[i][j] = beta[t+1][j] * self.A[i][j] * self.B[j][O_t_plus_1] * alpha[t][i]
                    gamma_t[i] += di_gamma_t[i][j]
                    sum_di_gamma[i][j] += di_gamma_t[i][j]
                sum_gamma[i] += gamma_t[i]
                sum_O_k_count_times_gamma[i][O_t] += gamma_t[i]
            if t == 0:
                new_pi = gamma_t
        
        new_A = create_zero_mat(self.N, self.N)
        new_B = create_zero_mat(self.N, self.K)
        for i in range(self.N):
            for j in range(self.N):
                new_A[i][j] = sum_di_gamma[i][j] / sum_gamma[i]
            for k in range(self.K):
                new_B[i][k] = sum_O_k_count_times_gamma[i][k] / sum_gamma[i]
        return new_A, new_B, new_pi, c

    def get_scaled_alpha(self):
        alpha = [None] * self.T
        c = [0.0] * self.T
        # t = 0
        O_t = self.O[0]
        alpha_t = [self.pi[i] * self.B[i][O_t] for i in range(self.N)]
        c[0] = 1.0 / sum(alpha_t)
        alpha[0] = [a * c[0] for a in alpha_t]
        # t = 1..T
        for t in range(1, self.T):
            O_t = self.O[t]
            alpha_t = [sum(alpha[t-1][j] * self.A[j][i] for j in range(self.N)) * self.B[i][O_t] for i in range(self.N)]
            c[t] = 1.0 / sum(alpha_t)
            alpha[t] = [a * c[t] for a in alpha_t]
        return alpha, c
    
    def get_scaled_beta(self, c):
        beta = [None] * self.T
        # t = T
        beta[self.T-1] = [c[self.T-1]] * self.N
        # t = T-1..1
        for t in range(self.T-2, -1, -1):
            O_t_plus_1 = self.O[t+1]
            beta[t] = [sum(self.B[j][O_t_plus_1] * self.A[i][j] * beta[t+1][j] for j in range(self.N)) * c[t] for i in range(self.N)]
        return beta


def main():
    A = input().split()
    B = input().split()
    pi = input().split()
    hmm = HMM(A, B, pi)

    O = input().split()
    A, B, pi = hmm.learning(O)
    print(repr_mat(A))
    print(repr_mat(B))
    

main()