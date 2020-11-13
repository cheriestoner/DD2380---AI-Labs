#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random


class HMM(object):
    EPS = 1e-3
    MAX_ITER = 100

    @staticmethod
    def create_zero_mat(m, n):
        return [[0.0] * n for _ in range(m)]

    @staticmethod
    def nearly_uniform_vec(n):
        from random import gauss
        v = [1.0 + gauss(0, 0.1) for _ in range(n)]
        sum_v = sum(v)
        return [i / sum_v for i in v]    # normalize

    @staticmethod
    def nearly_uniform_mat(m, n):
        return [HMM.nearly_uniform_vec(n) for _ in range(m)]

    @staticmethod
    def mat_dist(A, B, m, n):
        return sum(sum((A[i][j] - B[i][j]) ** 2 for j in range(n)) ** 0.5 for i in range(m)) / m

    @staticmethod
    def vec_dist(a, b, n):
        return sum((a[i] - b[i]) ** 2 for i in range(n)) ** 0.5

    @staticmethod
    def argmin(v):
        min_v = v[0]
        min_id = 0
        for i, x in enumerate(v):
            if x < min_v:
                min_v = x
                min_id = i
        return min_id

    @staticmethod
    def argmax(v):
        man_v = v[0]
        man_id = 0
        for i, x in enumerate(v):
            if x > man_v:
                man_v = x
                man_id = i
        return man_id

    def __init__(self, n_states, n_emissions, eps=1e-3, max_iter=100):
        self.N = n_states
        self.K = n_emissions
        self.A = HMM.nearly_uniform_mat(self.N, self.N)
        self.B = HMM.nearly_uniform_mat(self.N, self.K)
        self.pi = HMM.nearly_uniform_vec(self.N)
        HMM.EPS = eps
        HMM.MAX_ITER = max_iter

    def learning(self, O):
        from math import log
        self.T = len(O)
        self.O = O
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
        sum_di_gamma = HMM.create_zero_mat(self.N, self.N)
        sum_gamma = [0.0] * self.N
        sum_O_k_count_times_gamma = HMM.create_zero_mat(self.N, self.K)
        new_pi = None
        for t in range(self.T-1):
            di_gamma_t = HMM.create_zero_mat(self.N, self.N)
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

        new_A = HMM.create_zero_mat(self.N, self.N)
        new_B = HMM.create_zero_mat(self.N, self.K)
        for i in range(self.N):
            for j in range(self.N):
                new_A[i][j] = sum_di_gamma[i][j] / sum_gamma[i]
            for k in range(self.K):
                new_B[i][k] = sum_O_k_count_times_gamma[i][k] / sum_gamma[i]
        return new_A, new_B, new_pi, c

    def get_log_prob(self, O):
        from math import log
        self.T = len(O)
        self.O = O
        _, c = self.get_scaled_alpha()
        return -sum(log(ci) for ci in c)

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


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    MAX_FLOAT = 1000

    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        self.fish_types = [None] * N_FISH
        self.observations = [[] for _ in range(N_FISH)]
        self.type_counts = [0] * N_SPECIES

        self.N_STATES = 3

        self.hmms = [None] * N_FISH
        self.observation_steps = 109

    def choose_next_fish_id(self):
        # Choose next unguessed fish
        for i in range(N_FISH):
            if self.hmms[i] is None:
                return i

    def find_nearest_fish_type(self, fish_id):
        # Find by nearest distance between A
        nearest_dist = PlayerControllerHMM.MAX_FLOAT
        nearest_id = -1
        for i in range(N_FISH):
            if self.As[i] is None or i == fish_id:
                continue
            dist = HMM.mat_dist(self.As[i], self.As[fish_id], self.N_STATES, self.N_STATES)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_id = i
        if nearest_id == -1 or self.fish_types[nearest_id] is None:    # not find
            # return randint(0, N_EMISSIONS-1)     # randomly guess one
            return HMM.argmin(self.type_counts)    # with highest possibility
        else:
            return self.fish_types[nearest_id]

    def knn_fish_type(self, fish_id, K=3):
        # find nearest 2 fishes
        dists = [(self.fish_types[i], HMM.mat_dist(self.As[i], self.As[fish_id], self.N_STATES, self.N_STATES))
                 for i in range(N_FISH) if self.As[i] is not None and i != fish_id]
        sorted_dists = sorted(dists, key=lambda t: t[1])
        K = min(K, len(sorted_dists))
        k_nearest = [i for i, dist in sorted_dists[:K]]
        if len(k_nearest) == 0:
            return HMM.argmin(self.type_counts)
        else:
            counter = [0] * N_SPECIES
            for t in k_nearest:
                counter[t] += 1
            return HMM.argmax(counter)

    def find_nearest_fish_A_center(self, fish_id):
        # Find by nearest distance between center of A
        nearest_dist = PlayerControllerHMM.MAX_FLOAT
        nearest_id = -1
        dists = [None] * N_SPECIES
        for i, center in enumerate(self.A_centers):
            if center is None:
                continue
            dist = HMM.mat_dist(self.As[fish_id], center, self.N_STATES, self.N_STATES)
            dists[i] = dist
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_id = i
        if nearest_id == -1:
            return HMM.argmin(self.type_counts)
        else:
            return nearest_id

    def find_nearest_fish_B_center(self, fish_id):
        # Find by nearest distance between center of B
        nearest_dist = PlayerControllerHMM.MAX_FLOAT
        nearest_id = -1
        dists = [None] * N_SPECIES
        for i, center in enumerate(self.B_centers):
            if center is None:
                continue
            dist = HMM.mat_dist(self.Bs[fish_id], center, self.N_STATES, N_EMISSIONS)
            dists[i] = dist
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_id = i
        if nearest_id == -1:
            return HMM.argmin(self.type_counts)
        else:
            return nearest_id

    def max_observation_prob(self, fish_id, K=3):
        # a knn approach
        log_probs = []
        for i in range(N_FISH):
            hmm = self.hmms[i]
            if hmm is not None and i != fish_id:
                try:
                    log_prob = hmm.get_log_prob(self.observations[fish_id])
                    log_probs.append((self.fish_types[i], log_prob))
                except ZeroDivisionError:
                    try:
                        hmm.learning(self.observations[i])    # the model needs to be updated
                    except:
                        pass
        sorted_log_probs = sorted(log_probs, key=lambda t: t[1], reverse=True)
        K = min(K, len(sorted_log_probs))
        k_nearest = [i for i, dist in sorted_log_probs[:K]]
        if len(k_nearest) == 0:
            return HMM.argmin(self.type_counts)
        else:
            counter = [0] * N_SPECIES
            for t in k_nearest:
                counter[t] += 1
            return HMM.argmax(counter)

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """

        [l.append(o) for l, o in zip(self.observations, observations)]
        if step < self.observation_steps:
            return None

        fish_id = self.choose_next_fish_id()
        fish_type = None
        hmm = HMM(self.N_STATES, N_EMISSIONS)
        try:
            hmm.learning(self.observations[fish_id])
            self.hmms[fish_id] = hmm
        except:
            return None

        fish_type = self.max_observation_prob(fish_id, K=1)
        return (fish_id, fish_type) if fish_type is not None else None

    def reveal(self, correct, fish_id, true_type):
        """
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """
        # print(f'{fish_id}: {correct}, guess: {self.last_guessed}, true: {true_type}')
        self.fish_types[fish_id] = true_type
        self.type_counts[true_type] += 1
