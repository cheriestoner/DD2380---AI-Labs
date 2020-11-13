from player import HMM
from constants import *

import json
import pickle

with open('sequences.json', 'r') as f:
    load_dict = json.load(f)

fish_types = load_dict['fish_types']
sequences = load_dict['sequences']


def train(n_states):
    As = [None] * N_FISH
    Bs = [None] * N_FISH
    pis = [None] * N_FISH
    for i in range(N_FISH):
        O = sequences[i]
        hmm = HMM(n_states, N_EMISSIONS)
        As[i], Bs[i], pis[i] = hmm.learning(O)
    return As, Bs, pis


'''def train_and_dump(n_states):
    As, Bs, pis = train(n_states)
    with open(f'trained_{n_states}.pkl', 'wb') as f:
        pickle.dump({'A': As, 'B': Bs, 'pi': pis}, f)'''


for n_states in [2, 3, 4, 5, 6, 7]:
    # train_and_dump(n_states)
    As, Bs, pis = train(n_states)
    with open(f'trained_{n_states}.pkl', 'wb') as f:
        pickle.dump({'A': As, 'B': Bs, 'pi': pis}, f)

fish_ids = [[i for i in range(N_FISH) if fish_types[i] == t] for t in range(N_SPECIES)]

# def cross_dist(fish_type1, fish_type2, n_states):
#     A_dist = [[HMM.mat_dist(As[i], As[j], n_states, n_states) for j in fish_ids[fish_type2]] for i in fish_ids[fish_type1]]
#     B_dist = [[HMM.mat_dist(Bs[i], Bs[j], n_states, N_EMISSIONS) for j in fish_ids[fish_type2]] for i in fish_ids[fish_type1]]
#     size = len(fish_ids[fish_type1]) * len(fish_ids[fish_type2])
#     A_avg = sum(sum(m) for m in A_dist) / size
#     B_avg = sum(sum(m) for m in B_dist) / size
#     return A_avg, B_avg

# print("===============(0, 0)=================")
# print(cross_dist(0, 0))
# print("===============(0, 1)=================")
# print(cross_dist(0, 1))
# print("===============(0, 2)=================")
# print(cross_dist(0, 2))
# print("===============(2, 2)=================")
# print(cross_dist(2, 2))

# for t in range(N_SPECIES):
# for t in [0]:
#     A_dist = [[HMM.mat_dist(As[i], As[j], N_STATES, N_STATES) for j in fish_ids[t]] for i in fish_ids[t]]
#     B_dist = [[HMM.mat_dist(Bs[i], Bs[j], N_STATES, N_EMISSIONS) for j in fish_ids[t]] for i in fish_ids[t]]
#     pi_dist = [[HMM.vec_dist(pis[i], pis[j], N_STATES) for j in fish_ids[t]] for i in fish_ids[t]]
#     print(fish_ids[t])
#     print(A_dist)
#     print(B_dist)
#     print(pi_dist)

# for t in range(N_SPECIES):
for t in [0]:
     lambda_dist = [[HMM.KL_distance(As[i], Bs[i], As[j], Bs[j], pis[i]) for j in fish_ids[t]] for i in fish_ids[t]]
     pi_dist = [[HMM.vec_dist(pis[i], pis[j], N_STATES) for j in fish_ids[t]] for i in fish_ids[t]]
     print(fish_ids[t])
     print(lambda_dist)
     print(pi_dist)