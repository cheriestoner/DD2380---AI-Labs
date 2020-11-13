import random
from time import time

from fishing_game_core.shared import ACTION_TO_STR

WIDTH = 20
HEIGHT = 20
INF = 10000000
NUM_MOVES = 5

class Model(object):

    def __init__(self, initial_data, options):
        random.seed(options['random_seed'])
        self.hash_time = 0
        self.fish_type_mapping = {}
        self.max_depth = options['max_depth']
        self.timeout = options['timeout']
        self.use_hash = options['use_hash']
        self.use_euclidean_dist = options['dist_policy'] == 'euclidean'
        self.corrected_branch_num = options['corrected_branch_num']    # an empirical value indicating how fast the tree expands
        self.heuristic_alpha = options['heuristic_alpha']     # h(score, dist) = score * exp(-ALPHA * dist)
        self.adaptive_beta = options['adaptive_beta']    # if more than BETA of nodes will not be visited in next iteration, we regard it as timeout

        n_agent_types = 2    # player and opponent already counted
        # count fish types and map them into indices
        for k, v in initial_data.items():
            if k.startswith("fish"):
                if self.fish_type_mapping.get(v["score"]) is None:
                    self.fish_type_mapping[v["score"]] = n_agent_types
                    n_agent_types += 1
        self.lookup_table = [[[random.randint(1, (1<<64)-1) for k in range(HEIGHT)] for j in range(WIDTH)] for i in range(n_agent_types)]    # lookup_table[type_of_A_or_B_or_fish][WIDTH][HEIGHT]
        self.player_lookup_table = [random.randint(1, (1<<64)-1) for i in range(2)]
        self.score_cache = {}
        self.node_cache = {}

    @staticmethod
    def pick_one_of_max(v):
        # randomly pick one of the max values, and return its index
        m = None
        idx = []
        for i, x in enumerate(v):
            v = x[1]
            if m is None or v > m:
                m = v
                idx = [x[0]]
            elif x == m:
                idx.append(i)
        return random.choice(idx)

    def get_hash_of_state(self, state, hash_player=False):
        def x_correction(x, x0):
            # We use relative distance of x to player
            # so that more states will be hashed into the same key
            return x - x0 if x >= x0 else x - x0 + WIDTH
        h = 0
        hook_positions = state.hook_positions
        fish_scores = state.fish_scores
        fish_positions = state.fish_positions
        x0, y0 = hook_positions[0]
        x1, y1 = hook_positions[1]

        player_scores = state.player_scores
        current_score = player_scores[0] - player_scores[1]
        h ^= current_score     # we should take the current score into consideration

        if hash_player:
            h ^= self.player_lookup_table[state.player]

        h ^= self.lookup_table[0][0][y0]    # h ^= self.lookup_table[0][x_correction(x0, x0)][y0]
        h ^= self.lookup_table[1][x_correction(x1, x0)][y1]

        for k in fish_positions.keys():
            x, y = fish_positions[k]
            fish_type = self.fish_type_mapping[fish_scores[k]]
            h ^= self.lookup_table[fish_type][x_correction(x, x0)][y]
        return h

    def compute_and_get_score(self, state):
        if not self.use_hash:
            return self.estimate_score(state)

        hash_value = self.get_hash_of_state(state)
        score = self.score_cache.get(hash_value)
        if score is None:
            score = self.estimate_score(state)
            self.score_cache[hash_value] = score
        return score

    def try_get_node_from_cache(self, node):
        if not self.use_hash:
            return node

        hash_value = self.get_hash_of_state(node.state, hash_player=True)
        node_in_cache = self.node_cache.get(hash_value)
        if node_in_cache is None:
            node_in_cache = node
            self.node_cache[hash_value] = node
        return node_in_cache

    def dist_to_pos(self, player_position, opponent_position, pos, player=None):
        def euclidean_dist(x0, y0, x1, y1):
            return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5

        def manhatton_dist(x0, y0, x1, y1):
            return abs(x0 - x1) + abs(y0 - y1)

        player_x, player_y = player_position
        opponent_x, opponent_y = opponent_position
        fish_x, fish_y = pos
        # Use a coordinator in which fish is between player and opponent
        if fish_x <= player_x and fish_x <= opponent_x:
            if player_x < opponent_x:
                opponent_x -= WIDTH
            else:
                player_x -= WIDTH
        elif fish_x >= player_x and fish_x >= opponent_x:
            if player_x < opponent_x:
                player_x += WIDTH
            else:
                opponent_x += WIDTH
        if self.use_euclidean_dist:
            dist_0 = euclidean_dist(player_x, player_y, fish_x, fish_y)
            dist_1 = euclidean_dist(opponent_x, opponent_y, fish_x, fish_y)
        else:
            dist_0 = manhatton_dist(player_x, player_y, fish_x, fish_y)
            dist_1 = manhatton_dist(opponent_x, opponent_y, fish_x, fish_y)
        return dist_0, dist_1

    def estimate_score(self, state):
        def weighted_dist(score, dist):
            from math import exp
            return score / exp(self.heuristic_alpha * dist)  # alpha handles test 1-3

        # this is our heuristics
        scores = state.player_scores
        score_0, score_1 = scores[0], scores[1]
        fish_positions = state.fish_positions
        if not fish_positions:
            return score_0 - score_1
        fish_scores = state.fish_scores
        hook_positions = state.hook_positions
        player_position, opponent_position = hook_positions[0], hook_positions[1]
        for k in fish_positions.keys():
            fish_score = fish_scores[k]
            dist_0, dist_1 = self.dist_to_pos(player_position, opponent_position, fish_positions[k])
            score_0 += weighted_dist(fish_score, dist_0)
            score_1 += weighted_dist(fish_score, dist_1)
        return score_0 - score_1

    def get_next_action(self, root):
        max_depth = self.max_depth
        timeout = self.timeout
        self.almost_timeout = False
        self.num_visited = 0
        start = time()

        def order_children(children, ascending=False):
            return [c[0] for c in sorted(children, key=lambda x: x[1], reverse=not ascending)]

        def order_children_by_heuristic(children, ascending=False):
            return [c for c in sorted(children, key=lambda c: self.compute_and_get_score(c.state), reverse=not ascending)]

        def expand_children(new_children, children):
            existing_children = [c[0] for c in new_children]
            return existing_children + [c for c in children if c not in existing_children]
        
        def get_path_of_node(node):
            moves = []
            p = node
            while p.parent is not None:
                moves.append(p.move)
                p = p.parent
            return moves

        def alpha_beta_one_iter(node, alpha, beta, depth, move=None):
            node = self.try_get_node_from_cache(node)     # Jesus! It saves a lot of time!
            state = node.state
            children = node.compute_and_get_children()
            is_terminated = len(children) == 0
            self.almost_timeout = self.almost_timeout or time() - start > timeout
            if depth <= 0 or is_terminated or self.almost_timeout:
                score = self.compute_and_get_score(state)
                self.num_visited += 1
                return score, move
            new_children = []
            if state.player == 0:    # MAX
                v = -INF
                for child in children:
                    score, _ = alpha_beta_one_iter(child, alpha, beta, depth-1, child.move)
                    if score > v:
                        v, move = score, child.move
                    new_children.append((child, score))
                    alpha = max(v, alpha)
                    if alpha >= beta:
                        break
                if len(new_children) == NUM_MOVES:    # can still visited all children
                    node.children = order_children(new_children, ascending=False)    # move ordering
                else:
                    node.children = order_children_by_heuristic(children, ascending=False)
            else:    # MIN
                v = INF
                for child in children:
                    score, _ = alpha_beta_one_iter(child, alpha, beta, depth-1, child.move)
                    if score < v:
                        v, move = score, child.move
                    new_children.append((child, score))
                    beta = min(v, beta)
                    if alpha >= beta:
                        break
                if len(new_children) == NUM_MOVES:
                    node.children = order_children(new_children, ascending=True)    # move ordering
                else:
                    node.children = order_children_by_heuristic(children, ascending=True)
            return v, move

        def iterative_deepening(node, max_depth):
            v = n = None
            for depth in range(1, max_depth+1):
                v, move = alpha_beta_one_iter(node, -INF, INF, depth, None)
                if not self.almost_timeout:
                    time_consumption = time() - start
                    estimated_time = time_consumption * self.corrected_branch_num
                    print("time_consumption: {}, estimated_time: {}".format(time_consumption, estimated_time))
                    # adaptively change iteration number
                    self.almost_timeout = time_consumption > timeout \
                        or (time_consumption + estimated_time - timeout) / estimated_time > self.adaptive_beta
                if self.almost_timeout:
                    break
            return ACTION_TO_STR[move] if move is not None else random.randrange(5)

        if root.state.player_caught[0] != -1:
            return 1    # trivial case: if a fish is caught, return "UP"
        return iterative_deepening(root, max_depth)


from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController

class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate game tree object
        first_msg = self.receiver()
        # Initialize your minimax model
        model = self.initialize_model(initial_data=first_msg)

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(
                model=model, initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def initialize_model(self, initial_data):
        """
        Initialize your minimax model 
        :param initial_data: Game data for initializing minimax model
        :type initial_data: dict
        :return: Minimax model
        :rtype: object
        Sample initial data:
        { 'fish0': {'score': 11, 'type': 3}, 
          'fish1': {'score': 2, 'type': 1}, 
          ...
          'fish5': {'score': -10, 'type': 4},
          'game_over': False }
        Please note that the number of fishes and their types is not fixed between test cases.
        """
        options = {
            'max_depth': 7,
            'timeout': 0.065,
            'use_hash': True,
            'dist_policy': 'euclidean',
            'corrected_branch_num': 2.6,
            'heuristic_alpha': 0.8,
            'adaptive_beta': 0.32,
            'random_seed': 12
        }
        return Model(initial_data, options=options)

    def search_best_next_move(self, model, initial_tree_node):
        """
        Use your minimax model to find best possible next move for player 0 (green boat)
        :param model: Minimax model
        :type model: object
        :param initial_tree_node: Initial game tree node 
        :type initial_tree_node: game_tree.Node 
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # s = time()
        res = model.get_next_action(initial_tree_node)
        # res = model.next_move(initial_tree_node, max_time=10)
        # e = time()
        # print(e - s)
        return res