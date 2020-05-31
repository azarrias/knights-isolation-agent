
from sample_players import DataPlayer

DEBUG_MODE = False

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        import random

        if DEBUG_MODE:
          import time
          from isolation import DebugState

          print("\nTurn " + str(state.ply_count) + " - Player " + str(self.player_id + 1) + " goes:")
          print("Available actions: " + ','.join(map(str, state.actions())))
          debug_board = DebugState.from_state(state)
          print(debug_board)
          start = time.process_time()

        if state.ply_count <= 4:
          if self.data is not None and state in self.data and self.data[state] in state.actions():
            action = self.data[state]
            if DEBUG_MODE:
              print("Pick action from opening book: " + str(action))
            self.queue.put(action)
          else:
            self.queue.put(random.choice(state.actions()))
        else:
          depth_limit = 100
          for depth in range(1, depth_limit + 1):
            action = self.alpha_beta_search(state, depth)
            if action is None:
              action = random.choice(state.actions())
            if DEBUG_MODE:
              print("Calculate best action from minimax (depth=" + str(depth) + ") in " + str(time.process_time() - start) + " seconds: " + str(action))
            self.queue.put(action)

    def alpha_beta_search(self, state, depth):
      """ Return the move along a branch of the game tree that
      has the best possible value.  A move is a pair of coordinates
      in (column, row) order corresponding to a legal move for
      the searching player.
      """
      alpha = float("-inf")
      beta = float("inf")
      best_score = float("-inf")
      best_move = None
      for a in state.actions():
        v = self.min_value(state.result(a), alpha, beta, depth - 1)
#        print("a = " + str(a) + " | v = " + str(v))
        alpha = max(alpha, v)
        if v > best_score:
          best_score = v
          best_move = a
#      if DEBUG_MODE:
#        print("AB Search - best_score = " + str(best_score) + " | alpha = " + str(alpha) + " | beta = " + str(beta) + " | actions = " + str(state.actions()))
      return best_move

    def min_value(self, state, alpha, beta, depth):
      """ Return the game state utility if the game is over,
      otherwise return the minimum value over all legal successors
      """
      if state.terminal_test():
#        if DEBUG_MODE:
#          print("min value - terminal node with value " + str(state.utility(self.player_id)))
        return state.utility(self.player_id)
      if depth <= 0:
#        if DEBUG_MODE:
#          print("min value - heuristic value " + str(self.baseline(state)))
        return self.baseline(state)
        #return self.weighted_attacking(state, 3)
        #return self.weighted_defensive(state, 3)
        #return self.weighted_increasingly_attacking(state, 3)
        #return self.weighted_increasingly_defensive(state, 3)
        #return self.defensive_to_attacking(state, 3)
        #return self.attacking_to_defensive(state, 3)
      v = float("inf")
      for a in state.actions():
        v = min(v, self.max_value(state.result(a), alpha, beta, depth - 1))
#        if DEBUG_MODE:
#          print("min value - depth = " + str(depth) + " | a = " + str(a) + " | v = " + str(v) + " | alpha = " + str(alpha) + " | beta = " + str(beta) + " | actions = " + str(state.actions()))
        if v <= alpha:
          return v
        beta = min(beta, v)
      return v

    def max_value(self, state, alpha, beta, depth):
      """ Return the game state utility if the game is over,
      otherwise return the maximum value over all legal successors
      """
      if state.terminal_test():
#        if DEBUG_MODE:
#          print("max value - terminal node with value " + str(state.utility(self.player_id)))
        return state.utility(self.player_id)
      if depth <= 0:
#        if DEBUG_MODE:
#          print("max value - heuristic value " + str(self.baseline(state)))
        return self.baseline(state)
        #return self.weighted_attacking(state, 3)
        #return self.weighted_defensive(state, 3)
        #return self.weighted_increasingly_attacking(state, 3)
        #return self.weighted_increasingly_defensive(state, 3)
        #return self.defensive_to_attacking(state, 3)
        #return self.attacking_to_defensive(state, 3)
      v = float("-inf")
      for a in state.actions():
        v = max(v, self.min_value(state.result(a), alpha, beta, depth - 1))
#        if DEBUG_MODE:
#          print("max value - depth = " + str(depth) + " | a = " + str(a) + " | v = " + str(v) + " | alpha = " + str(alpha) + " | beta = " + str(beta) + " | actions = " + str(state.actions()))
        if v >= beta:
          return v
        alpha = max(alpha, v)
      return v

    # #my_moves - #opponent_moves heuristic
    def baseline(self, state):
      own_loc = state.locs[self.player_id]
      opp_loc = state.locs[1 - self.player_id]
      own_liberties = state.liberties(own_loc)
      opp_liberties = state.liberties(opp_loc)
      return len(own_liberties) - len(opp_liberties)

    # attacking heuristic
    # variation of the baseline heuristic
    # it is more important to reduce opponents moves
    # than to increase our player's own moves
    def weighted_attacking(self, state, weight):
      own_loc = state.locs[self.player_id]
      opp_loc = state.locs[1 - self.player_id]
      own_liberties = state.liberties(own_loc)
      opp_liberties = state.liberties(opp_loc)
      return len(own_liberties) - weight * len(opp_liberties)

    # defensive heuristic
    # variation of the baseline heuristic
    # it is more important to have available moves
    # than to reduce the opponents moves
    def weighted_defensive(self, state, weight):
      own_loc = state.locs[self.player_id]
      opp_loc = state.locs[1 - self.player_id]
      own_liberties = state.liberties(own_loc)
      opp_liberties = state.liberties(opp_loc)
      return weight * len(own_liberties) - len(opp_liberties)

    # increase importance of moves as the game goes forward
    # increasingly attacking player
    def weighted_increasingly_attacking(self, state, weight):
      from isolation.isolation import _WIDTH
      from isolation.isolation import _HEIGHT
      board_size = _WIDTH * _HEIGHT
      progress = state.ply_count / board_size
      own_loc = state.locs[self.player_id]
      opp_loc = state.locs[1 - self.player_id]
      own_liberties = state.liberties(own_loc)
      opp_liberties = state.liberties(opp_loc)
      return len(own_liberties) - weight * len(opp_liberties) * progress

    # increase importance of moves as the game goes forward
    # increasingly defensive player
    def weighted_increasingly_defensive(self, state, weight):
      from isolation.isolation import _WIDTH
      from isolation.isolation import _HEIGHT
      board_size = _WIDTH * _HEIGHT
      progress = state.ply_count / board_size
      own_loc = state.locs[self.player_id]
      opp_loc = state.locs[1 - self.player_id]
      own_liberties = state.liberties(own_loc)
      opp_liberties = state.liberties(opp_loc)
      return weight * len(own_liberties) * progress - len(opp_liberties)

    # will prioritize playing defense on the first half of the game
    # and prioritize playing attack on the second half
    def defensive_to_attacking(self, state, weight):
      from isolation.isolation import _WIDTH
      from isolation.isolation import _HEIGHT
      board_size = _WIDTH * _HEIGHT
      progress = state.ply_count / board_size
      if progress <= 0.5:
        return self.weighted_defensive(state, weight)
      else:
        return self.weighted_attacking(state, weight)

    # will prioritize playing defense on the first half of the game
    # and prioritize playing attack on the second half
    def attacking_to_defensive(self, state, weight):
      from isolation.isolation import _WIDTH
      from isolation.isolation import _HEIGHT
      board_size = _WIDTH * _HEIGHT
      progress = state.ply_count / board_size
      if progress <= 0.5:
        return self.weighted_attacking(state, weight)
      else:
        return self.weighted_defensive(state, weight)

    # distance to opponent heuristic
    def opponent_distance(self, state):
      own_loc = self.ind2xy(state.locs[self.player_id])
      opp_loc = self.ind2xy(state.locs[1 - self.player_id])
      return self.manhattanDistance(own_loc, opp_loc)

    def manhattanDistance(self, xy1, xy2):
      "Returns the Manhattan distance between points xy1 and xy2"
      return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

    def euclideanDistance(self, xy1, xy2):
      "Returns the Euclidean distance between points xy1 and xy2"
      from math import sqrt
      return sqrt((xy1[0] - xy2[0]) ** 2 + (xy1[1] -xy2[1]) ** 2)

    def ind2xy(self, ind):
      """ Convert from board index value to xy coordinates

      The coordinate frame is 0 in the bottom right corner, with x increasing
      along the columns progressing towards the left, and y increasing along
      the rows progressing towards teh top.
      """
      from isolation.isolation import _WIDTH
      from isolation.isolation import _HEIGHT
      return (ind % (_WIDTH + 2), ind // (_WIDTH + 2))