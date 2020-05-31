import pickle
import random

from collections import defaultdict, Counter
from isolation import Isolation

NUM_ROUNDS = 10000000
MAX_DEPTH = 4

def build_table(num_rounds=NUM_ROUNDS):
    # Builds a table that maps from game state -> action
    # by choosing the action that accumulates the most
    # wins for the active player. (Note that this uses
    # raw win counts, which are a poor statistic to
    # estimate the value of an action; better statistics
    # exist.)
    book = defaultdict(lambda : (Counter(), Counter()))
    #plays = defaultdict(Counter)
    for _ in range(num_rounds):
        state = Isolation()
        build_tree(state, book, MAX_DEPTH)

    states_dict = {}
    for state, counters in book.items():
        best_action = None
        max_win_ratio = 0
        for action, wins in counters[0].items():
            tries = counters[1][action]
            win_ratio = wins / tries
            if win_ratio >= 0.5 and win_ratio > max_win_ratio:
                best_action = action
                max_win_ratio = win_ratio
        if best_action is not None:
            states_dict[state] = best_action
    return states_dict

def build_tree(state, book, depth=4):
    if state.terminal_test() or depth <= 0:
        return not simulate(state)
    #if state.terminal_test() or depth <= 0:
    #    return -baseline(state)
    action = random.choice(state.actions())
    has_won = build_tree(state.result(action), book, depth - 1)
    book[state][0][action] += 1 if has_won else 0
    book[state][1][action] += 1
    #plays[state][action] += 1
    #if book[state][action]:
    #    book[state][action] = (book[state][action][0] + reward, book[state][action][1] + 1)
    #else:
    #    book[state][action] = (reward, 1)
    return not has_won

def simulate(state):
    player_id = state.player()
    while not state.terminal_test():
        state = state.result(random.choice(state.actions()))
    #return -1 if state.utility(player_id) < 0 else 1
    return False if state.utility(player_id) < 0 else True

# #my_moves - #opponent_moves heuristic
def baseline(state):
    own_loc = state.locs[state.player()]
    opp_loc = state.locs[1 - state.player()]
    own_liberties = state.liberties(own_loc)
    opp_liberties = state.liberties(opp_loc)
    return len(own_liberties) - len(opp_liberties)

opening_book = build_table()
with open("data.pickle", 'wb') as f:
    pickle.dump(opening_book, f)