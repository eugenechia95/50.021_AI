from search import *
from string import *

WORDS = set(i.lower().strip() for i in open('words2.txt'))

def is_valid_word(word):
  return word in WORDS


class WordLadder(Problem):

  def actions(self, state):
    actions = []
    for i in range(len(state)):
      for j in list(string.ascii_lowercase):
        new_string = state[:i] + str(j) + state[i+1:]
        if is_valid_word(new_string):
          actions.append([i, j])
    return actions

  
  def result(self, state, action):
    idx = action[0]
    new_character = action[1]
    new_state = state[:idx] + new_character + state[idx+1:]
    return new_state

  def value(self, state):
    return None

q1_problem = WordLadder("cars", "cats") 
print(breadth_first_tree_search(q1_problem).solution())
print(depth_first_graph_search(q1_problem).solution())

    

