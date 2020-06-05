from search import *
from string import *

WORDS = set(i.lower().strip() for i in open('words2.txt'))

def is_valid_word(word):
  return word in WORDS


class WordLadder(Problem):

  # every action is in the form of an array [<index of string to replace>,<new char>]
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

q1_problem_1 = WordLadder("cars", "cats") 
q1_problem_2 = WordLadder("cold", "warm")
q1_problem_3 = WordLadder("best", "math")

print("'cars' -> 'cats':" + str(breadth_first_tree_search(q1_problem_1).solution()))
print("'cold' -> 'warm':" + str(breadth_first_tree_search(q1_problem_2).solution()))
print("'best' -> 'math':" + str(breadth_first_tree_search(q1_problem_3).solution()))

# Other types of search functions can be used.
# print(depth_first_graph_search(q1_problem_1).solution())

    

