from search import *

class Flight:

  def __init__(self, start_city, start_time, end_city, end_time):
    self.start_city = start_city
    self.start_time = start_time
    self.end_city = end_city
    self.end_time = end_time

  def __str__(self):
    return str((self.start_city, self.start_time))+ '->' + str((self.end_city, self.end_time))

  __repr__ = __str__

  # Part 2: "matches" method that checks 
  # 1. if the flight's start city == the pair's city
  # 2. and the flight's start_time is later than the pair's time
  def matches(self, pair):
    return self.start_city == pair[0] and self.start_time >= pair[1]
  
flightDB = [Flight('Rome', 1, 'Paris', 4), Flight('Rome', 3, 'Madrid', 5), Flight('Rome', 5, 'Istanbul', 10), Flight('Paris', 2, 'London', 4), Flight('Paris', 5, 'Oslo', 7), Flight('Paris', 5, 'Istanbul', 9), Flight('Madrid', 7, 'Rabat', 10), Flight('Madrid', 8, 'London', 10), Flight('Istanbul', 10, 'Constantinople', 10)]

#Part 1: State - A good choice of state in this problem is the current city and time.

class FlightProblem(Problem):

  # actions are all possible flights that match the current state using the "matches" method in Flight Class
  def actions(self, state):
    actions = []
    for i in flightDB:
      # print(i.matches((state[0], state[1])))
      if i.matches((state[0], state[1])):
        # print(i)
        actions.append(i)
    return actions

  # After taking action, the new state should be a tuple consisting the end_city and end_time
  def result(self, state, action):
    return (action.end_city, action.end_time)

  # We need to overwrite the default goal test method and check that
  # 1. the current city = goal city
  # 2. the current time <= goal time a.k.a deadline
  def goal_test(self, state):
    return state[0] == self.goal[0] and state[1] <= self.goal[1]

  def value(self, state):
    return None

# Part 3: find_itinerary
def find_itinerary(start_city, start_time, end_city, deadline):
  flight_problem = FlightProblem((start_city, start_time),(end_city, deadline))
  try:
    # We having chosen bfs in this case.
    # Other search algorithms can be used too.
    return breadth_first_tree_search(flight_problem).solution()
  except:
    return None
  
# print(find_itinerary('Rome', 1, 'Paris', 7))

# Part 4: Going Further

# Will this strategy find the path that arrives soonest, given that we start at time
# 1?

# Yes, it will find the path that arrives soonest. 
# This is similar to a modified version of iterative deepening search. 
# We will get an optimal solution

# Imagine that if we use this strategy to solve a problem whose shortest path
# (shortest time) is length 100, it takes an amount x of calls on find itinerary
# to solve. Roughly how many calls to find itinerary will it take to solve a
# problem whose shortest path is length 200?

# It should take 2x number of calls.

def find_shortest_itinerary(start_city, end_city):
  idx = 1
  times_called = 0
  path = find_itinerary(start_city, 1, end_city, idx)
  times_called += 1
  while path == None:
    idx += 1
    path = find_itinerary(start_city, 1, end_city, idx)
    times_called += 1
  print("No. of times called:" + str(times_called))
  return path

print("Find Shortest Itinerary: 'Paris'->'Constantinople' \n" + str (find_shortest_itinerary('Paris', 'Constantinople')))

# Jumps in increments of 3, upon finding, decrease deadline until no path can be found
def find_shortest_itinerary_improved(start_city, end_city):
  idx = 1
  times_called = 0
  path = find_itinerary(start_city, 1, end_city, idx)
  times_called += 1
  while path == None:
    idx += 3
    path = find_itinerary(start_city, 1, end_city, idx)
    times_called += 1
  while path != None:
    idx -= 1
    final_path = path
    path = find_itinerary(start_city, 1, end_city, idx)
    times_called += 1
  print("No. of times called:" + str(times_called))
  return final_path

print("Find Shortest Itinerary IMPROVED: 'Paris'->'Constantinople' \n" + str (find_shortest_itinerary_improved('Paris', 'Constantinople')))

# find_shortest_itinerary_improved has reduced the number of unnecessary calls made to the function find_itinerary!
