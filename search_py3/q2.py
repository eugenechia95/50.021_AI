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

  def matches(self, pair):
    return self.start_city == pair[0] and self.start_time >= pair[1]
  
flightDB = [Flight('Rome', 1, 'Paris', 4), Flight('Rome', 3, 'Madrid', 5), Flight('Rome', 5, 'Istanbul', 10), Flight('Paris', 2, 'London', 4), Flight('Paris', 5, 'Oslo', 7), Flight('Paris', 5, 'Istanbul', 9), Flight('Madrid', 7, 'Rabat', 10), Flight('Madrid', 8, 'London', 10), Flight('Istanbul', 10, 'Constantinople', 10)]

class FlightProblem(Problem):

  def actions(self, state):
    actions = []
    for i in flightDB:
      # print(i.matches((state[0], state[1])))
      if i.matches((state[0], state[1])):
        # print(i)
        actions.append(i)
    return actions

  
  def result(self, state, action):
    return (action.end_city, action.end_time)

  def goal_test(self, state):
    return state[0] == self.goal[0] and state[1] <= self.goal[1]

  def value(self, state):
    return None



def find_itinerary(start_city, start_time, end_city, deadline):
  flight_problem = FlightProblem((start_city, start_time),(end_city, deadline))
  return breadth_first_tree_search(flight_problem).solution()

print(find_itinerary('Rome', 1, 'Paris', 8))
