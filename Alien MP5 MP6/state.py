import copy
import math
from itertools import count

# NOTE: using this global index means that if we solve multiple
#       searches consecutively the index doesn't reset to 0... this is fine
global_index = count()


# TODO VI
# Euclidean distance between two state tuples, of the form (x,y, shape)
def euclidean_distance(a, b):
    
    x1, y1 = a[0],a[1]
    x2, y2 = b[0],b[1]
    
    # Calculate the Euclidean distance
    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    return distance


from abc import ABC, abstractmethod


class AbstractState(ABC):
    def __init__(self, state, goal, dist_from_start=0., use_heuristic=True):
        self.state = state
        self.goal = goal
        # we tiebreak based on the order that the state was created/found
        self.tiebreak_idx = next(global_index)
        # dist_from_start is classically called "g" when describing A*, i.e., f = g + h
        self.dist_from_start = dist_from_start
        self.use_heuristic = use_heuristic
        if use_heuristic:
            self.h = self.compute_heuristic()
        else:
            self.h = 0

    # To search a space we will iteratively call self.get_neighbors()
    # Return a list of State objects
    @abstractmethod
    def get_neighbors(self):
        pass

    # Return True if the state is the goal
    @abstractmethod
    def is_goal(self):
        pass

    # A* requires we compute a heuristic from eahc state
    # compute_heuristic should depend on self.state and self.goal
    # Return a float
    @abstractmethod
    def compute_heuristic(self):
        pass

    # The "less than" method ensures that states are comparable
    #   meaning we can place them in a priority queue
    # You should compare states based on f = g + h = self.dist_from_start + self.h
    # Return True if self is less than other
    @abstractmethod
    def __lt__(self, other):
        # NOTE: if the two states (self and other) have the same f value, tiebreak using tiebreak_idx as below
        if self.tiebreak_idx < other.tiebreak_idx:
            return True

    # __hash__ method allow us to keep track of which
    #   states have been visited before in a dictionary
    # You should hash states based on self.state (and sometimes self.goal, if it can change)
    # Return a float
    @abstractmethod
    def __hash__(self):
        pass

    # __eq__ gets called during hashing collisions, without it Python checks object equality
    @abstractmethod
    def __eq__(self, other):
        pass


# State: a length 3 list indicating the current location in the grid and the shape
# Goal: a tuple of locations in the grid that have not yet been reached
#   NOTE: it is more efficient to store this as a binary string...
# maze: a maze object (deals with checking collision with walls...)
class MazeState(AbstractState):
    def __init__(self, state, goal, dist_from_start, maze, use_heuristic=True):
        # NOTE: it is technically more efficient to store both the mst_cache and the maze_neighbors functions globally,
        #       or in the search function, but this is ultimately not very inefficient memory-wise
        self.maze = maze
        self.maze_neighbors = maze.get_neighbors
        super().__init__(state, goal, dist_from_start, use_heuristic)



    def get_neighbors(self):
        """Returns a list of neighboring states for the current state.

        The cost of moving to a neighboring state depends on whether the alien's shape
        changes:
            * If the shape changes, the cost is 10.
            * Otherwise, the cost is the Euclidean distance between the current state and
            the neighboring state.

        Args:
            self: The current state.

        Returns:
            A list of neighboring states.
        """

        neighbors = []

        for neighbor_state in self.maze_neighbors(*self.state):
            if neighbor_state[2] == self.state[2]:
                cost = euclidean_distance(self.state, neighbor_state) + self.dist_from_start
            else:
                cost = abs(neighbor_state[2] - self.state[2])*10 + self.dist_from_start

            new_state = MazeState(neighbor_state, self.goal, cost, self.maze,
                                self.use_heuristic)

            if new_state not in neighbors:
                neighbors.append(new_state)

        return neighbors



    # TODO VI
    def is_goal(self):
        return self.state[:2] in self.goal

    # We hash BOTH the state and the remaining goals
    #   This is because (x, y, h, (goal A, goal B)) is different from (x, y, h, (goal A))
    #   In the latter we've already visited goal B, changing the nature of the remaining search
    # NOTE: the order of the goals in self.goal matters, needs to remain consistent
    # TODO VI
    def __hash__(self):
        state_and_goals = (self.state[0], self.state[1], self.state[2], tuple(self.goal))
        return hash(state_and_goals)



    # TODO VI
    def __eq__(self, other):
        return (self.state[0] == other.state[0]) and (self.state[1] == other.state[1]) and (self.state[2] == other.state[2])


    # Our heuristic is: distance(self.state, nearest_goal)
    # We euclidean distance
    # TODO VI
    # TODO VI
    def compute_heuristic(self):
        dist = []
        for i in self.goal:
            dist.append(euclidean_distance(self.state, i))
        return min(dist)



    # This method allows the heap to sort States according to f = g + h value
    # TODO VI
    def __lt__(self, other):
        # Compare states based on f = g + h
        f_self = self.dist_from_start + self.h
        f_other = other.dist_from_start + other.h
        if f_self == f_other:
            # If f values are equal, use tiebreaker (based on tiebreak_idx)
            return self.tiebreak_idx < other.tiebreak_idx
        return f_self < f_other


    # str and repr just make output more readable when your print out states
    def __str__(self):
        return str(self.state) + ", goals=" + str(self.goal)

    def __repr__(self):
        return str(self.state) + ", goals=" + str(self.goal)
