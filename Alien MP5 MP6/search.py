# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
import heapq
from state import MazeState


# Search should return the path and the number of states explored.
# The path should be a list of MazeState objects that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (astar)
# You may need to slight change your previous search functions in MP2 since this is 3-d maze


def search(maze, searchMethod):
    return {
        "astar": astar,
    }.get(searchMethod, [])(maze)


# TODO: VI
# Define the A* search algorithm
import heapq

import heapq

def astar(maze):
    '''
    Implementation of best first search algorithm

    Input:
        starting_state: an AbstractState object

    Return:
        A path consisting of a list of AbstractState states
        The first state should be starting_state
        The last state should have state.is_goal() == True
    '''
    # we will use this visited_states dictionary to serve multiple purposes
    # - visited_states[state] = (parent_state, distance_of_state_from_start)
    #   - keep track of which states have been visited by the search algorithm
    #   - keep track of the parent of each state, so we can call backtrack(visited_states, goal_state) and obtain the path
    #   - keep track of the distance of each state from start node
    #       - if we find a shorter path to the same state we can update with the new state 
    # NOTE: we can hash states because the __hash__/__eq__ method of AbstractState is implemented
    start = maze.get_start()
    visited_states = {start: (None, 0)}

    # The frontier is a priority queue
    # You can pop from the queue using "heapq.heappop(frontier)"
    # You can push onto the queue using "heapq.heappush(frontier, state)"
    # NOTE: states are ordered because the __lt__ method of AbstractState is implemented
    frontier = []
    heapq.heappush(frontier, start)

    # TODO(III): implement the rest of the best first search algorithm
    # HINTS:
    #   - add new states to the frontier by calling state.get_neighbors()
    #   - check whether you've finished the search by calling state.is_goal()
    #       - then call backtrack(visited_states, state)...
    #   - you can reuse the search code from mp3...
    # Your code here ---------------
    while frontier:
        current_state = heapq.heappop(frontier)
        # Check if the current state is the goal state
        if current_state.is_goal():
            # Backtrack to construct the path
            return backtrack(visited_states, current_state)

        # Explore neighboring states
        for neighbor_state in current_state.get_neighbors():
            # Calculate the total distance from the start to the neighbor state
            if neighbor_state not in visited_states.keys():
                    heapq.heappush(frontier,neighbor_state)
                    visited_states[neighbor_state] = (current_state,neighbor_state.dist_from_start)
            else:
                if visited_states[neighbor_state][1]>neighbor_state.dist_from_start:
                    visited_states[neighbor_state] = (current_state,neighbor_state.dist_from_start)
                    heapq.heappush(frontier,neighbor_state)
    # ------------------------------
    
    # if you do not find the goal return an empty list
    return None




# Go backwards through the pointers in visited_states until you reach the starting state
# NOTE: the parent of the starting state is None
# TODO: VI
def backtrack(visited_states, goal_state):
    path = []
    # Your code here ---------------
    current_state = goal_state
    # Continue tracing back from the goal state to the starting state
    while current_state:
        path.insert(0, current_state)  # Insert at the beginning to maintain the correct order
        current_state = visited_states[current_state][0]  # Get the parent state
    # ------------------------------
    return path
