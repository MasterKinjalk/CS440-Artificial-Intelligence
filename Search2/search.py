import heapq
# You do not need any other imports

def best_first_search(starting_state):
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
    visited_states = {starting_state: (None, 0)}

    # The frontier is a priority queue
    # You can pop from the queue using "heapq.heappop(frontier)"
    # You can push onto the queue using "heapq.heappush(frontier, state)"
    # NOTE: states are ordered because the __lt__ method of AbstractState is implemented
    frontier = []
    heapq.heappush(frontier, starting_state)
    
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
            if neighbor_state != current_state:
                if not visited_states.get(neighbor_state,None):
                    heapq.heappush(frontier,neighbor_state)
                    visited_states[neighbor_state] = (current_state,neighbor_state.dist_from_start)
                else:
                    if visited_states[neighbor_state][1]>neighbor_state.dist_from_start:
                        visited_states[neighbor_state] = (current_state,neighbor_state.dist_from_start)
    # ------------------------------
    
    # if you do not find the goal return an empty list
    return []

# TODO(III): implement backtrack method, to be called by best_first_search upon reaching goal_state
# Go backwards through the pointers in visited_states until you reach the starting state
# You can reuse the backtracking code from MP3
# NOTE: the parent of the starting state is None
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