import utils

class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma
        self.display_width = display_width
        self.display_height = display_height
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def update_n(self, state, action):
        # TODO - MP11: Update the N-table. 
        self.N[state][action] += 1

    def update_q(self, s, a, r, s_prime):
    # TODO - MP11: Update the Q-table.
        max_value = max(self.Q[s_prime])  
        alpha = self.C / (self.C + self.N[s][a])  # This is the correct formula for learning rate
        self.Q[s][a] += alpha * (r + self.gamma * max_value - self.Q[s][a])  # This is the correct formula for Q update
 

    def choose_action(self, s):
        # Convert s to a hashable type
        s = tuple(s)

        max_value = -float('inf')  # Initialize the maximum value to negative infinity
        best_action = None  # Initialize the best action to None

        if self._train == True:
            for a in range(3, -1, -1):  # Loop through all possible actions in reverse order (RIGHT > LEFT > DOWN > UP)
                if self.N[s][a] < self.Ne:  # If the action has not been explored enough
                    return a  # Return that action
                else:  # Otherwise
                    value = self.Q[s][a]  # Get the Q-value of the action
                    if value > max_value:  # If the value is greater than the current maximum
                        max_value = value  # Update the maximum value
                        best_action = a  # Update the best action

        if self._train == False:
            for a in range(3, -1, -1):
                value = self.Q[s][a]  # Get the Q-value of the action
                if value > max_value:  # If the value is greater than the current maximum
                    max_value = value  # Update the maximum value
                    best_action = a  # Update the best action

        return best_action  # Return the best action



    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)   
        # Update Q and N based on the previous state, action, and points
        if self.s is not None and self.a is not None:
            # Calculate the incremental reward
            if points - self.points == 1:
                r = 1
            elif dead:
                r = -1
                self.update_n(self.s, self.a)  # Update N-table
                self.update_q(self.s, self.a, r, s_prime)  # Update Q-table
                self.reset()
                return 0
            else:
                r = -0.1
                
            self.update_n(self.s, self.a)  # Update N-table
            self.update_q(self.s, self.a, r, s_prime)  # Update Q-table

        # If the snake is dead, reset bookkeeping variables and return a default action
        if self._train == False and dead:
            self.reset()
            return utils.RIGHT

        # Choose an action using the exploration policy
        action = self.choose_action(s_prime)

        # Store the current state and action for the next time step
        self.s = s_prime
        self.a = action
        self.points = points  # Update the reward for the next time step

        return action




    def generate_state(self, environment):
        snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y = environment

        # Direction of food relative to the snake head
        food_dir_x = 0
        if food_x < snake_head_x:
            food_dir_x = 1
        elif food_x > snake_head_x:
            food_dir_x = 2

        food_dir_y = 0
        if food_y < snake_head_y:
            food_dir_y = 1
        elif food_y > snake_head_y:
            food_dir_y = 2
        rock_pos = [(rock_x,rock_y),(rock_x+1,rock_y)]

        if self.display_width == 18:
            # Adjoining wall or rock next to the snake head
            adjoining_wall_x = 0
            if snake_head_x == 1  or  (snake_head_x-1, snake_head_y) in rock_pos:
                adjoining_wall_x = 1
            elif snake_head_x == 16 or  (snake_head_x+1, snake_head_y) in rock_pos :
                adjoining_wall_x = 2


            # Adjoining wall or rock next to the snake head
            adjoining_wall_y = 0
            if snake_head_y == 1 or (snake_head_x,snake_head_y - 1) in rock_pos:
                adjoining_wall_y = 1
            elif snake_head_y == 8 or  (snake_head_x,snake_head_y + 1) in rock_pos:
                adjoining_wall_y = 2

        if self.display_width == 10:
            # Adjoining wall or rock next to the snake head
            adjoining_wall_x = 0
            if snake_head_x == 1  or  (snake_head_x-1, snake_head_y) in rock_pos:
                adjoining_wall_x = 1
            elif snake_head_x == 8 or  (snake_head_x+1, snake_head_y) in rock_pos :
                adjoining_wall_x = 2

            # Adjoining wall or rock next to the snake head
            adjoining_wall_y = 0
            if snake_head_y == 1 or (snake_head_x,snake_head_y - 1) in rock_pos:
                adjoining_wall_y = 1
            elif snake_head_y == 16 or  (snake_head_x,snake_head_y + 1) in rock_pos:
                adjoining_wall_y = 2
            


        # Adjoining body parts around the snake head
        adjoining_body_top = 1 if (snake_head_x, snake_head_y - 1) in snake_body else 0
        adjoining_body_bottom = 1 if (snake_head_x, snake_head_y + 1) in snake_body else 0
        adjoining_body_left = 1 if (snake_head_x - 1, snake_head_y) in snake_body else 0
        adjoining_body_right = 1 if (snake_head_x + 1, snake_head_y) in snake_body else 0

        state = (
            food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y,
            adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right
        )

        return state