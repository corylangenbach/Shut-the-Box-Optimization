import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.special import softmax
import sys


# this class sets up the SHUT THE BOX environment
class STB:

    def __init__(self,  num_tiles):
        self.num_tiles = num_tiles      # number of tiles on game board: [1..n]
        self.board = None   # game board (as a binary number)
        self.dice = None  # roll of die or dice
        self.act_space = 2 ** self.num_tiles  # number of possible states of game: each tile can be flipped up or down
        self.obs_space = self.act_space * (2 * 6)  # all possible states * possible dice rolls at each state

    # returns a random dice roll for the current state
    def roll(self):
        if self.score(self.board) <= 6:
            return np.random.randint(1, 7)  # a single 6-sided die, rolled when current score is less than 6
        else:
            return sum(np.random.randint(1, 7, 2))  # the sum of 2 6-sided dice

    # resets the board to initial state of all ones
    def reset(self):
        self.board = (2 ** self.num_tiles) - 1
        self.dice = self.roll()
        return self.state()

    # returns the action_space value, defined above
    def action_space(self):
        return self.act_space,

    # returns the observation_space value, defined above
    def observation_space(self):
        return self.obs_space,

    # the current state the game is in is found by multiplying the board state by the current roll
    def state(self):
        return self.board * self.dice

    # gets a random sample within the entire action space
    def sample(self):
        return np.random.randint(0, self.act_space)

    # calculates the score of the board by summing up the tiles that are currently flipped up
    # adds one to each tile since first tile has value '1' but it as position 0
    def score(self, board):
        return sum([(x + 1) for x in range(self.num_tiles) if board & 2 ** x])

    # preforms a move if it's legal, and returns the new state and score and if the game has ended
    def step(self, action):
        # if action is legal (tiles are up) and action matches roll of dice
        if action & self.board == action and self.score(action) == self.dice:
            # update game board to reflect current action
            self.board -= action
            self.dice = self.roll()
            return self.state(), 0, self.board == 0
        else:
            # note: score is always negative (unless the game is over), so it increases as tiles are flipped down
            return self.state(), -self.score(self.board), True

    # converts the board into a readable binary for the user so they can interpret the moves the program takes
    # this function is used when the user renders the program
    def __str__(self):
        return str([(x + 1) for x in range(self.num_tiles) if self.board & 2 ** x])

    # prints the current board and dice roll after each step the program takes
    def render(self):
        print("current board: ", self)
        print("dice roll: ", self.dice)


# saves the current q-table and parameters to the given files
def save(p, q, q_file_name, params_file_name):
    f = None
    try:
        print('saving...')
        np.save(q_file_name, q)
        f = open(params_file_name, 'wb')
        pickle.dump(p, f)
        f.close()
    except Exception as e:
        print('could not save', e)
    finally:
        if f is not None:
            f.close()


# get the current q table from the previously created file,
# or initialize a new q table with zeros if file not yet created
def reload(observation_size, action_size, q_file_name, params_file_name):
    f = None
    try:
        q = np.load(q_file_name)
        f = open(params_file_name, 'rb')
        p = pickle.load(f)
        f.close()
        print('reloaded with params',p)
        return p, q
    except Exception as e:
        # base parameters
        params = {}
        params['epsilon'] = 0.80
        params['min_epsilon'] = 0.01
        params['decay_rate'] = 0.999
        params['learning_rate'] = 0.01
        params['gamma'] = 0.975
        params['episodes'] = 0
        print(e)
        print('could not reload; starting with base parameters')
        return params, np.zeros([observation_size, action_size])
    finally:
        if f is not None:
            f.close()


# **** REINFORCEMENT LEARNING CODE ****


# epsilon-greedy strategy
def epsilon_greedy_strategy(params, sampler, q_vals, actions):
    return sampler if np.random.random() < params['epsilon'] else np.argmax(q_vals)


# softmax strategy
def softmax_strategy(params, sampler, q_vals, actions):
    return np.random.choice(actions, p=softmax(q_vals))


# cumulatively train the program each time it's run
def train(strategy, q_file_name, params_file_name):
    try:
        x = 0
        env = STB(num_tiles=10)
        observation_size = env.observation_space()[0]
        action_size = env.action_space()[0]
        actions = [n for n in range(action_size)]
        params, q = reload(observation_size, action_size, q_file_name, params_file_name)
        print('training is based on', params['episodes'],'episodes')
        print('(Only valuable to epsilon greedy strategy): epsilon is', params['epsilon'])
        # set up score and buf arrays to keep track of results
        score = []
        buf = np.zeros(10000)

        while True:
            state = env.reset()
            done = False

            # every 10,000 episodes, decay epsilon based on the given decay rate
            if params['episodes'] % 10000 == 0:
                print('episode:', params['episodes'])
                params['epsilon'] *= params['decay_rate']
                params['epsilon'] = max(params['epsilon'], params['min_epsilon'])

            while not done:
                # print current board and dice roll
                # env.render()

                # call to the strategy function given by user (epsilon_greedy or softmax) to get action
                action = strategy(params, env.sample(), q[state], actions)

                # perform the action
                obs, reward, done = env.step(action)

                # use q formula to get next value, based on both current and future value
                current_val = q[state][action]
                next_best = np.max(q[obs])
                # q formula
                new_val = (1 - params['learning_rate']) * current_val + params['learning_rate'] *\
                          (reward + params['gamma'] * next_best)
                # set new value into table
                q[state][action] = new_val

                state = obs

                if done:
                    buf[x % 10000] = reward
                    # add accumulated score into array every 10,000 episodes
                    if x > 0 and x % 10000 == 0:
                        score.append(sum(buf))

                    # save every 1,000,000 episodes
                    if x % 100000 == 0 and x > 0:
                        save(params, q, q_file_name, params_file_name)

                    # print final game board and dice roll
                    # env.render()
                    break
            params['episodes'] += 1
            x += 1

    # when program is stopped, save progress and display graph of scores
    except KeyboardInterrupt:
        print('stopping')
        save(params, q, q_file_name, params_file_name)
        # plot score
        plt.plot(score)
        plt.title('Program Learning Shown Via Accumulated Reward')
        plt.ylabel('Cumulative Score after Every 10,000 Episodes')
        plt.xlabel('Groups of 10,000 Episodes')
        plt.show()


# parse through the command line argument, which should be "epsilon_greedy" or "softmax" depending on which
# strategy the user would like to use
if len(sys.argv) == 1:
    print("No command line argument was given.")
    sys.exit()
if sys.argv[1] == 'epsilon_greedy':
    strategy = epsilon_greedy_strategy
    q_file = 'q_epsilon_greedy.npy'
    param_file = 'params_epsilon_greedy.pickle'
elif sys.argv[1] == 'softmax':
    strategy = softmax_strategy
    q_file = 'q_softmax.npy'
    param_file = 'params_softmax.pickle'
else:
    print("Command line arguments are invalid.")
    sys.exit()

# call train function with correct parameters given command line argument
train(strategy, q_file, param_file)
