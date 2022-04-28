import gym
import numpy as np

from dqn import Agent
from utils import plot_learning_curve

"""
Train the DQN agent.
"""
if __name__ == '__main__':
    env = gym.make('LunarLander-v2')  # create an instance of the environment
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01,
                  input_dims=[8], lr=0.001)  # create an instance of the agent
    scores, eps_history = [], []  # store the results
    n_games = 500  # number of games

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)  # choose the action
            observation_, reward, done, info = env.step(action)  # operate the action
            score += reward  # get the reward
            agent.store_transition(observation, action, reward,
                                   observation_, done)  # store the transition
            agent.learn()  # let the agent learn
            observation = observation_  # one step further
        scores.append(score)  # store the scores
        eps_history.append(agent.epsilon)  # store the epsilon

        avg_score = np.mean(scores[-100:])  # running average score

        # debug info
        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)

    # visualization
    x = [i + 1 for i in range(n_games)]
    filename_score = 'lunar_lander_score.png'
    plot_learning_curve(x, scores, filename_score)
    filename_epsilon = 'lunar_lander_epsilon.png'
    plot_learning_curve(x, scores, filename_epsilon)
