import gym
import numpy as np

from ppo import Agent
from utils import plot_learning_curve

"""
Train the PPO agent.
"""
if __name__ == '__main__':
    env = gym.make('CartPole-v0')  # create an instance of the environment
    N = 20  # training interval
    batch_size = 5  # batch size
    n_epochs = 4  # epochs in learn
    n_games = 300  # number of games
    alpha = 0.0003  # learning rate

    # scores
    best_score = env.reward_range[0]
    score_history = []

    # intermediate values
    learn_iters = 0  # count the number of times the agent has learned
    avg_score = 0  # average score
    n_steps = 0  # count the steps the agent took in a game

    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape)  # create an instance of the agent

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])  # running average

        # store the best score
        if avg_score > best_score:
            best_score = avg_score
            # agent.save_models()

        # debug info
        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)

    # visualization
    # x = [i+1 for i in range(len(score_history))]
    # figure_file = 'cartpole.png'
    # plot_learning_curve(x, score_history, figure_file)
