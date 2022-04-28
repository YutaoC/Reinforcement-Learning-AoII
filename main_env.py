import numpy as np

from env import CommSystem
import matplotlib.pyplot as plt

"""
Test the environment
"""
if __name__ == '__main__':
    N = 5  # number of users
    num_episode = 100  # number of episodes
    num_epochs = 5000  # number of epochs in an episode
    avg_rewards = np.zeros((N, num_episode))  # averaged reward for each user in each episode

    env = CommSystem(N, num_epochs)  # create an instance of the environment

    for i in range(num_episode):
        print('Episode #', i+1)
        done = False
        eps_rewards = 0
        observation = env.reset()
        while not done:
            action = np.random.randint(N)  # the action is chosen randomly (uniformly)
            observation_, reward, done, info = env.step(action)
            eps_rewards += reward
        avg_rewards[:, i] = eps_rewards / num_epochs

    # visualization
    plt.plot(avg_rewards[1, :])
    plt.show()
