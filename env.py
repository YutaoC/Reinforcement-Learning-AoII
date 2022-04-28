import numpy as np


class CommSystem(object):
    """
    Define the environment (A communication system in a slotted-time system).
    The system consists of n users. At each time slot, a base station chooses one user to update.
    The update is transmitted through a common channel with random delays.
    The sources follow normal distributions with possibly different means and standard deviations.
    The receivers use the last received updates as estimates and will send ACK/NACK packets.
    The time penalty function is linear and the information penalty function is threshold with threshold being 0.5.
    """
    def __init__(self, n, max_epoch):
        self.N = n  # number of users in the system
        self.action_space = [i + 1 for i in range(n)]
        self.transmission_prob = list(np.ones(15)*0.4)  # random transmission delay (Geometric)
        self.mu = np.zeros(self.N)  # signal mean
        self.sigma = np.ones(self.N)  # signal standard deviation
        self.threshold = 0.5  # threshold in information penalty function

        self.signal = np.zeros(self.N)  # source signals
        self.estimate = np.zeros(self.N)  # receiver's estimates
        self.penalty = np.zeros(self.N)  # values of information penalty function
        self.transmission_time = 0  # remaining transmission time (0 when the channel is idle)
        self.transmitting_update = -1  # the transmitting update (-1 when the channel is idle)

        self.isDone = False  # end of episode flag
        self.epoch_cntr = 0  # epoch counter
        self.max_epoch = max_epoch  # number of epochs in an episode

    def step(self, action):
        if self.isDone:
            resulting_state = self.reset()
            reward = 0

            return resulting_state, reward, False, None

        # Shared communication channel
        is_update_arrived = np.random.choice(2, 1, p=[self.transmission_prob[self.transmission_time],
                                                      1 - self.transmission_prob[self.transmission_time]])
        # when channel is idle
        if action <= self.N and self.transmission_time == 0:
            if is_update_arrived == 1:
                self.estimate[action-1] = self.signal[action-1]
                self.transmission_time = 0
                self.transmitting_update = -1
            else:
                self.transmission_time += 1
                self.transmitting_update = self.signal[action-1]
        # when the channel is busy
        elif action <= self.N and self.transmission_time > 0:
            if is_update_arrived == 1:
                self.estimate[action-1] = self.transmitting_update
                self.transmission_time = 0
                self.transmitting_update = -1
            else:
                self.transmission_time += 1

        # signal evolution
        self.signal = np.random.normal(self.mu, self.sigma)

        # update the values of information penalty function
        difference = np.absolute(self.estimate - self.signal)
        plus_penalty = np.where(difference > self.threshold, difference, np.zeros(self.N))
        self.penalty += plus_penalty
        self.penalty[np.where(plus_penalty == 0)] = 0

        # the returns
        self.isDone = self.epoch_cntr == self.max_epoch
        resulting_state = np.concatenate((self.signal, self.estimate, self.penalty, self.transmission_time,
                                          self.transmitting_update), axis=None)
        reward = self.penalty

        self.epoch_cntr += 1

        return resulting_state, reward, self.isDone, None

    def reset(self):
        self.signal = np.zeros(self.N)
        self.estimate = np.zeros(self.N)
        self.penalty = np.zeros(self.N)
        self.transmission_time = 0
        self.transmitting_update = -1

        self.isDone = False
        self.epoch_cntr = 0

        resulting_state = np.append(np.zeros(3 * self.N), [0, -1])

        return resulting_state

