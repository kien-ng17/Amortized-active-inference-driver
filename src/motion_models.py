# motion_models.py — pedestrian motion model utilities.
# NOTE: 'foward_sim' keeps the original typo.

import numpy as np


# position propagation

def next_position_sim(x, y, heading, speed, dt=0.1, cross=False):
    if x == 40 and y <= 0 and cross:
        return 40, y - speed
    heading = (heading + 180) % 360
    heading_rad = np.deg2rad(heading)
    delta_x = speed * np.cos(heading_rad)
    delta_y = speed * np.sin(heading_rad)
    new_x = x + delta_x
    new_y = y + delta_y
    if new_x <= 40 and cross:
        return 40, y - speed
    if new_y < 0:
        new_y = y + speed * np.sin(np.deg2rad(heading + 30))
    return new_x, new_y


def next_position(x, y, heading, speed, dt=0.1):
    heading_rad = np.deg2rad(heading)
    delta_x = speed * np.cos(heading_rad)
    delta_y = speed * np.sin(heading_rad)
    new_x = x + delta_x
    new_y = y + delta_y
    return new_x, new_y


# pedestrian particle

class Particle_ped:
    def __init__(self, prior_belief, x, y, mean_cross=10, mean_not_cross=-10):
        num_particles = 3000
        self.num_particles = 3000
        self.x = x
        self.y = y
        self.obs_std = 1
        self.non_obs_std = 1
        self.prior_belief = prior_belief
        self.particles = np.zeros((num_particles, 3))
        p = np.random.multinomial(num_particles, pvals=[1 - self.prior_belief, self.prior_belief])
        self.particles[:, 0] = p[0] * [0] + p[1] * [1]
        self.particles[:p[0], 1] = np.random.normal(mean_not_cross, 20, size=p[0])
        self.particles[p[0]:, 1] = np.random.normal(mean_cross, 20, size=p[1])
        self.particles[:, 2] = [3 for x in range(3000)]
        self.weights = np.ones(num_particles) / num_particles

    def sample(self, truth_value):
        s = [i for i in self.particles if i[0] == truth_value]
        if len(s) > 0:
            return s[np.random.choice(len(s))]
        else:
            return self.particles[np.random.choice(len(self.particles))]

    def predict(self):
        pred = self.estimate()
        return next_position(self.x, self.y, pred[1], pred[2])

    def observation_likelihood(self, pedestrian_presence, observation):
        if pedestrian_presence == observation:
            return 1
        else:
            return 0

    def update(self, truth, observation, observed=True):
        coord = [next_position(self.x, self.y, c[1], c[2]) for c in self.particles]
        distances = np.linalg.norm(coord - observation, axis=1)
        if observed:
            self.weights *= (1 / (self.obs_std * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * (distances / self.obs_std) ** 2)
        else:
            penalty = (1 / (self.non_obs_std * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * (distances / self.non_obs_std) ** 2)
            self.weights *= (penalty)
        self.x = observation[0]
        self.y = observation[1]
        self.weights += 1.e-300
        self.weights /= np.sum(self.weights)

    def resample(self):
        indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        return np.average(self.particles, axis=0, weights=self.weights)


# forward simulation

def foward_sim(belief, ground_truth, step, x, y, alert=False):  # typo kept
    if not alert:
        pf = Particle_ped(belief, x, y)
        speed = 1
    else:
        pf = Particle_ped(belief, x, y, mean_cross=45, mean_not_cross=-45)
        speed = 2.5

    pos = np.zeros((step + 1, 2))
    pos[0, :] = x, y
    for i in range(step):
        h = pf.sample(ground_truth)
        location = next_position_sim(x, y, h[1], speed, cross=ground_truth)
        x, y = location
        pos[i + 1, :] = location
    return pos
