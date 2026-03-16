# particle_filter.py — particle filter classes used by the environments.
#
#   ParticleFilter_train     — pragmatic EFE only  (used for all_model_NoE)
#   ParticleFilter_train_AIF — full EFE: 0.75*pragmatic + epistemic  (all_model, all_model_mode)
#   Particle_Planner         — tree-search planner baseline

import numpy as np
from scipy.stats import entropy
from queue import PriorityQueue

num_particles = 500   # module-level constant (matches original)



class ParticleFilter_train:
    def __init__(self, prior_belief, x, y, obs_std):
        self.fixed_prior = prior_belief
        self.num_particles = 500
        self.x = x
        self.y = y
        self.obs_std = obs_std
        self.non_obs_std = obs_std
        self.prior_belief = prior_belief
        self.particles = np.zeros((self.num_particles, 3))
        p = np.random.multinomial(self.num_particles,
                                  pvals=[1 - self.prior_belief, self.prior_belief])
        self.particles[:, 0] = p[0] * [0] + p[1] * [1]
        self.particles[:p[0], 1] = np.random.beta(5, 3, size=p[0]) * -90
        self.particles[p[0]:, 1] = np.random.beta(5, 3, size=p[1]) * 90
        self.particles[:, 2] = [1 for x in range(self.num_particles)]
        self.weights = np.ones(num_particles) / num_particles

    def sample(self, truth_value):
        s = [i for i in self.particles if i[0] == truth_value]
        # return s[np.random.choice(len(s))]

    def predict(self):
        pred = self.estimate()
        from src.motion_models import next_position
        return next_position(self.x, self.y, pred[1], pred[2])

    def observation_likelihood(self, pedestrian_presence, observation):
        if pedestrian_presence == observation:
            return 1
        else:
            return 0

    def update(self, truth, observation):
        from src.motion_models import next_position_sim
        coord = [next_position_sim(self.x, self.y, c[1], c[2]) if c[0] == 0
                 else next_position_sim(self.x, self.y, c[1], c[2], cross=True)
                 for c in self.particles]
        distances = np.linalg.norm(coord - observation, axis=1)
        self.weights *= (1 / (self.obs_std * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * (distances / self.obs_std) ** 2)
        self.x = observation[0]
        self.y = observation[1]
        self.weights += 1.e-300
        self.weights /= np.sum(self.weights)

    def compute_efe(self, env, show=False):
        pragmatic_value = (np.mean(self.particles[:, 0]) * env.compute_reward(assumed_state='ped_cross')
                           + (1 - np.mean(self.particles[:, 0])) * env.compute_reward(assumed_state='other'))
        epistemic_value = entropy(
            np.array([np.mean(self.particles[:, 0]), 1 - np.mean(self.particles[:, 0])]),
            np.array([self.prior_belief, 1 - self.prior_belief]))
        if show:
            print(f"pragmatic value : {pragmatic_value} , epistemic:  {epistemic_value}"
                  f" : {np.mean(self.particles[:, 0])}")
        return pragmatic_value   # pragmatic only

    def resample(self):
        self.prior_belief = np.mean(self.particles[:, 0])
        indices = np.random.choice(range(self.num_particles), size=self.num_particles,
                                   p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def show(self):
        np.mean(self.particles[0, :])

    def reset(self):
        self.particles = np.zeros((num_particles, 3))
        p = np.random.multinomial(num_particles,
                                  pvals=[1 - self.prior_belief, self.prior_belief])
        self.particles[:, 0] = p[0] * [0] + p[1] * [1]
        self.particles[:p[0], 1] = np.random.beta(5, 3, size=p[0]) * -90
        self.particles[p[0]:, 1] = np.random.beta(5, 3, size=p[1]) * 90
        self.particles[:, 2] = [1 for x in range(500)]
        self.weights = np.ones(num_particles) / num_particles
        self.prior_belief = self.fixed_prior

    def estimate(self):
        return np.average(self.particles, axis=0, weights=self.weights)



class ParticleFilter_train_AIF:
    def __init__(self, prior_belief, x, y, obs_std):
        self.fixed_prior = prior_belief
        self.num_particles = 500
        self.x = x
        self.y = y
        self.ego_velocity = 70
        self.obs_std = obs_std
        self.non_obs_std = obs_std
        self.prior_belief = prior_belief
        self.particles = np.zeros((self.num_particles, 3))
        p = np.random.multinomial(self.num_particles,
                                  pvals=[1 - self.prior_belief, self.prior_belief])
        self.particles[:, 0] = p[0] * [0] + p[1] * [1]
        self.particles[:p[0], 1] = np.random.beta(5, 3, size=p[0]) * -90
        self.particles[p[0]:, 1] = np.random.beta(5, 3, size=p[1]) * 90
        self.particles[:, 2] = [1 for x in range(self.num_particles)]
        self.weights = np.ones(num_particles) / num_particles

    def sample(self, truth_value):
        s = [i for i in self.particles if i[0] == truth_value]
        # return s[np.random.choice(len(s))]

    def predict(self):
        pred = self.estimate()
        from src.motion_models import next_position
        return next_position(self.x, self.y, pred[1], pred[2])

    def observation_likelihood(self, pedestrian_presence, observation):
        if pedestrian_presence == observation:
            return 1
        else:
            return 0

    def update(self, truth, observation):
        from src.motion_models import next_position_sim
        coord = [next_position_sim(self.x, self.y, c[1], c[2]) if c[0] == 0
                 else next_position_sim(self.x, self.y, c[1], c[2], cross=True)
                 for c in self.particles]
        distances = np.linalg.norm(coord - observation, axis=1)
        self.weights *= (1 / (self.obs_std * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * (distances / self.obs_std) ** 2)
        self.x = observation[0]
        self.y = observation[1]
        self.weights += 1.e-300
        self.weights /= np.sum(self.weights)

    def compute_efe(self, env, show=False):
        pragmatic_value = (np.mean(self.particles[:, 0]) * env.compute_reward(assumed_state='ped_cross')
                           + (1 - np.mean(self.particles[:, 0])) * env.compute_reward(assumed_state='other'))
        epistemic_value = entropy(
            np.array([np.mean(self.particles[:, 0]), 1 - np.mean(self.particles[:, 0])]),
            np.array([self.prior_belief, 1 - self.prior_belief]))
        if show:
            print(f"pragmatic value : {pragmatic_value} , epistemic:  {epistemic_value}"
                  f" : {np.mean(self.particles[:, 0])}")
        return 0.75 * pragmatic_value + epistemic_value   # full EFE

    def resample(self):
        self.prior_belief = np.mean(self.particles[:, 0])
        indices = np.random.choice(range(self.num_particles), size=self.num_particles,
                                   p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def show(self):
        np.mean(self.particles[0, :])

    def reset(self):
        self.particles = np.zeros((num_particles, 3))
        p = np.random.multinomial(num_particles,
                                  pvals=[1 - self.prior_belief, self.prior_belief])
        self.particles[:, 0] = p[0] * [0] + p[1] * [1]
        self.particles[:p[0], 1] = np.random.beta(5, 3, size=p[0]) * -90
        self.particles[p[0]:, 1] = np.random.beta(5, 3, size=p[1]) * 90
        self.particles[:, 2] = [1 for x in range(500)]
        self.weights = np.ones(num_particles) / num_particles
        self.prior_belief = self.fixed_prior

    def estimate(self):
        return np.average(self.particles, axis=0, weights=self.weights)



class Particle_Planner:
    def __init__(self, prior_belief, x, y):
        num_p = 500
        self.num_particles = 500
        self.x = x
        self.y = y
        self.obs_std = 1.5
        self.non_obs_std = 1.5
        self.prior_belief = prior_belief
        self.fixed_prior = prior_belief
        self.particles = np.zeros((num_p, 3))
        p = np.random.multinomial(num_p, pvals=[1 - self.prior_belief, self.prior_belief])
        self.particles[:, 0] = p[0] * [0] + p[1] * [1]
        self.particles[:p[0], 1] = np.random.beta(3, 5, size=p[0]) * 90
        self.particles[p[0]:, 1] = np.random.beta(5, 3, size=p[1]) * -90
        self.particles[:, 2] = [2 for x in range(500)]
        self.weights = np.ones(num_p) / num_p

    def sample(self, truth_value):
        s = [i for i in self.particles if i[0] == truth_value]
        return s[np.random.choice(len(s))]

    def predict(self):
        pred = self.estimate()
        from src.motion_models import next_position
        return next_position(self.x, self.y, pred[1], pred[2])

    def observation_likelihood(self, pedestrian_presence, observation):
        if pedestrian_presence == observation:
            return 1
        else:
            return 0

    def update(self, truth, observation, observed=True):
        from src.motion_models import next_position
        coord = [next_position(self.x, self.y, c[1], c[2]) for c in self.particles]
        distances = np.linalg.norm(coord - observation, axis=1)
        self.weights *= (1 / (self.obs_std * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * (distances / self.obs_std) ** 2)
        self.x = observation[0]
        self.y = observation[1]
        self.weights += 1.e-300
        self.weights /= np.sum(self.weights)

    def compute_efe(self, env, show=False):
        pragmatic_value = (np.mean(self.particles[:, 0]) * env.compute_reward(assumed_state='ped_cross')
                           + (1 - np.mean(self.particles[:, 0])) * env.compute_reward(assumed_state='other'))
        epistemic_value = entropy(
            np.array([np.mean(self.particles[:, 0]), 1 - np.mean(self.particles[:, 0])]),
            np.array([self.prior_belief, 1 - self.prior_belief]))
        if show:
            print(f"pragmatic value : {pragmatic_value} , epistemic:  {epistemic_value}"
                  f" : {np.mean(self.particles[:, 0])}")
        return -pragmatic_value - epistemic_value

    def resample(self):
        indices = np.random.choice(range(self.num_particles), size=self.num_particles,
                                   p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        return np.average(self.particles, axis=0, weights=self.weights)

    def reset(self):
        p = np.random.multinomial(500, pvals=[1 - self.prior_belief, self.prior_belief])
        self.particles = np.zeros((500, 3))
        self.particles[:, 0] = p[0] * [0] + p[1] * [1]
        self.particles[:p[0], 1] = np.random.beta(5, 3, size=p[0]) * -90
        self.particles[p[0]:, 1] = np.random.beta(5, 3, size=p[1]) * 90
        self.particles[:, 2] = [1 for x in range(500)]
        self.weights = np.ones(500) / 500
        self.prior_belief = self.fixed_prior
