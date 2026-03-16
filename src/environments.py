# environments.py — Gymnasium environments.
#
#   Pedestrian_Behaviour    — pedestrian counter-agent env (trains ped_model)
#   PedestrianCross_train   — ego-vehicle env, full AIF EFE  (trains all_model)
#   PedestrianCross_train_2 — ego-vehicle env, mode-of-belief obs (trains all_model_mode)
#
# Call set_ped_model() with the loaded QRDQN before stepping any crossing env.

import numpy as np
import cv2
import random
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

from src.motion_models import foward_sim
from src.particle_filter import ParticleFilter_train, ParticleFilter_train_AIF


_ped_model = None

def set_ped_model(model):
    global _ped_model
    _ped_model = model



class Point(object):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        self.x = 0
        self.y = 0
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.name = name
        self.reward = 0

    def set_position(self, x, y):
        self.x = self.clamp(x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(y, self.y_min, self.y_max - self.icon_h)

    def get_position(self):
        return (self.x, self.y)

    def move(self, del_x, del_y):
        self.x += del_x
        self.y += del_y
        self.x = self.clamp(self.x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(self.y, self.y_min, self.y_max - self.icon_h)

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)


class Ego(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Ego, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("assets/ego.png")
        self.icon_w = 12
        self.icon_h = 12


class Pedestrian(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Pedestrian, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("assets/ped.png")
        self.icon_w = 10
        self.icon_h = 10


class Crosswalk(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Crosswalk, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("assets/cw.png")
        self.icon_w = 12
        self.icon_h = 12



class Pedestrian_Behaviour(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, ego_start=np.array([0.0, 0.0]), line_of_sight_threshold=40.0,
                 ped_cross=True, yielding=False):
        super(Pedestrian_Behaviour, self).__init__()
        self.ego_start = ego_start
        self.ego_location = self.ego_start.copy()
        self.ego_velocity = 50
        self.ped_cross = ped_cross
        self.ped_loc = foward_sim(0.5, self.ped_cross, 1, 50, 5)[0]
        self.ped_location = self.ped_loc
        self.step_env = 0
        self.line_of_sight = line_of_sight_threshold
        self.conflict = False
        self.observation_shape = (200, 300, 3)
        self.observation_space = spaces.Box(low=-250, high=250, shape=(6,), dtype=np.float32)
        self.reward = 20
        self.action_space = spaces.Discrete(3)
        self.yielding = yielding
        self.ped_speed = 1
        self.reset()

    def line_of_sight_check(self):
        return ((self.ped_loc[1] < -3 and self.ped_loc[0] == 40)
                or (self.step_env > 13) or self.conflict)

    def step(self, action):
        speed = [1, 2.5, 0]
        if self.yielding and self.ego_location[0] >= 20:
            self.ego_velocity = np.random.normal(20, 5)
        if action == 0:
            self.ped_loc = foward_sim(0.5, self.ped_cross, 1,
                                      self.ped_loc[0], self.ped_loc[1])[1]
        elif action == 1:
            self.ped_loc = foward_sim(0.5, self.ped_cross, 1,
                                      self.ped_loc[0], self.ped_loc[1], alert=True)[1]
        if (self.ped_loc[0] == 40 and self.ped_loc[1] > (-3) and self.ped_loc[1] < 0) and (
                abs((self.ego_location[0] + self.delta_t * self.ego_velocity) - 40) <= 5):
            self.conflict = True
        self.ego_location[0] += self.ego_velocity * self.delta_t
        self.ego_velocity = np.maximum(self.ego_velocity, 0)
        self.ego.move(self.ego_velocity * self.delta_t, 0)
        self.step_env += 1
        self.ped_speed = speed[action]
        self.TTA = (40 - self.ego_location[0]) / (self.delta_t * (self.ego_velocity + 1))
        self.pedestrian.set_position(self.ped_loc[0], self.ped_loc[1] + 100)
        self.compute_reward()
        self.draw_elements_on_canvas()
        terminated = bool(self.conflict)
        truncated = bool((self.step_env > 13) or
                         (self.ped_loc[1] < -3 and self.ped_loc[0] == 40 and not self.conflict))
        obs = np.array([self.ego_location[0], self.ego_velocity,
                        self.ped_loc[0], self.ped_loc[1],
                        self.ped_speed, self.TTA]).astype(np.float32)
        return obs, self.reward, terminated, truncated, {}

    def ground_truth(self):
        return self.ped_cross

    def compute_reward(self):
        if self.TTA >= 1:
            term = -0.01 * self.step_env - (3 / self.TTA)
        else:
            term = -0.01 * self.step_env
        if self.conflict:
            self.reward = -20
        elif (self.ped_loc[1] < -3 and self.ped_loc[0] == 40) and not self.conflict:
            self.reward = 20 + term
        else:
            reward = 20 if 20 < term else term
            self.reward = -20 if reward < -20 else reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        a = np.random.choice(2, 1, p=[1 / 3, 2 / 3])
        self.yielding = (a == 0)
        self.ego_location = self.ego_start.copy()
        self.ego_velocity = np.random.normal(np.random.choice([70, 80, 40]), 10)
        self.delta_t = 0.1
        self.reward = 20
        self.conflict = False
        self.ego = Ego("ego", 300, 0, 200, 0)
        self.ego.set_position(self.ego_location[0], self.ego_location[1] + 80)
        self.elements = [self.ego]
        self.pedestrian = Pedestrian("ped", 300, 0, 200, 0)
        self.pedestrian.set_position(self.ped_loc[0], self.ped_loc[1] + 100)
        self.elements.append(self.pedestrian)
        self.ped_loc = foward_sim(0.5, self.ped_cross, 1, 50, 5)[0]
        self.ped_speed = 1
        self.step_env = 0
        self.TTA = (40 - self.ego_location[0]) / (self.delta_t * (self.ego_velocity + 1))
        obs = np.array([self.ego_location[0], self.ego_velocity,
                        self.ped_loc[0], self.ped_loc[1],
                        self.ped_speed, self.TTA]).astype(np.float32)
        return obs, {}

    def draw_elements_on_canvas(self):
        self.canvas = np.ones((200, 300, 3)) * 255
        for elem in self.elements:
            if elem.icon is None:
                continue
            elem_shape = elem.icon.shape
            x, y = int(elem.x), int(elem.y)
            self.canvas[y: y + elem_shape[0], x: x + elem_shape[1]] = elem.icon

    def render(self, mode='human'):
        if mode == 'human':
            cv2.imshow("Driving Environment", self.canvas)
            cv2.waitKey(10)
        elif mode == 'rgb_array':
            return self.canvas

    def close(self):
        cv2.destroyAllWindows()



class PedestrianCross_train(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, ego_start=np.array([0.0, 0.0]),
                 line_of_sight_threshold=40.0, ped_cross=True, prior_belief=0.5):
        super(PedestrianCross_train, self).__init__()
        self.ego_start = ego_start
        self.ped_cross = ped_cross
        self.ped_loc = foward_sim(0.5, self.ped_cross, 1, 46, 5)[0]
        self.ped_location = self.ped_loc
        self.ego_location = np.array([0.0, 0.0])
        self.ego_velocity = 70
        self.ego_acceleration = 0
        self.ped_visible = False
        self.delta_t = 0.1
        self.reward = 0
        self.step_env = 0
        self.ep = 0
        self.ped_speed = 1
        self.line_of_sight = line_of_sight_threshold
        self.conflict = False
        self.observation_shape = (200, 300, 3)
        self.observation_space = spaces.Box(low=-250, high=250, shape=(5,), dtype=np.float32)
        self.total_reward = 0
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.prior_belief = prior_belief
        self.pf = ParticleFilter_train_AIF(prior_belief, 50, 5, 1.5)
        x = random.randrange(int(200 * 0.05), int(200 * 0.10))
        y = random.randrange(int(300 * 0.15), int(300 * 0.20))
        self.cw = Crosswalk("cw", 300, 0, 200, 0)
        self.ego = Ego("ego", 300, 0, 200, 0)
        self.ego.set_position(self.ego_location[0], self.ego_location[1] + 80)
        self.elements = [self.ego]
        self.pedestrian = Pedestrian("ped", 300, 0, 200, 0)
        self.pedestrian.set_position(self.ped_loc[0], self.ped_loc[1] + 100)
        self.elements.append(self.pedestrian)
        self.cw.set_position(self.line_of_sight, 85)
        self.canvas = np.ones(self.observation_shape) * 255
        self.elements.append(self.cw)
        self.draw_elements_on_canvas()
        self.seed()

    def line_of_sight_check(self):
        return ((self.success_cross() and self.ego_location[0] > 40)
                or self.conflict
                or (self.success_not_cross() and self.ego_location[0] > 40)
                or (self.ep > 20))

    def success_cross(self):
        return (self.ped_loc[1] < -3 and self.ped_loc[0] == 40)

    def success_not_cross(self):
        return (self.ped_loc[0] < 40)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        done = False
        self.ego_velocity += 20 * action[0]
        self.TTA = (40 - self.ego_location[0]) / (self.delta_t * (self.ego_velocity + 1))
        if self.ped_cross:
            if self.step_env < 2 or _ped_model is None:
                self.ped_loc = foward_sim(self.prior_belief, self.ped_cross, 1,
                                          self.ped_loc[0], self.ped_loc[1])[1]
            else:
                ob = np.array([self.ego_location[0], self.ego_velocity,
                               self.ped_loc[0], self.ped_loc[1],
                               self.ped_speed, self.TTA]).astype(np.float32)
                a = _ped_model.predict(ob, deterministic=False)[0]
                if a == 1:
                    self.ped_loc = foward_sim(self.prior_belief, self.ped_cross, 1,
                                              self.ped_loc[0], self.ped_loc[1], alert=True)[1]
                    self.ped_speed = 2.5
                elif a == 0:
                    self.ped_loc = foward_sim(self.prior_belief, self.ped_cross, 1,
                                              self.ped_loc[0], self.ped_loc[1])[1]
                    self.ped_speed = 1
                else:
                    self.ped_speed = 0
        else:
            if self.ego_location[0] >= 20 and self.ego_velocity <= 53:
                self.ped_loc = foward_sim(self.prior_belief, self.ped_cross, 1,
                                          self.ped_loc[0], self.ped_loc[1], alert=True)[1]
                self.ped_speed = 2.5
            else:
                self.ped_loc = foward_sim(self.prior_belief, self.ped_cross, 1,
                                          self.ped_loc[0], self.ped_loc[1])[1]
                self.ped_speed = 1
        if (self.ped_loc[0] == 40 and self.ped_loc[1] > (-3) and self.ped_loc[1] < 0) and (
                abs((self.ego_location[0] + self.delta_t * self.ego_velocity) - 40) <= 5):
            self.conflict = True
        self.ego_location[0] += self.ego_velocity * self.delta_t
        self.ego_velocity = np.maximum(self.ego_velocity, 0)
        self.ego.move(self.ego_velocity * self.delta_t, 0)
        if self.success_cross() or self.success_not_cross():
            self.step_env = 0
        else:
            self.step_env += 1
        self.ep += 1
        done = self.line_of_sight_check()
        self.pf.update(True, np.array(self.ped_loc))
        self.pf.resample()
        self.pedestrian.set_position(self.ped_loc[0], self.ped_loc[1] + 100)
        self.compute_reward()
        return (np.array([self.ego_location[0], self.ego_velocity,
                          self.ped_loc[0], self.ped_loc[1],
                          self.pf.particles[:, 0].mean()]).astype(np.float32),
                self.pf.compute_efe(self), done, self.step_env >= 40, {})

    def ground_truth(self):
        return self.ped_cross

    def compute_reward(self, assumed_state=None):
        reward = 0
        if self.TTA >= 1:
            term = 0.05 * self.step_env
        else:
            term = 0.05 * self.step_env
        if assumed_state is None:
            self.reward = -abs(self.ego_velocity - 70) / 200 - term
            if self.conflict and self.ped_cross:
                self.reward -= 20
            self.total_reward += self.reward
        else:
            reward = -abs(self.ego_velocity - 70) / 200 - term
            if self.conflict and assumed_state == "ped_cross":
                reward -= 20
            return reward

    def reset(self, seed=None):
        self.ego_location = np.array([0.0, 0.0])
        self.ego_velocity = 70
        self.ego_acceleration = 0
        self.ped_visible = False
        self.delta_t = 0.1
        self.reward = 0
        self.step_env = 0
        self.ep = 0
        self.conflict = False
        x = random.randrange(int(200 * 0.05), int(200 * 0.10))
        y = random.randrange(int(300 * 0.15), int(300 * 0.20))
        self.cw = Crosswalk("cw", 300, 0, 200, 0)
        self.ego = Ego("ego", 300, 0, 200, 0)
        self.ego.set_position(self.ego_location[0], self.ego_location[1] + 80)
        self.elements = [self.ego]
        self.pedestrian = Pedestrian("ped", 300, 0, 200, 0)
        self.pedestrian.set_position(self.ped_loc[0], self.ped_loc[1] + 100)
        self.elements.append(self.pedestrian)
        self.cw.set_position(self.line_of_sight, 85)
        self.canvas = np.ones(self.observation_shape) * 255
        self.elements.append(self.cw)
        self.draw_elements_on_canvas()
        self.ped_speed = 1
        self.ped_loc = foward_sim(0.5, self.ped_cross, 1, 46, 5)[0]
        self.TTA = (40 - self.ego_location[0]) / (self.delta_t * (self.ego_velocity + 1))
        if np.random.choice([1, 0], p=[self.prior_belief, 1 - self.prior_belief]) == 0:
            self.ped_cross = False
        else:
            self.ped_cross = True
        self.pf.reset()
        self.total_reward = 0
        self.prior_belief = np.random.uniform(low=0.0, high=1.0)
        self.pf = ParticleFilter_train_AIF(self.prior_belief, 50, 5, 1.5)
        return (np.array([self.ego_location[0], self.ego_velocity,
                          self.ped_loc[0], self.ped_loc[1],
                          self.pf.particles[:, 0].mean()],
                         dtype=np.float32).astype(np.float32)), {}

    def draw_elements_on_canvas(self):
        self.canvas = np.ones(self.observation_shape) * 255
        for elem in self.elements:
            if elem.icon is None:
                continue
            elem_shape = elem.icon.shape
            x, y = int(elem.x), int(elem.y)
            self.canvas[y: y + elem_shape[0], x: x + elem_shape[1]] = elem.icon

    def render(self, mode='human'):
        if mode == 'human':
            cv2.imshow("Driving Environment", self.canvas)
            cv2.waitKey(10)
        elif mode == 'rgb_array':
            return self.canvas

    def get_ego_velocity(self):
        return self.ego_velocity

    def get_ego_location(self):
        return self.ego_location

    def states(self):
        return [self.ego_velocity, self.ego_location[0], (1 if self.conflict else 0)]

    def close(self):
        cv2.destroyAllWindows()



class PedestrianCross_train_2(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, ego_start=np.array([0.0, 0.0]),
                 line_of_sight_threshold=40.0, ped_cross=True, prior_belief=0.5):
        super(PedestrianCross_train_2, self).__init__()
        self.ego_start = ego_start
        self.ped_cross = ped_cross
        self.ped_loc = foward_sim(0.5, self.ped_cross, 1, 46, 5)[0]
        self.ped_location = self.ped_loc
        self.step_env = 0
        self.ep = 0
        self.ego_location = np.array([0.0, 0.0])
        self.ego_velocity = 70
        self.ego_acceleration = 0
        self.ped_visible = False
        self.delta_t = 0.1
        self.reward = 0
        self.ped_speed = 1
        self.line_of_sight = line_of_sight_threshold
        self.conflict = False
        self.observation_shape = (200, 300, 3)
        self.observation_space = spaces.Box(low=-250, high=250, shape=(5,), dtype=np.float32)
        self.total_reward = 0
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.prior_belief = prior_belief
        self.pf = ParticleFilter_train_AIF(prior_belief, 50, 5, 1.5)
        x = random.randrange(int(200 * 0.05), int(200 * 0.10))
        y = random.randrange(int(300 * 0.15), int(300 * 0.20))
        self.cw = Crosswalk("cw", 300, 0, 200, 0)
        self.ego = Ego("ego", 300, 0, 200, 0)
        self.ego.set_position(self.ego_location[0], self.ego_location[1] + 80)
        self.elements = [self.ego]
        self.pedestrian = Pedestrian("ped", 300, 0, 200, 0)
        self.pedestrian.set_position(self.ped_loc[0], self.ped_loc[1] + 100)
        self.elements.append(self.pedestrian)
        self.cw.set_position(self.line_of_sight, 85)
        self.canvas = np.ones(self.observation_shape) * 255
        self.elements.append(self.cw)
        self.draw_elements_on_canvas()
        self.seed()

    def line_of_sight_check(self):
        return ((self.success_cross() and self.ego_location[0] > 40)
                or self.conflict
                or (self.success_not_cross() and self.ego_location[0] > 40)
                or (self.ep > 20))

    def success_cross(self):
        return (self.ped_loc[1] < -3 and self.ped_loc[0] == 40)

    def success_not_cross(self):
        return (self.ped_loc[0] < 40)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        done = False
        self.ego_velocity += 20 * action[0]
        self.TTA = (40 - self.ego_location[0]) / (self.delta_t * (self.ego_velocity + 1))
        if self.ped_cross:
            if self.step_env < 2 or _ped_model is None:
                self.ped_loc = foward_sim(self.prior_belief, self.ped_cross, 1,
                                          self.ped_loc[0], self.ped_loc[1])[1]
            else:
                ob = np.array([self.ego_location[0], self.ego_velocity,
                               self.ped_loc[0], self.ped_loc[1],
                               self.ped_speed, self.TTA]).astype(np.float32)
                a = _ped_model.predict(ob, deterministic=False)[0]
                if a == 1:
                    self.ped_loc = foward_sim(self.prior_belief, self.ped_cross, 1,
                                              self.ped_loc[0], self.ped_loc[1], alert=True)[1]
                    self.ped_speed = 2.5
                elif a == 0:
                    self.ped_loc = foward_sim(self.prior_belief, self.ped_cross, 1,
                                              self.ped_loc[0], self.ped_loc[1])[1]
                    self.ped_speed = 1
                else:
                    self.ped_speed = 0
        else:
            if self.ego_location[0] >= 20 and self.ego_velocity <= 53:
                self.ped_loc = foward_sim(self.prior_belief, self.ped_cross, 1,
                                          self.ped_loc[0], self.ped_loc[1], alert=True)[1]
                self.ped_speed = 2.5
            else:
                self.ped_loc = foward_sim(self.prior_belief, self.ped_cross, 1,
                                          self.ped_loc[0], self.ped_loc[1])[1]
                self.ped_speed = 1
        if (self.ped_loc[0] == 40 and self.ped_loc[1] > (-3) and self.ped_loc[1] < 0) and (
                abs((self.ego_location[0] + self.delta_t * self.ego_velocity) - 40) <= 5):
            self.conflict = True
        self.ego_location[0] += self.ego_velocity * self.delta_t
        self.ego_velocity = np.maximum(self.ego_velocity, 0)
        self.ego.move(self.ego_velocity * self.delta_t, 0)
        if self.success_cross() or self.success_not_cross():
            self.step_env = 0
        else:
            self.step_env += 1
        self.ep += 1
        done = self.line_of_sight_check()
        self.pf.update(True, np.array(self.ped_loc))
        self.pf.resample()
        self.pedestrian.set_position(self.ped_loc[0], self.ped_loc[1] + 100)
        self.compute_reward()
        return (np.array([self.ego_location[0], self.ego_velocity,
                          self.ped_loc[0], self.ped_loc[1],
                          np.rint(self.pf.particles[:, 0].mean())]).astype(np.float32),
                self.pf.compute_efe(self), done, self.step_env >= 40, {})

    def ground_truth(self):
        return self.ped_cross

    def compute_reward(self, assumed_state=None):
        reward = 0
        if self.TTA >= 1:
            term = 0.05 * self.step_env
        else:
            term = 0.05 * self.step_env
        if assumed_state is None:
            self.reward = -abs(self.ego_velocity - 70) / 200 - term
            if self.conflict and self.ped_cross:
                self.reward -= 20
            self.total_reward += self.reward
        else:
            reward = -abs(self.ego_velocity - 70) / 200 - term
            if self.conflict and assumed_state == "ped_cross":
                reward -= 20
            return reward

    def reset(self, seed=None):
        self.ego_location = np.array([0.0, 0.0])
        self.ego_velocity = 70
        self.ego_acceleration = 0
        self.ped_visible = False
        self.delta_t = 0.1
        self.reward = 0
        self.step_env = 0
        self.ep = 0
        self.conflict = False
        x = random.randrange(int(200 * 0.05), int(200 * 0.10))
        y = random.randrange(int(300 * 0.15), int(300 * 0.20))
        self.cw = Crosswalk("cw", 300, 0, 200, 0)
        self.ego = Ego("ego", 300, 0, 200, 0)
        self.ego.set_position(self.ego_location[0], self.ego_location[1] + 80)
        self.elements = [self.ego]
        self.pedestrian = Pedestrian("ped", 300, 0, 200, 0)
        self.pedestrian.set_position(self.ped_loc[0], self.ped_loc[1] + 100)
        self.elements.append(self.pedestrian)
        self.cw.set_position(self.line_of_sight, 85)
        self.canvas = np.ones(self.observation_shape) * 255
        self.elements.append(self.cw)
        self.draw_elements_on_canvas()
        self.ped_speed = 1
        self.ped_loc = foward_sim(0.5, self.ped_cross, 1, 46, 5)[0]
        self.TTA = (40 - self.ego_location[0]) / (self.delta_t * (self.ego_velocity + 1))
        if np.random.choice([1, 0], p=[self.prior_belief, 1 - self.prior_belief]) == 0:
            self.ped_cross = False
        else:
            self.ped_cross = True
        self.total_reward = 0
        self.pf.reset()
        self.prior_belief = np.random.uniform(low=0.0, high=1.0)
        self.pf = ParticleFilter_train_AIF(self.prior_belief, 50, 5, 1.5)
        return (np.array([self.ego_location[0], self.ego_velocity,
                          self.ped_loc[0], self.ped_loc[1],
                          np.rint(self.pf.particles[:, 0].mean())],
                         dtype=np.float32).astype(np.float32)), {}

    def draw_elements_on_canvas(self):
        self.canvas = np.ones(self.observation_shape) * 255
        for elem in self.elements:
            if elem.icon is None:
                continue
            elem_shape = elem.icon.shape
            x, y = int(elem.x), int(elem.y)
            self.canvas[y: y + elem_shape[0], x: x + elem_shape[1]] = elem.icon

    def render(self, mode='human'):
        if mode == 'human':
            cv2.imshow("Driving Environment", self.canvas)
            cv2.waitKey(10)
        elif mode == 'rgb_array':
            return self.canvas

    def get_ego_velocity(self):
        return self.ego_velocity

    def get_ego_location(self):
        return self.ego_location

    def states(self):
        return [self.ego_velocity, self.ego_location[0], (1 if self.conflict else 0)]

    def close(self):
        cv2.destroyAllWindows()



from gymnasium.envs.registration import register as _register

_register(id='Pedestrian_Behaviour-v1',
          entry_point='src.environments:Pedestrian_Behaviour')
_register(id='PedestrianCross_train-v1',
          entry_point='src.environments:PedestrianCross_train')
_register(id='PedestrianCross_train_2-v1',
          entry_point='src.environments:PedestrianCross_train_2')
