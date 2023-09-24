import math
import logging

import numpy as np
import cartpole.utils.rendering as rendering

from gymnasium import spaces
from modym.modym_env import ModymEnv


logger = logging.getLogger(__name__)


NINETY_DEGREES_IN_RAD = (90 / 180) * math.pi
TWELVE_DEGREES_IN_RAD = (12 / 180) * math.pi


class CartPoleEnv(ModymEnv):
    def __init__(
            self,
            config: dict,
            log_level="DEBUG"
        ):

        logger.setLevel(log_level)

        self.force = config.get('force')
        self.x_threshold = 2.4
        self.theta_threshold = TWELVE_DEGREES_IN_RAD

        self.viewer = None
        self.display = None
        self.pole_transform = None
        self.cart_transform = None

        super().__init__(config, log_level)

    def step(self, action):
        action = self.force if action > 0 else -self.force
        return super().step([action])

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return True

        screen_width = 600
        screen_height = 400

        scene_width = self.x_threshold * 2
        scale = screen_width / scene_width
        cart_y = 100  # TOP OF CART
        pole_width = 10.0
        pole_len = scale * 1.0
        cart_width = 50.0
        cart_height = 30.0

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)

            # add cart to the rendering
            l, r, t, b = -cart_width / 2, cart_width / 2, cart_height / 2, -cart_height / 2
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.cart_transform = rendering.Transform()
            cart.add_attr(self.cart_transform)
            self.viewer.add_geom(cart)

            # add pole to the rendering
            pole_joint_depth = cart_height / 4
            l, r, t, b = -pole_width / 2, pole_width / 2, pole_len - pole_width / 2, -pole_width / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.pole_transform = rendering.Transform(translation=(0, pole_joint_depth))
            pole.add_attr(self.pole_transform)
            pole.add_attr(self.cart_transform)
            self.viewer.add_geom(pole)

            # add joint to the rendering
            joint = rendering.make_circle(pole_width / 2)
            joint.add_attr(self.pole_transform)
            joint.add_attr(self.cart_transform)
            joint.set_color(.5, .5, .8)
            self.viewer.add_geom(joint)

            # add bottom line to the rendering
            track = rendering.Line((0, cart_y - cart_height / 2), (screen_width, cart_y - cart_height / 2))
            track.set_color(0, 0, 0)
            self.viewer.add_geom(track)

        # set new position according to the environment current state
        x, _, theta, _ = self.state
        cart_x = x * scale + screen_width / 2.0  # MIDDLE OF CART

        self.cart_transform.set_translation(cart_x, cart_y)
        self.pole_transform.set_rotation(theta - NINETY_DEGREES_IN_RAD)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def reset(self, seed = None, options:dict = {}):

        values = self.np_random.uniform(
            low= np.array([10, 1, NINETY_DEGREES_IN_RAD-0.05, -0.05]), 
            high= np.array([10, 1, NINETY_DEGREES_IN_RAD+0.05, 0.05]), 
            size=(4,)
        )

        options['params'] = ['m_cart', 'm_pole', 'theta_0', 'theta_dot_0']
        options['values'] = values

        return super().reset(seed, options)

    def close(self):
        return self.render(close=True)

    def _get_action_space(self):
        return spaces.Discrete(2)

    def _get_observation_space(self):
        return spaces.Box(
            np.array([-self.x_threshold,-np.inf, NINETY_DEGREES_IN_RAD-self.theta_threshold, -np.inf]), 
            np.array([self.x_threshold, np.inf, NINETY_DEGREES_IN_RAD+self.theta_threshold, np.inf])
        )

    def _reward_policy(self):
        return self.negative_reward if self.done else self.positive_reward
    
    def _is_done(self):
        x, x_dot, theta, theta_dot = self.state
        logger.debug("x: {0}, x_dot: {1}, theta: {2}, theta_dot: {3}".format(x, x_dot, theta, theta_dot))

        theta = abs(theta - NINETY_DEGREES_IN_RAD)

        if abs(x) > self.x_threshold:
            done = True
        elif theta > self.theta_threshold:
            done = True
        else:
            done = False

        return done
