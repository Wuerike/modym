import logging
import os
import numpy as np
import gymnasium as gym

from pyfmi import load_fmu

logger = logging.getLogger(__name__)

class ModymEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(self, config:dict, log_level):
        """
        :param config: dictionary with model specifications:
            fmu_mode            FMU model mode: "CS" or "ME"
            fmi_version         FMI model version: 1 or 2
            model_path          path to the model FMU. Absolute path is advised
            time_step           duration, in seconds, of each simulation step
            start_time          time to start simulation at each reset, usually 0
            positive_reward     positive reward value
            negative_reward     negative reward value
            model_parameters    dictionary of default parameters values
            model_input_names   names of parameters to be used as action.
            model_output_names  names of parameters to be used as state descriptors.
        :param log_level: level of logging to be used
        """
        logger.setLevel(log_level)

        self.fmi_version = config.get('fmi_version')
        self.model_name = config.get('model_path').split(os.path.sep)[-1]
        self.positive_reward = config.get('positive_reward')
        self.negative_reward = config.get('negative_reward')
        self.start_time = config.get('start_time')
        self.tau = config.get('time_step')
        self.model_input_names = config.get('model_input_names')
        self.model_output_names = config.get('model_output_names')
        self.model_parameters = config.get('model_parameters')

        self.model = load_fmu(
            fmu=config.get('model_path'), 
            kind=config.get('fmu_mode'), 
            log_level=5,
            log_file_name='./trash/log.txt'
        )

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

    def reset(self, seed = None, options = None):
        """
        Determines restart procedure of the environment
        :param seed: required by Gymnasium API but not used at this implementation.
        :param options: dict with parameters names and its values to apply on reset.
            If None default values from initial config will be used.
        :return: environment state after restart
        """
        logger.debug("Experiment reset was called. Resetting the model.")

        self.model.reset()
        if self.fmi_version == 2:
            self.model.setup_experiment(start_time=0)

        self._set_init_parameter(options)
        self.model.initialize()

        # get initial state of the model from the fmu
        self.start = 0
        self.stop = self.start_time
        self.state = self._do_simulation()

        self.start = self.start_time
        self.stop = self.start + self.tau
        self.done = self._is_done()
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        """
        Determines how one simulation step is performed for the environment.
        Simulation step is execution of the given action in a current state of the environment.
        :param action: action to be executed.
        :return: resulting state
        """
        logger.debug("Experiment next step was called.")
        if self.done:
            logging.warning(
                """You are calling 'step()' even though this environment has already returned done = True.
                You should always call 'reset()' once you receive 'done = True' -- any further steps are
                undefined behavior."""
            )
            return np.array(self.state), self.negative_reward, self.done, {}

        # check if action is a list. If not - create list of length 1
        try:
            iter(action)
        except TypeError:
            action = [action]
            logging.warning("Model input values (action) should be passed as a list")

        # Check if number of model inputs equals number of values passed
        if len(action) != len(list(self.model_input_names)):
            message = f"List of values for model inputs should be of the length {len(list(self.model_input_names))}," \
                      f"equal to the number of model inputs. Actual length {len(action)}"
            logging.error(message)
            raise ValueError(message)

        # Set input values of the model
        logger.debug("model input: {}, values: {}".format(self.model_input_names, action))
        self.model.set(list(self.model_input_names), list(action))

        # Simulate and observe result state
        self.state = self._do_simulation()

        # Check if experiment has finished
        self.done = self._is_done()

        # Move simulation time interval if experiment continues
        if not self.done:
            logger.debug("Experiment step done, experiment continues.")
            self.start = self.stop
            self.stop += self.tau
        else:
            logger.debug("Experiment step done, experiment done.")

        return np.array(self.state, dtype=np.float32), self._reward_policy(), self.done, False, {}

    def _get_action_space(self):
        """
        Returns action space according to Gymnasium API requirements.
        Should be implemented by your environment class.

        :return: one of gymnasium.spaces classes that describes action space according to environment specifications.
        """
        pass

    def _get_observation_space(self):
        """
        Returns state space according to Gymnasium API API requirements.
        Should be implemented by your environment class.

        :return: one of gymnasium.spaces classes that describes state space according to environment specifications.
        """
        pass

    def _do_simulation(self):
        """
        Executes simulation by FMU in the time interval [start_time; stop_time]
        currently saved in the environment.

        :return: resulting state of the environment.
        """
        options = self.model.simulate_options()
        options['ncp'] = 50
        options['initialize'] = False
        options['silent_mode'] = True
        options["result_handling"] = "memory"
        options['result_file_name'] = './trash/result.txt'

        logger.debug("Simulation started for time interval {}-{}".format(self.start, self.stop))
        result = self.model.simulate(start_time=self.start, final_time=self.stop, options=options)

        model_outputs = self.model_output_names
        return tuple([result.final(k) for k in model_outputs])

    def _is_done(self):
        """
        Determines logic when experiment is considered to be done.
        Should be implemented by your environment class.

        :return: boolean flag if current state of the environment indicates that experiment has ended.
        """
        pass

    def _reward_policy(self):
        """
        Determines reward based on the current environment state.
        Should be implemented by your environment class.

        :return: reward associated with the current state as an integer value
        """
        pass

    def _set_init_parameter(self, options = None):
        """
        Sets initial parameters of a model, usually called at each reset.

        :param options: dict with parameters names and its values.
            If None default values from initial config will be used.

        :return: environment
        """
        if options:
            self.model.set(options.get('params'), options.get('values'))

        elif self.model_parameters is not None:
            self.model.set(list(self.model_parameters), list(self.model_parameters.values()))

        return self
