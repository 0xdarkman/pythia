from abc import abstractmethod, ABCMeta
import numpy as np


class QFunction(object, metaclass=ABCMeta):
    def __init__(self, action_space):
        self.action_space = action_space

    @abstractmethod
    def __getitem__(self, state_action):
        pass

    @abstractmethod
    def learn(self, state, action, signal):
        pass

    def unpack_state_action(self, state_action):
        state, action = self.retrieve_state_action(state_action)
        state = self.box_state(state)

        state.append(action)
        return state

    def retrieve_state_action(self, state_action):
        state, action = state_action
        self.check_action(action)
        return state, action

    def all_values_of_state(self, state):
        state = self.box_state(state)
        num_actions = len(self.action_space)
        vs = np.zeros(num_actions)
        for i, a in enumerate(self.action_space):
            vs[i] = self[list(state), a]
        return vs

    @staticmethod
    def box_state(state):
        if not isinstance(state, list):
            state = [state]
        return list(state)

    def check_action(self, action):
        if action not in self.action_space:
            raise InvalidAction("The action {} is not part of action space {}".format(action, str(self.action_space)))


class InvalidAction(Exception):
    pass
