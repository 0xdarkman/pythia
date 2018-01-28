from abc import ABCMeta, abstractmethod


class QModel(object, metaclass=ABCMeta):
    def __init__(self, input_size):
        self.input_size = input_size

    def predict(self, state):
        self.check_state(state[0])
        return self.do_prediction(state)

    @abstractmethod
    def do_prediction(self, state):
        pass

    def check_state(self, state):
        state_size = len(state)
        if state_size != self.input_size:
            raise InvalidDataSize(
                "Input size required is {}, but received state has length {}".format(self.input_size, state_size))

    def train(self, states, targets):
        self.check_state(states[0])
        self.check_target(targets[0])
        self.check_input_arrays(states, targets)

        self.do_training(states, targets)

    @abstractmethod
    def do_training(self, states, targets):
        pass

    @staticmethod
    def check_target(target):
        target_size = len(target)
        if target_size != 1:
            raise InvalidDataSize(
                "Output size of the model is {}, the training model_data has length {}".format(1, target_size))

    @staticmethod
    def check_input_arrays(states, targets):
        num_states = len(states)
        num_targets = len(targets)
        if num_states != num_targets:
            raise InvalidDataSize(
                "The number of states ({}) differs from the numbers of targets ({})".format(num_states, num_targets))


class InvalidDataSize(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)
