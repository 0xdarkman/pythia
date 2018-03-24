from collections import deque


class Memory:
    def __init__(self, size):
        self._input_records = deque(maxlen=size)
        self._signal_records = deque(maxlen=size)

    @property
    def is_full(self):
        return len(self._input_records) == self._input_records.maxlen

    def record(self, ann_input, signal):
        self._input_records.append(ann_input)
        self._signal_records.append([signal])

    @property
    def inputs(self):
        return self._input_records

    @property
    def signals(self):
        return self._signal_records


class QNeuronal:
    def __init__(self, ann, n, memory_size=None):
        """
        This class implements a nonlinear Q function approximation using an artificial neural network. The ANN is
        trained on the environment state and the Q-Value signals to be able to predict Q-Values given a specific state.

        States are provided to the ANN as a list of lists, i.e.:
        ``[[0, 0], [0, 1], [0, 1]]``
        Signals are provided to the ANN as list of of single element lists, i.e.:
        ``[[0.5], [-0.9], [1.0]]``

        :param ann: Artificial Neural Network to be used for Q-Value prediction.
        :param n: Number of actions
        :param memory_size: (optional) Size of recorded state, signal pairs that are passed to the ANN to as a batch.
        """
        self.ann = ann
        self.memory = Memory(1 if memory_size is None else memory_size)
        self.action_space = list(range(n))

    def __getitem__(self, state_action):
        s, a = state_action
        return self.ann.predict([self._make_input(list(s), a)])[0][0]

    @staticmethod
    def _make_input(s, a):
        s.append(a)
        return s

    def learn(self, state, action, signal):
        self.memory.record(self._make_input(state, action), signal)
        if self.memory.is_full:
            self.ann.train(self.memory.inputs, self.memory.signals)

    def all_values_of_state(self, state):
        return [self[state, a] for a in self.action_space]
