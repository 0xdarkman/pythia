from collections import deque

from pythia.core.reinforcement.q_function import QFunction


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


class QNeuronal(QFunction):
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
        super().__init__(list(range(n)))
        self.ann = ann
        self.memory = Memory(1 if memory_size is None else memory_size)

    def __getitem__(self, state_action):
        return self.ann.predict([self._make_input_state(state_action)])[0][0]

    def learn(self, state, action, signal):
        self.memory.record(self._make_input_state((state, action)), signal)
        if self.memory.is_full:
            self.ann.train(self.memory.inputs, self.memory.signals)
