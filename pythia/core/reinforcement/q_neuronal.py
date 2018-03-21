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
        self._signal_records.append(signal)

    @property
    def inputs(self):
        return self._input_records

    @property
    def signals(self):
        return self._signal_records


class QNeuronal:
    def __init__(self, ann, memory_size=None):
        self.ann = ann
        self.memory = Memory(1 if memory_size is None else memory_size)

    def __getitem__(self, state_action):
        s, a = state_action
        return self.ann.predict(self._make_input(s, a))

    @staticmethod
    def _make_input(s, a):
        s.append(a)
        return s

    def learn(self, state, action, signal):
        self.memory.record(self._make_input(state, action), signal)
        if self.memory.is_full:
            self.ann.train(self.memory.inputs, self.memory.signals)