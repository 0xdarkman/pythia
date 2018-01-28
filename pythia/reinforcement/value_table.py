from pythia.utils.ansi_formats import Formats


class ValueTable(object):
    def __init__(self, initializer=None):
        if initializer is None:
            initializer = self.zero_init
        self.initializer = initializer
        self.storage = {}

    @staticmethod
    def zero_init():
        return 0

    def get_state(self, state):
        key = self.to_key(state)
        if key not in self.storage:
            return self.initializer()
        return self.storage[key]

    @staticmethod
    def to_key(state):
        return ' '.join(map(str, state))

    def update(self, state, value):
        key = self.to_key(state)
        self.storage[key] = value

    def print_states(self, state_sequences):
        print_strings = []
        for states in state_sequences:
            states_strings = []
            max_value = -float("inf")
            max_idx = 0
            for i, state in enumerate(states):
                key = self.to_key(state)
                if key in self.storage:
                    value = self.storage[key]
                    states_strings.append(
                        "{" + str(state) + ", " + Formats.REWARD + "{:6.2f}".format(
                            value) + Formats.END + "}")
                    if value > max_value:
                        max_value = value
                        max_idx = i
                else:
                    print("The state", state, "hasn't been set yet.")
                    return

            states_strings[max_idx] = states_strings[max_idx].replace(Formats.REWARD, Formats.HIGHLIGHT)
            print_strings.append(', '.join(states_strings))

        print('\n'.join(print_strings))

    def __getitem__(self, state):
        return self.get_state(state)

    def __setitem__(self, state, value):
        self.update(state, value)

    def print_all(self):
        return self.print_all_sorted_by(None)

    def print_all_sorted_by(self, state_index):
        out_strings = []
        sorter = None
        if state_index is not None:
            sorter = lambda t: t.split()[state_index]

        for key in sorted(self.storage, key=sorter):
            out_strings.append("[" + str(key) + "] = " + str(self.storage[key]))

        print('\n'.join(out_strings))
