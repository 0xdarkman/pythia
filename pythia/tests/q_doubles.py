from pythia.core.reinforcement.q_table import QTable


class QFunctionWrapper(QTable):
    def set_state_action_values(self, state, zero_value, one_value):
        self[state, 0] = zero_value
        self[state, 1] = one_value