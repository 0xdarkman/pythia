class QFunction:
    def __init__(self, action_space):
        self.action_space = action_space

    def _make_input_state(self, state_action):
        state, action = self._retrieve_state_action(state_action)
        state = self.box_state(state)

        state.append(action)
        return state

    def _retrieve_state_action(self, state_action):
        state, action = state_action
        self.check_action(action)
        return state, action

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
