from collections import deque


class Call:
    def __init__(self, *expected_input, returns):
        self.expected_input = expected_input
        self.returns = returns


class MockFilter:
    def __init__(self, *expected_calls):
        self.expected_calls = deque(expected_calls)

    def __call__(self, *args):
        call = self.expected_calls.popleft()
        assert args == call.expected_input
        return call.returns