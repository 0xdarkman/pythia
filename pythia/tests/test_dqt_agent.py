import random

import pytest
import numpy as np
from collections import deque

from pythia.core.agents.dqt_agent import DQTAgent


class ModelStub:
    def __init__(self, state_predictions=None):
        if state_predictions is None:
            self.state_predictions = {}
        else:
            self.state_predictions = state_predictions

    def predict(self, state):
        return np.array(self.state_predictions.get(state, [0.0, 0.0]))

    def train(self, batch):
        pass

    def interpolate(self, model, factor):
        pass


class ModelSpy(ModelStub):
    def __init__(self, state_predictions=None):
        super().__init__(state_predictions)
        self.received_states = []
        self.received_batch = None
        self.received_interpolation_model = None
        self.received_interpolation_factor = None

    def predict(self, state):
        self.received_states.append(state)
        return super().predict(state)

    def train(self, batch):
        self.received_batch = batch

    def interpolate(self, model, factor):
        self.received_interpolation_model = model
        self.received_interpolation_factor = factor


class InstanceFactory:
    def __init__(self, *instances):
        self.instances = deque(instances)

    def __call__(self):
        return self.instances.popleft()


@pytest.fixture
def model_spy():
    return ModelSpy()


def make_agent(target_model=ModelStub(), model=ModelStub(), alpha=1.0, gamma=1.0, batch_size=1, memory=1000,
               interpolation=1.0, actions=None):
    if actions is None:
        actions = ["Action1", "Action2"]
    return DQTAgent(InstanceFactory(target_model, model), actions=actions, alpha=alpha, gamma=gamma,
                    batch_size=batch_size, memory=memory, interpolation=interpolation)


def test_start_selects_best_action_from_target_model():
    dqt = make_agent(target_model=ModelStub({"S": [1.0, 2.0]}))
    a = dqt.start("S")
    assert a == "Action2"


def test_target_model_receives_state_on_start(model_spy):
    dqt = make_agent(target_model=model_spy)
    dqt.start("State")
    assert model_spy.received_states == ["State"]


def test_step_returns_best_action_from_target_model():
    dqt = make_agent(target_model=ModelStub({"S": [1.0, 2.0]}))
    dqt.start("S0")
    a = dqt.step("S", [0.0, 0.0])
    assert a == "Action2"


def test_target_model_receives_state_when_stepping(model_spy):
    dqt = make_agent(target_model=model_spy)
    dqt.start("S0")
    dqt.step("State", [0.0, 0.0])
    assert model_spy.received_states[1] == "State"


def test_step_trains_model_when_batch_size_is_reached():
    model_spy = ModelSpy({"S0": [2.0, -1.0]})
    dqt = make_agent(target_model=ModelStub({"S0": [2.0, -1.0], "S1": [1.0, 1.0]}), model=model_spy)
    dqt.start("S0")
    dqt.step("S1", [-1.0, 2.0])
    assert model_spy.received_batch["inputs"][0] == "S0"
    assert (model_spy.received_batch["targets"] == ([[-1.0 + 1.0 - 2.0, 2.0 + 1.0 + 1.0]])).all()


def test_reward_signal_is_calculated_taking_gamma_and_alpha_into_account():
    model_spy = ModelSpy({"S0": [2.0, -1.0]})
    dqt = make_agent(target_model=ModelStub({"S0": [2.0, -1.0], "S1": [1.0, 1.0]}),
                     model=model_spy, alpha=0.5, gamma=0.9)
    dqt.start("S0")
    dqt.step("S1", [-1.0, 2.0])
    assert model_spy.received_batch["inputs"][0] == "S0"
    assert (model_spy.received_batch["targets"] == [[0.5 * (-1 + (0.9 * 1) - 2), 0.5 * (2 + (0.9 * 1) + 1)]]).all()


def test_model_is_not_trained_when_batch_size_is_not_reached(model_spy):
    dqt = make_agent(model=model_spy, batch_size=2)
    dqt.start("S0")
    dqt.step("S1", [0.0, 0.0])
    assert model_spy.received_batch is None


def test_batch_is_composed_of_previous_states_when_its_size_has_been_reached():
    model_spy = ModelSpy({"S0": [-1.0, 1.0], "S1": [-1.0, 1.0]})
    dqt = make_agent(target_model=ModelStub({"S0": [-1.0, 1.0], "S1": [-1.0, 1.0], "S2": [-1.0, 1.0]}), model=model_spy,
                     batch_size=2)
    dqt.start("S0")
    dqt.step("S1", [-1.0, 1.0])
    dqt.step("S2", [-2.0, 2.0])
    assert (model_spy.received_batch["inputs"] == ["S0", "S1"]).all()
    assert (model_spy.received_batch["targets"] == [[-1.0, 1.0], [-2.0, 2.0]]).all()


def test_randomly_select_batches_from_memory(model_spy):
    random.seed(42)
    dqt = make_agent(model=model_spy, batch_size=2)
    dqt.start("S0")
    for e in range(1, 100):
        dqt.step("S{}".format(e), [0.0, 0.0])
    assert (model_spy.received_batch["inputs"] == ["S29", "S30"]).all()
    for e in range(1, 100):
        dqt.step("S{}".format(e), [0.0, 0.0])
    assert (model_spy.received_batch["inputs"] == ["S36", "S37"]).all()


def test_when_memory_is_full_forgets_last_state(model_spy):
    random.seed(42)
    dqt = make_agent(model=model_spy, memory=1)
    dqt.start("S0")
    dqt.step("S1", [0.0, 0.0])
    dqt.step("S2", [1.0, 1.0])
    assert (model_spy.received_batch["inputs"] == ["S1"]).all()


def test_target_model_is_interpolated_with_model_after_training(model_spy):
    model = ModelStub()
    dqt = make_agent(target_model=model_spy, model=model, interpolation=0.5)
    dqt.start("S0")
    dqt.step("S1", [0.0, 0.0])
    assert model_spy.received_interpolation_model == model
    assert model_spy.received_interpolation_factor == 0.5


def test_agent_does_not_allow_exchanging_a_token_to_itself():
    dqt = make_agent(target_model=ModelStub({"S0": [1.0, -1.0], "S1": [1.0, -1.0]}))
    assert dqt.start("S0") == "Action1"
    assert dqt.step("S1", [1.0, -1.0]) == "Action2"


def test_none_actions_are_always_allowed():
    dqt = make_agent(target_model=ModelStub({"S0": [1.0, -1.0], "S1": [1.0, -1.0]}), actions=[None, "Action1"])
    assert dqt.start("S0") is None
    assert dqt.step("S1", [1.0, -1.0]) is None
