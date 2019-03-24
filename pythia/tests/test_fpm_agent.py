import os

import pytest

from pythia.core.agents.fpm_agent import FpmAgent


class MemorySpy:
    def __init__(self):
        self.batch = BatchStub((None, None), None)
        self.last_record = (PriceStub(), PortfolioStub())
        self.is_ready = True
        self.received_state = None
        self.received_weights_to_update = None
        self.queried_batch_size = None
        self.received_save_file = None
        self.received_restore_file = None

    def set_batch(self, batch):
        self.batch = batch

    def set_last_record(self, prices, portfolio):
        self.last_record = (prices, portfolio)

    def set_ready(self, is_ready):
        self.is_ready = is_ready

    def record(self, prices, portfolio):
        self.received_state = (prices, portfolio)

    def ready(self):
        return self.is_ready

    def get_latest(self):
        return self.last_record

    def get_random_batch(self, size):
        self.queried_batch_size = size
        return self.batch

    def update(self, batch):
        self.received_weights_to_update = batch.predictions

    def save(self, file):
        self.received_save_file = file

    def restore(self, file):
        self.received_restore_file = file


class AnnSpy:
    def __init__(self):
        self.portfolio = None
        self.training_predictions = None
        self.predicted = None
        self.received_batch = None
        self.received_save_file = None
        self.received_restore_file = None

    def set_predictions(self, portfolio):
        self.portfolio = portfolio

    def set_training_predictions(self, predictions):
        self.training_predictions = predictions

    def predict(self, prices, previous_portfolio):
        self.predicted = (prices, previous_portfolio)
        return self.portfolio

    def train(self, states, future_prices):
        self.received_batch = (states, future_prices)
        return self.training_predictions

    def save(self, file):
        self.received_save_file = file

    def restore(self, file):
        self.received_restore_file = file


class LoggerSpy:
    def __init__(self):
        self.received_info_logs = list()

    def info(self, msg):
        self.received_info_logs.append(msg)


class PriceStub:
    def __init__(self, index=0):
        self.index = index

    def __repr__(self):
        return "PriceStub({})".format(self.index)

    def __eq__(self, other):
        return self.index == other.index


class PortfolioStub:
    def __init__(self, index=0):
        self.index = index

    def __repr__(self):
        return "PortfolioStub({})".format(self.index)

    def __eq__(self, other):
        return self.index == other.index


class BatchStub:
    def __init__(self, states, future, is_empty=False):
        self.prices, self.weights = states
        self.future = future
        self.empty = is_empty


class RandomPortfolio:
    def __repr__(self):
        return "RandomPortfolio"

    def __eq__(self, other):
        return isinstance(other, RandomPortfolio)


@pytest.fixture
def ann():
    return AnnSpy()


@pytest.fixture
def memory():
    return MemorySpy()


@pytest.fixture
def logger():
    return LoggerSpy()


@pytest.fixture
def initial_portfolio():
    return PortfolioStub()


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def config(initial_portfolio, batch_size):
    return {"setup": {"initial_portfolio": initial_portfolio}, "training": {"batch_size": batch_size}}


@pytest.fixture
def agent(ann, memory, config, logger):
    return FpmAgent(ann, memory, lambda: RandomPortfolio(), config, logger)


def make_prices(index=0):
    return PriceStub(index)


def make_portfolio(index=0):
    return PortfolioStub(index)


def make_state(index=0):
    return make_prices(index), make_portfolio(index)


def make_batch(states, future):
    return BatchStub(states, future)


def make_empty_batch():
    return BatchStub((None, None), None, True)


def test_memory_records_first_prices_and_initial_portfolio(agent, memory, initial_portfolio):
    price = make_prices()
    agent.step(price)
    assert memory.received_state == (price, initial_portfolio)


def test_ann_predicts_last_recorded_memory(agent, ann, memory):
    memory.set_last_record(make_prices(1), make_portfolio(1))
    agent.step(make_prices())
    assert ann.predicted == (make_prices(1), make_portfolio(1))


def test_ann_prediction_is_the_returned_action(agent, ann):
    ann.set_predictions(make_portfolio(1))
    a = agent.step(make_prices())
    assert a == make_portfolio(1)


def test_memory_is_queried_for_the_configured_batch_size(agent, memory, batch_size):
    agent.step(make_prices())
    assert memory.queried_batch_size == batch_size


def test_ann_is_trained_on_batch_from_memory(agent, ann, memory):
    memory.set_batch(make_batch(make_state(1), make_prices(2)))
    agent.step(make_prices())
    assert ann.received_batch == (make_state(1), make_prices(2))


def test_memory_is_updated_with_predicted_weights_from_training(agent, ann, memory):
    ann.set_training_predictions(make_portfolio(1))
    agent.step(make_prices())
    assert memory.received_weights_to_update == make_portfolio(1)


def test_record_previously_predicted_portfolio_on_next_step(agent, ann, memory):
    ann.set_predictions(make_portfolio(1))
    agent.step(make_prices(1))
    ann.set_predictions(make_portfolio(2))
    agent.step(make_prices(2))
    assert memory.received_state == (make_prices(2), make_portfolio(1))


def test_agent_returns_random_actions_when_memory_is_not_ready(agent, memory):
    memory.set_ready(False)
    a = agent.step(make_prices())
    assert a == RandomPortfolio()


def test_agent_records_random_portfolio_when_memory_is_not_ready(agent, memory):
    memory.set_ready(False)
    agent.step(make_prices(1))
    agent.step(make_prices(2))
    assert memory.received_state == (make_prices(2), RandomPortfolio())


def test_agent_does_not_train_on_empty_batches(agent, ann, memory):
    memory.set_batch(make_empty_batch())
    agent.step(make_prices(1))
    assert ann.received_batch is None


def test_agent_saves_model_and_memory(agent, ann, memory):
    path = "save_directory"
    agent.save(path)
    assert ann.received_save_file == os.path.join(path, agent.model_file_name)
    assert memory.received_save_file == os.path.join(path, agent.memory_file_name)


def test_agent_restores_model_and_memory(agent, ann, memory):
    path = "restore_directory"
    agent.restore(path)
    assert ann.received_restore_file == os.path.join(path, agent.model_file_name)
    assert memory.received_restore_file == os.path.join(path, agent.memory_file_name)


def test_info_logs_saving(agent, logger):
    path = "save_directory"
    agent.save(path)
    assert os.path.join(path, agent.model_file_name) in logger.received_info_logs[0]
    assert os.path.join(path, agent.memory_file_name) in logger.received_info_logs[1]


def test_info_logs_restoring(agent, logger):
    path = "restore_directory"
    agent.restore(path)
    assert os.path.join(path, agent.model_file_name) in logger.received_info_logs[0]
    assert os.path.join(path, agent.memory_file_name) in logger.received_info_logs[1]
