import pytest

from pythia.core.agents.fpm_agent import FpmAgent


class MemorySpy:
    def __init__(self):
        self.batch = BatchStub((None, None), None)
        self.received_state = None
        self.received_weights_to_update = None
        self.queried_batch_size = None

    def set_batch(self, batch):
        self.batch = batch

    def record(self, prices, portfolio):
        self.received_state = (prices, portfolio)

    def get_random_batch(self, size):
        self.queried_batch_size = size
        return self.batch

    def update(self, batch):
        self.received_weights_to_update = batch.predictions


class AnnSpy:
    def __init__(self):
        self.portfolio = None
        self.training_predictions = None
        self.predicted = None
        self.received_batch = None

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
    def __init__(self, states, future):
        self.prices, self.weights = states
        self.future = future


@pytest.fixture
def ann():
    return AnnSpy()


@pytest.fixture
def memory():
    return MemorySpy()


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
def agent(ann, memory, config):
    return FpmAgent(ann, memory, config)


def make_prices(index=0):
    return PriceStub(index)


def make_portfolio(index=0):
    return PortfolioStub(index)


def make_state(index=0):
    return make_prices(index), make_portfolio(index)


def make_batch(states, future):
    return BatchStub(states, future)


def test_memory_records_first_prices_and_initial_portfolio(agent, memory, initial_portfolio):
    price = make_prices()
    agent.step(price)
    assert memory.received_state == (price, initial_portfolio)


def test_ann_predicts_first_prices_and_initial_portfolio(agent, ann, initial_portfolio):
    price = make_prices()
    agent.step(price)
    assert ann.predicted == (price, initial_portfolio)


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


def test_use_previously_predicted_portfolio_in_prediction_on_next_step(agent, ann, memory):
    ann.set_predictions(make_portfolio(1))
    agent.step(make_prices(1))
    ann.set_predictions(make_portfolio(2))
    agent.step(make_prices(2))
    assert ann.predicted == (make_prices(2), make_portfolio(1))
