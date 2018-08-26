"""
https://poloniex.com/public?command=returnChartData&currencyPair=BTC_ETH&period=1800&start=1535211000
[{"date":1535211000,"high":0.04164207,"low":0.04145529,"open":0.04155495,"close":0.04159499,"volume":1.16195203,"quoteVolume":27.94722352,"weightedAverage":0.04157665},{"date":1535212800,"high":0.04159809,"low":0.04151278,"open":0.04159802,"close":0.04151278,"volume":0.28771422,"quoteVolume":6.91684214,"weightedAverage":0.04159618}]
"""
import pytest
import pythia.core.streams.poloniex_history as sut

from pythia.core.streams.poloniex_history import PoloniexHistory


class UrlibSpy:
    class ResultStub:
        def __init__(self, res):
            self.res = res

        def read(self):
            return self.res.encode('UTF-8')

    def __init__(self):
        self.received_urls = []
        self.returns = '[{"date":1535211000,"high":0.04164207,"low":0.04145529,"open":0.04155495,"close":0.04159499,"volume":1.16195203,"quoteVolume":27.94722352,"weightedAverage":0.04157665}]'

    def open(self, url):
        self.received_urls.append(url)
        return self.ResultStub(self.returns)


class FileHandleSpy:
    def __init__(self, data=None):
        self.written = None
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def write(self, data):
        self.written = data

    def readlines(self):
        return self.data.splitlines()


class FilesSpy:
    def __init__(self):
        self.written_to = None
        self.appended_to = None
        self.files = {}

    def open(self, *args):
        path = args[0]
        if args[1] == "w":
            self.written_to = path
        if args[1] == "a":
            self.appended_to = path

        if path in self.files:
            return self.files[path]

        file = FileHandleSpy()
        self.files[self.written_to] = file
        return file

    def exists(self, path):
        return path in self.files


TEST_OUTPUT_DIR = "History/output/"


@pytest.fixture(autouse=True)
def requests():
    prev = sut.urlopen
    u = UrlibSpy()
    sut.urlopen = lambda *args: u.open(args[0])
    yield u
    sut.urlopen = prev


@pytest.fixture(autouse=True)
def files():
    prev = sut.os.path.exists
    f = FilesSpy()
    sut.open = lambda *args: f.open(*args)
    sut.os.path.exists = lambda *args: f.exists(args[0])
    yield f
    sut.os.path.exists = prev
    del sut.open


@pytest.fixture
def history():
    return make_history("https://poloniex.test/api", "BTC", ["ETH"], 1800, "2015/07/01")


def make_history(api, cash, coins, period, start):
    config = {"trading": {"api": api, "cash": cash, "coins": coins, "period": period, "start": start}}
    return PoloniexHistory(config, TEST_OUTPUT_DIR, 0)


def make_history_for(coins):
    return make_history("https://poloniex.test/api", "BTC", coins, 1800, "2015/07/01")


def url_for(coin):
    return f"https://poloniex.test/api?command=returnChartData&currencyPair=BTC_{coin}&period=1800&start=1435701600"


def output_of(coin):
    return TEST_OUTPUT_DIR + f"BTC_{coin}.csv"


def json_from(date, high, low, open, close, volume, quoteVolume, weightedAverage):
    return f'{{"date":{date},"high":{high},"low":{low},"open":{open},"close":{close},"volume":{volume},"quoteVolume":{quoteVolume},"weightedAverage":{weightedAverage}}}'


def return_string_of(*params):
    rt = "["
    rt += ",".join([json_from(*p) for p in params])
    return rt + "]"


def csv_from(date, high, low, open, close, volume, quoteVolume, weightedAverage):
    return f'{date},{open},{high},{low},{close},{volume},{quoteVolume},{weightedAverage}'


def csv_string_of(with_header, *params):
    rt = ""
    if with_header:
        rt = "timestamp,open,high,low,close,volume,quoteVolume,weightedAverage\n"
    rt += "\n".join([csv_from(*p) for p in params])
    return rt


def make_file(data):
    return FileHandleSpy(data)


def row_of(time):
    return time, 0, 0, 0, 0, 0, 0, 0


@pytest.mark.parametrize("test_config,expected", [
    (("https://poloniex.test/api", "BTC", ["ETH"], 1800, "2015/07/01"),
     "https://poloniex.test/api?command=returnChartData&currencyPair=BTC_ETH&period=1800&start=1435701600"),
    (("https://poloniex.test/api/v2", "ETH", ["BTC"], 60, "2016/01/01"),
     "https://poloniex.test/api/v2?command=returnChartData&currencyPair=ETH_BTC&period=60&start=1451602800"),
])
def test_that_queries_coin_from_start_when_file_does_not_exist(test_config, expected, requests):
    make_history(*test_config).update()
    assert requests.received_urls[0] == expected


def test_that_requests_are_dispatched_for_all_coins(requests):
    make_history_for(["ETH", "LTC"]).update()
    assert requests.received_urls[0] == url_for("ETH")
    assert requests.received_urls[1] == url_for("LTC")


def test_that_a_new_csv_is_written_when_it_does_not_exist_yet(history, requests, files):
    history.update()
    assert files.written_to == output_of("ETH")


def test_that_a_new_csv_writes_header_and_returned_data(history, requests, files):
    requests.returns = return_string_of(
        (1435701600, 0.04164207, 0.04145529, 0.04155495, 0.04159499, 1.16195203, 27.94722352, 0.04157665),
        (1435703400, 0.04159809, 0.04151278, 0.04159802, 0.04151278, 0.28771422, 6.91684214, 0.04159618))
    history.update()
    assert files.files[files.written_to].written == csv_string_of(True,
                                                                  (1435701600, 0.04164207, 0.04145529, 0.04155495,
                                                                   0.04159499, 1.16195203, 27.94722352, 0.04157665),
                                                                  (1435703400, 0.04159809, 0.04151278, 0.04159802,
                                                                   0.04151278, 0.28771422, 6.91684214, 0.04159618))


def test_if_data_already_exists_query_from_last_data_point(history, requests, files):
    files.files[output_of("ETH")] = make_file(csv_string_of(True, row_of(1435701600), row_of(1435703400)))
    history.update()
    assert requests.received_urls[0] == \
           "https://poloniex.test/api?command=returnChartData&currencyPair=BTC_ETH&period=1800&start=1435705200"


def test_if_data_already_exists_append_to_existing_file(history, files):
    files.files[output_of("ETH")] = make_file(csv_string_of(True, row_of(1435701600), row_of(1435703400)))
    history.update()
    assert files.appended_to == output_of("ETH")


def test_if_data_already_exists_append_new_data_to_file_without_header(history, files, requests):
    files.files[output_of("ETH")] = make_file(csv_string_of(True, row_of(1435701600), row_of(1435703400)))
    requests.returns = return_string_of(row_of(1435705200))
    history.update()
    assert files.files[files.appended_to].written == csv_string_of(False, row_of(1435705200))
