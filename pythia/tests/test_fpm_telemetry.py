import pytest

import pythia.core.remote.telemetry as sut
from pythia.core.remote.telemetry import Telemetry

CHART_KEYS = ("date", "high", "low", "open", "close")


def uniform_chart(value):
    return {k: value for k in CHART_KEYS}


def uniform_csv(value):
    return "{},{},{},{}\n".format(*(value,) * 4)


class FileStub:
    def __init__(self):
        self.text = uniform_csv(0)

    def set_text(self, text):
        self.text = text

    def read(self):
        return self.text


class FileSpy(FileStub):
    def __init__(self):
        super().__init__()
        self.mode = None
        self.recorded_modes = []
        self.path = None
        self.received_text = ""

    def write(self, text):
        self.received_text += text

    def writelines(self, lines):
        self.received_text = "".join(lines)


class FileStreamSpy:
    def __init__(self):
        self.file = FileSpy()

    def __enter__(self):
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class FileOpenSpy:
    def __init__(self):
        self.file_stream = FileStreamSpy()

    def __call__(self, *args, **kwargs):
        self.file_stream.file.path = args[0]
        self.file_stream.file.mode = args[1]
        self.file_stream.file.recorded_modes.append(args[1])
        if args[1] == "w":
            self.file_stream.file.received_text = ""
        return self.file_stream


class IsFileStub:
    def __init__(self):
        self.isfile = True

    def set(self, value):
        self.isfile = value

    def __call__(self, *args, **kwargs):
        return self.isfile


class ListDirStub:
    def __init__(self):
        self.fs = []

    def files(self, *fs):
        self.fs = list(fs)

    def __call__(self, _):
        return self.fs


class RemoveSpy:
    def __init__(self):
        self.received_files = []

    def __call__(self, file):
        self.received_files.append(file)


@pytest.fixture(autouse=True)
def file_open():
    prev_open = sut.open_file
    sut.open_file = FileOpenSpy()
    yield sut.open_file
    sut.open_file = prev_open


@pytest.fixture
def file(file_open):
    return file_open.file_stream.file


@pytest.fixture(autouse=True)
def isfile():
    prev_isfile = sut.path.isfile
    sut.path.isfile = IsFileStub()
    yield sut.path.isfile
    sut.path.isfile = prev_isfile


@pytest.fixture(autouse=True)
def listdir():
    prev_listdir = sut.listdir
    sut.listdir = ListDirStub()
    yield sut.listdir
    sut.listdir = prev_listdir


@pytest.fixture(autouse=True)
def remove():
    prev_remove = sut.remove
    sut.remove = RemoveSpy()
    yield sut.remove
    sut.remove = prev_remove


@pytest.fixture
def folder():
    return "folder/"


@pytest.fixture
def limit():
    return 100


@pytest.fixture
def telemetry(folder, limit):
    return Telemetry(folder, limit)


@pytest.mark.parametrize("pair,date,close,high,low", [("CASH_SYMBOL", 1551434400, 2, 3, 1),
                                                      ("SYM0_SYM1", 1551436000, 3, 5, 2)])
def test_write_chart_data_the_first_time_creates_respective_file(telemetry, folder, file, pair, date, close, high, low):
    telemetry.write_chart({f"{pair}": {"date": date, "close": close, "high": high, "low": low}})
    assert file.mode == "a+"
    assert file.path == f"{folder}chart_{pair.lower()}.csv"
    assert file.received_text == f"{date},{close},{high},{low}\n"


def test_append_chart_data_to_existing_file(telemetry, file):
    telemetry.write_chart({"CASH_SYMBOL": uniform_chart(1)})
    telemetry.write_chart({"CASH_SYMBOL": uniform_chart(2)})
    assert file.received_text == "1,1,1,1\n2,2,2,2\n"


@pytest.mark.parametrize("default", [10, 100])
def test_find_last_chart_ts_returns_default_value_when_file_does_not_exist(telemetry, isfile, default):
    isfile.set(False)
    assert telemetry.find_last_chart_ts("CASH_SYMBOL", default) == default


def test_find_last_chart_ts_queries_the_according_chart_file_if_it_exists(telemetry, file, isfile, folder):
    isfile.set(True)
    telemetry.find_last_chart_ts("CASH_SYMBOL", 10)
    assert file.mode == "r"
    assert file.path == f"{folder}chart_cash_symbol.csv"


def test_find_last_chart_ts_returns_timestamp_of_last_chart_value_in_file(telemetry, file):
    file.set_text("1,1,1,1\n2,2,2,2\n")
    assert telemetry.find_last_chart_ts("CASH_SYMBOL", 0) == 2


def test_rotate_file_if_limit_is_reached(telemetry, file, limit):
    file.set_text("".join(uniform_csv(i) for i in range(0, limit)))
    telemetry.write_chart({"CASH_SYMBOL": uniform_chart(limit)})
    assert file.recorded_modes[-2] == "w"
    assert file.received_text == "".join(uniform_csv(i) for i in range(1, limit)) + uniform_csv(limit)


def test_reset_clears_all_files(telemetry, listdir, remove, folder):
    listdir.files("A.csv", "B.csv")
    telemetry.reset()
    assert remove.received_files == [folder + "A.csv", folder + "B.csv"]
