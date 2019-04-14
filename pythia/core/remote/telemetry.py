import os

from os import path

open_file = open


def _to_csv(d, *ks):
    return ",".join([str(d[k]) for k in ks])


class Telemetry:
    def __init__(self, folder, limit):
        self._folder = folder
        self._limit = limit

    def write_chart(self, data):
        pair = next(iter(data))
        file = self._get_file_for_pair(pair)
        if path.isfile(file):
            self._rotate_lines_in_file(file)

        with open_file(file, "a+") as f:
            f.write(_to_csv(data[pair], "date", "close", "high", "low") + "\n")

    def _rotate_lines_in_file(self, file):
        with open_file(file, "r") as f:
            lines = f.read().strip().splitlines()
        if len(lines) >= self._limit:
            with open_file(file, "w") as f:
                f.writelines(lines[1:])
                f.write(os.linesep)

    def _get_file_for_pair(self, pair):
        return path.join(self._folder, "chart_{}.csv".format(pair.lower()))

    def find_last_chart_ts(self, pair, default):
        file = self._get_file_for_pair(pair)
        if not path.isfile(file):
            return default

        with open_file(file, "r") as f:
            lines = f.read().strip().splitlines()
            return int(lines[-1].split(',')[0])
