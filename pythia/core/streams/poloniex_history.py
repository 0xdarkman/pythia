import json
import os
import time
from datetime import datetime
from urllib.request import urlopen
from urllib.parse import urlencode

HEADER = "timestamp,open,high,low,close,volume,quoteVolume,weightedAverage\n"


class PoloniexHistory:
    def __init__(self, config, output_path, sleep=5):
        self.trading_cfg = config["trading"]
        self.requests = [{"coin": coin,
                          "url": f"{self.trading_cfg['api']}?",
                          "args": self._build_url_arguments(coin)} for coin in self.trading_cfg["coins"]]
        self.output_path = output_path
        self.sleep = sleep

    def _build_url_arguments(self, coin):
        args = {"command": "returnChartData",
                "currencyPair": f"{self.trading_cfg['cash']}_{coin}",
                "period": self.trading_cfg["period"],
                "start": str(int(time.mktime(datetime.strptime(self.trading_cfg['start'], "%Y/%m/%d").timetuple())))}
        return args

    def update(self):
        for r in self.requests:
            path = self._make_path_for(r["coin"])
            exists = os.path.exists(path)
            if exists:
                self._update_start_time(path, r)

            rates = json.loads(urlopen(r["url"] + urlencode(r["args"])).read().decode(encoding='UTF-8'))
            if exists:
                self._append_csv(path, rates)
            else:
                self._create_csv(path, rates)

            time.sleep(self.sleep)

    def _make_path_for(self, coin):
        return os.path.join(self.output_path, f"{self.trading_cfg['cash']}_{coin}.csv")

    def _update_start_time(self, path, r):
        with open(path, 'r') as f:
            last_ts = f.readlines()[-1].split(',')[0]
            r["args"]["start"] = int(last_ts) + self.trading_cfg["period"]

    def _append_csv(self, path, rates):
        with open(path, "a") as f:
            f.write("\n".join([self._rate_to_csv(r) for r in rates]))

    def _create_csv(self, path, rates):
        with open(path, "w") as f:
            csv = HEADER
            csv += "\n".join([self._rate_to_csv(r) for r in rates])
            f.write(csv)

    @staticmethod
    def _rate_to_csv(rate):
        return f'{rate["date"]},{rate["open"]},{rate["high"]},{rate["low"]},{rate["close"]},{rate["volume"]},{rate["quoteVolume"]},{rate["weightedAverage"]}'
