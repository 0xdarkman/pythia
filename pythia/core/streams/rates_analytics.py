from decimal import Decimal
from functools import reduce
from math import sqrt

ANALYSIS_STR_HEADER = " EXCHANGE |   MEAN   |    SD    |  MEDIAN  |   MIN   |   MAX   |   DIF   \n" \
                      "-------------------------------------------------------------------------\n"
ANALYSIS_CSV_HEADER = "EXCHANGE,MEAN,SD,MEDIAN,MIN,MAX,DIF\n"


class RatesExchangeReport:
    def __init__(self):
        self.records = list()

    def append(self, *columns):
        self.records.append(columns)

    def __str__(self):
        if len(self.records) == 0:
            return ""

        def print_records(report, columns):
            exchange, mean, sd, median, min, max, dif = columns
            return report + " {:<9}|{:>10.10}|{:>10.10}|{:>10.10}|{:>9.9}|{:>9.9}|{:>9.9}\n" \
                .format(exchange, str(mean), str(sd), str(median), str(min), str(max), str(dif))

        return reduce(print_records, self.records, ANALYSIS_STR_HEADER)

    def to_csv(self):
        if len(self.records) == 0:
            return ""

        def print_records(report, columns):
            return report + ",".join(map(str, columns)) + "\n"

        return reduce(print_records, self.records, ANALYSIS_CSV_HEADER)


def analyze(rates):
    def fold_rates(all, pairs):
        for key in pairs:
            if key not in all:
                all[key] = list()
            all[key].append(pairs[key].rate)
        return all

    exchanges = reduce(fold_rates, rates, dict())

    report = RatesExchangeReport()
    for k in exchanges:
        r = exchanges[k]
        dif = r[-1] - r[0]
        r = sorted(r)
        s = sum(r)
        l = len(r)
        mean = s / l
        median = r[(l // 2)]
        sd = sqrt(reduce(lambda t, x: t + (x - mean) ** 2, r, Decimal(0)) / l)
        report.append(k, mean, sd, median, min(r), max(r), dif)

    return report
