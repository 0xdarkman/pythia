import sys

from pythia.core.streams.rates_analytics import analyze
from pythia.core.streams.share_rates import Symbol, ShareRates

if __name__ == '__main__':
    path = "../data/recordings/shares/SPY.csv" if len(sys.argv) == 1 else sys.argv[0]
    with open(path) as stream:
        rates = ShareRates(Symbol("SPY", stream))
        print(analyze(rates))
