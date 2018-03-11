import sys

import os

from pythia.core.streams.shape_shift_rates import ShapeShiftRates, rates_filter

if __name__ == "__main__":
    in_path = "../data/recordings/2018-02-28-shapeshift-exchange-records.json" if len(sys.argv) == 1 else sys.argv[1]
    out_path = "../data/recordings/filtered/2018-02-28-shapeshift-BTC_ETH.json" if len(sys.argv) == 1 else sys.argv[2]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(in_path, 'r') as in_stream, open(out_path, 'w') as out_stream:
        out_stream.write(rates_filter(ShapeShiftRates(in_stream), ["BTC_ETH", "ETH_BTC"]))
