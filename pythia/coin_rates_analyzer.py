import sys

from pythia.core.streams.shape_shift_rates import ShapeShiftRates, analyze

if __name__ == "__main__":
    path = "../data/recordings/2018-02-28-shapeshift-exchange-records.json" if len(sys.argv) == 1 else sys.argv[1]
    with open(path, 'r') as stream:
        print(str(analyze(ShapeShiftRates(stream))))
