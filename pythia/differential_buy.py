import pandas as pd

data_file = pd.read_csv('data/currencies.csv')
prices = data_file.iloc[:, 3]

diff_span = 0.01
win_span = 0.1

depot = 10000
hold = 0
bought_price = 0

for idx, price in enumerate(prices):
    if idx < 1:
        continue

    if hold == 0:
        diff = (price - prices[idx-1]) / price
        if diff >= diff_span:
            hold = 1
            bought_price = price
            depot -= bought_price

    if hold == 1:
        diff = price - bought_price
        if diff >= win_span:
            depot += price
            hold = 0
            bought_price = 0

print(depot)
