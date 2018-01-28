#old dynamic prog

# for price_idx, shares_idx in itertools.product(range(0, V.shape[0]), range(0, V.shape[1])):
#     if shares_idx > price_idx:
#         continue
#
#     v = V[price_idx, shares_idx]
#     terminal = price_idx == V.shape[0] - 1
#     if shares_idx == 0 and not terminal:
#         hold_value = gamma * V[price_idx + 1, shares_idx]
#         buy_value = -stock_data.data[shares_idx][0] + gamma * V[price_idx + 1, price_idx + 1]
#         V[price_idx, shares_idx] = max(hold_value, buy_value)
#     elif shares_idx > 0 and not terminal:
#         hold_value = gamma * V[price_idx + 1, shares_idx]
#         sell_value = stock_data.data[shares_idx][0] + gamma * V[price_idx + 1, 0]
#         V[price_idx, shares_idx] = max(hold_value, sell_value)
#     elif terminal:
#         if shares_idx == 0:
#             V[price_idx, shares_idx] = 0
#         else:
#             hold_value = 0
#             sell_value = stock_data.data[shares_idx][0]
#             V[price_idx, shares_idx] = max(hold_value, sell_value)
#
#     delta = max(delta, abs(v - V[price_idx, shares_idx]))

# for s1 in range(1, number_of_stocks):
#     a1 = np.argmax(V[s1, 0:s1 + 1])
#     dir = 0
#     if a1 > a:
#         dir = 1
#     elif a1 < a:
#         dir = -1
#
#     a = a1
#     _, r, done, _ = env.step(dir)
#
#     rewards.append(r)
#     if done:
#         break