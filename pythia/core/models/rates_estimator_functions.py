from tensorflow.python.feature_column.feature_column import numeric_column


def make_feature_columns(input_size):
    features = [numeric_column("token"), numeric_column("balance")]
    for i in range(input_size - 2):
        features.append(numeric_column("price_{}".format(i)))
    return features


def process_input_states(states):
    s = [list(e) for e in zip(*states)]
    f = {"token": s[0], "balance": s[1]}
    for i in range(len(s) - 2):
        f["price_{}".format(i)] = s[i + 2]
    return f


def process_reward_targets(targets):
    return [e for target in targets for e in target]