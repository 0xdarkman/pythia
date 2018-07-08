from pythia.core.streams.data_frame_stream import Exchange


def exchange(close):
    return Exchange(close, close, close, close, close)


def make_raw_state(rates):
    return {"token": "SYMA", "balance": 1, "rates": rates}