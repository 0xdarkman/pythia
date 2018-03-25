import pytest
from decimal import Decimal

from pythia.core.streams.shape_shift_rates import analyze, CoinExchangeReport
from pythia.tests.crypto_doubles import RecordsStub, RatesStub, entry


@pytest.fixture
def rates():
    s = RecordsStub()
    yield RatesStub(s)
    s.close()


def test_no_exchange_data(rates):
    assert str(analyze(rates)) == ""


def test_one_record_single_coin_exchange(rates):
    rates.add_record(entry("BTC_ETH", "1.267")).finish()
    assert str(analyze(rates)) == " EXCHANGE |   MEAN   |    SD    |  MEDIAN  |   MIN   |   MAX   |   DIF   \n" \
                                  "-------------------------------------------------------------------------\n" \
                                  " BTC_ETH  |     1.267|       0.0|     1.267|    1.267|    1.267|    0.000\n"


def test_different_value_length(rates):
    rates.add_record(entry("BTCD_NTCD", "1.2")).finish()
    assert str(analyze(rates)) == " EXCHANGE |   MEAN   |    SD    |  MEDIAN  |   MIN   |   MAX   |   DIF   \n" \
                                  "-------------------------------------------------------------------------\n" \
                                  " BTCD_NTCD|       1.2|       0.0|       1.2|      1.2|      1.2|      0.0\n"


def test_truncating_to_max_length(rates):
    rates.add_record(entry("BTCD_NTCD", "12.3456789101112")).finish()
    assert str(analyze(rates)) == " EXCHANGE |   MEAN   |    SD    |  MEDIAN  |   MIN   |   MAX   |   DIF   \n" \
                                  "-------------------------------------------------------------------------\n" \
                                  " BTCD_NTCD|12.3456789|       0.0|12.3456789|12.345678|12.345678|    0E-13\n"


def test_multiple_exchange_rates(rates):
    rates.add_record(entry("BTC_ETH", "1.267"), entry("LIC_GAME", "10.9")).finish()
    assert str(analyze(rates)) == " EXCHANGE |   MEAN   |    SD    |  MEDIAN  |   MIN   |   MAX   |   DIF   \n" \
                                  "-------------------------------------------------------------------------\n" \
                                  " BTC_ETH  |     1.267|       0.0|     1.267|    1.267|    1.267|    0.000\n" \
                                  " LIC_GAME |      10.9|       0.0|      10.9|     10.9|     10.9|      0.0\n"


def test_calculate_correct_statistics(rates):
    rates.add_record(entry("BTC_ETH", "10")) \
        .add_record(entry("BTC_ETH", "5")) \
        .add_record(entry("BTC_ETH", "12")).finish()
    assert str(analyze(rates)) == " EXCHANGE |   MEAN   |    SD    |  MEDIAN  |   MIN   |   MAX   |   DIF   \n" \
                                  "-------------------------------------------------------------------------\n" \
                                  " BTC_ETH  |         9|2.94392028|        10|        5|       12|        2\n"


def test_calculate_correct_statistics_of_multiple_entries(rates):
    rates.add_record(entry("BTC_ETH", "10"), entry("LIC_GAME", "2")) \
        .add_record(entry("BTC_ETH", "6"), entry("LIC_GAME", "1")).finish()
    assert str(analyze(rates)) == " EXCHANGE |   MEAN   |    SD    |  MEDIAN  |   MIN   |   MAX   |   DIF   \n" \
                                  "-------------------------------------------------------------------------\n" \
                                  " BTC_ETH  |         8|       2.0|        10|        6|       10|       -4\n" \
                                  " LIC_GAME |       1.5|       0.5|         2|        1|        2|       -1\n"


def test_report_csv_export():
    report = CoinExchangeReport()
    report.append("BTC_ETH", Decimal("1.267"), Decimal("0.23"), Decimal("1.5"), Decimal("0.3"), Decimal("1.998"), Decimal("-0.2"))
    report.append("LIC_GAME", Decimal("11.2"), Decimal("12.3456789"), Decimal("12.2"), Decimal("10"), Decimal("20.12"), Decimal("0.3"))
    assert report.to_csv() == "EXCHANGE,MEAN,SD,MEDIAN,MIN,MAX,DIF\n" \
                              "BTC_ETH,1.267,0.23,1.5,0.3,1.998,-0.2\n" \
                              "LIC_GAME,11.2,12.3456789,12.2,10,20.12,0.3\n"
