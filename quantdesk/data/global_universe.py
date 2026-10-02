"""The global markets the desk watches, and how each is *expected* to lean on Indian equities.

`india` is the hypothesised sign of the effect of a rise in that market on NIFTY/BANKNIFTY
(+1: moves with India, e.g. US equities; -1: against, e.g. crude, the dollar, US yields, US VIX;
0: no prior). The edge research tests every one of these on real data; the brain only puts weight
on links that survive. `intraday`: trades during Indian hours, so its live moves can be watched.
"""
GLOBAL = {
    "ES":     {"yahoo": "ES=F",      "name": "S&P 500 futures",     "region": "US",       "kind": "equity", "india": 1, "intraday": True},
    "NQ":     {"yahoo": "NQ=F",      "name": "Nasdaq 100 futures",  "region": "US",       "kind": "equity", "india": 1, "intraday": True},
    "SPX":    {"yahoo": "^GSPC",     "name": "S&P 500",             "region": "US",       "kind": "equity", "india": 1, "intraday": False},
    "NASDAQ": {"yahoo": "^IXIC",     "name": "Nasdaq Composite",    "region": "US",       "kind": "equity", "india": 1, "intraday": False},
    "DJI":    {"yahoo": "^DJI",      "name": "Dow Jones",           "region": "US",       "kind": "equity", "india": 1, "intraday": False},
    "N225":   {"yahoo": "^N225",     "name": "Nikkei 225",          "region": "Asia",     "kind": "equity", "india": 1, "intraday": True},
    "HSI":    {"yahoo": "^HSI",      "name": "Hang Seng",           "region": "Asia",     "kind": "equity", "india": 1, "intraday": True},
    "KOSPI":  {"yahoo": "^KS11",     "name": "Kospi",               "region": "Asia",     "kind": "equity", "india": 1, "intraday": True},
    "SSE":    {"yahoo": "000001.SS", "name": "Shanghai Composite",  "region": "Asia",     "kind": "equity", "india": 1, "intraday": True},
    "STOXX":  {"yahoo": "^STOXX50E", "name": "Euro Stoxx 50",       "region": "Europe",   "kind": "equity", "india": 1, "intraday": True},
    "DAX":    {"yahoo": "^GDAXI",    "name": "DAX",                 "region": "Europe",   "kind": "equity", "india": 1, "intraday": True},
    "FTSE":   {"yahoo": "^FTSE",     "name": "FTSE 100",            "region": "Europe",   "kind": "equity", "india": 1, "intraday": True},
    "USVIX":  {"yahoo": "^VIX",      "name": "CBOE VIX",            "region": "US",       "kind": "vol",    "india": -1, "intraday": False},
    "DXY":    {"yahoo": "DX-Y.NYB",  "name": "US dollar index",     "region": "FX",       "kind": "fx",     "india": -1, "intraday": True},
    "USDINR": {"yahoo": "INR=X",     "name": "USD/INR",             "region": "FX",       "kind": "fx",     "india": -1, "intraday": True},
    "BRENT":  {"yahoo": "BZ=F",      "name": "Brent crude",         "region": "Commodities", "kind": "commodity", "india": -1, "intraday": True},
    "GOLD":   {"yahoo": "GC=F",      "name": "Gold",                "region": "Commodities", "kind": "commodity", "india": 0, "intraday": True},
    "UST10":  {"yahoo": "^TNX",      "name": "US 10-year yield",    "region": "Rates",    "kind": "rates",  "india": -1, "intraday": False},
}
