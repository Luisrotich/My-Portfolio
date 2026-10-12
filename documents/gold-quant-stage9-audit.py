# Gold Quant Project - Version 15f
# Stage 9 final transaction-ledger audit.
# No model changes, no tuning, and no test-period use.

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 260)

DATA_CSV = "gold_ml_dataset.csv"
RAW_CSV = "gold_daily_full_v2.csv"
PREDICT_DAYS = 5
COST_PER_TRADE = 0.0002
THRESHOLD = 0.5
RANDOM_STATE = 42
TOL = 1e-10

NOM_TRAIN_END = "2018-12-31"
NOM_VAL_START = "2019-01-01"
NOM_VAL_END = "2022-12-31"
NOM_TEST_START = "2023-01-01"

for filename in (DATA_CSV, RAW_CSV):
    if not os.path.exists(filename):
        raise SystemExit(f"{filename} not found.")

data = pd.read_csv(DATA_CSV, index_col="Date", parse_dates=True).sort_index()
raw = pd.read_csv(RAW_CSV, index_col="Date", parse_dates=True).sort_index()
feature_cols = [column for column in data.columns if column not in ("target", "forward_return")]
raw_dates = raw.index
raw_pos = {date: position for position, date in enumerate(raw_dates)}


def label_endpoint(date):
    position = raw_pos.get(date)
    if position is None or position + PREDICT_DAYS >= len(raw_dates):
        return None
    return raw_dates[position + PREDICT_DAYS]


nom_train = data.loc[:NOM_TRAIN_END]
nom_val = data.loc[NOM_VAL_START:NOM_VAL_END]
nom_test = data.loc[NOM_TEST_START:]


def last_clean(segment, next_first):
    for date in reversed(segment.index.tolist()):
        endpoint = label_endpoint(date)
        if endpoint is not None and endpoint < next_first:
            return date
    raise SystemExit("No clean cut found.")


cut_train = last_clean(nom_train, nom_val.index[0])
cut_val = last_clean(nom_val, nom_test.index[0])
train = nom_train.loc[:cut_train]
val = nom_val.loc[:cut_val]

for name, left, right in (
    ("train->val", train.index[-1], val.index[0]),
    ("val->test", val.index[-1], nom_test.index[0]),
):
    endpoint = label_endpoint(left)
    assert endpoint < right, f"Boundary {name} failed."
print(f"Splits verified. Train {len(train)} rows, Val {len(val)} rows.")

X_train = train[feature_cols].to_numpy(float)
y_train = train["target"].to_numpy(int)
X_val = val[feature_cols].to_numpy(float)
y_val = val["target"].to_numpy(int)
scaler = StandardScaler().fit(X_train)
model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
model.fit(scaler.transform(X_train), y_train)
proba_val = model.predict_proba(scaler.transform(X_val))[:, 1]
signal = pd.Series((proba_val > THRESHOLD).astype(int), index=val.index)

open_px = raw["Open"]
c = COST_PER_TRADE
capital = 1.0
trades = []
signal_dates = list(signal.index)
i = 0
while i < len(signal_dates):
    date = signal_dates[i]
    if signal.loc[date] != 1:
        i += 1
        continue
    position = raw_pos.get(date)
    if position is None or position + 1 + PREDICT_DAYS >= len(raw_dates):
        i += 1
        continue

    entry_date = raw_dates[position + 1]
    exit_date = raw_dates[position + 1 + PREDICT_DAYS]
    entry_px = float(open_px.iloc[position + 1])
    exit_px = float(open_px.iloc[position + 1 + PREDICT_DAYS])

    cap_before = capital
    gross_notional_e = cap_before
    entry_cost = cap_before * c
    notional_invested = cap_before * (1.0 - c)
    units = notional_invested / entry_px
    gross_proceeds = units * exit_px
    exit_cost = gross_proceeds * c
    capital_after = gross_proceeds * (1.0 - c)
    gross_ret = exit_px / entry_px - 1.0
    net_ret = capital_after / cap_before - 1.0

    trades.append({
        "signal_date": date,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "entry_px": entry_px,
        "exit_px": exit_px,
        "cap_before": cap_before,
        "gross_notional_entry": gross_notional_e,
        "entry_cost": entry_cost,
        "notional_invested": notional_invested,
        "units": units,
        "gross_proceeds": gross_proceeds,
        "exit_cost": exit_cost,
        "capital_after": capital_after,
        "gross_ret": gross_ret,
        "net_ret": net_ret,
    })
    capital = capital_after
    j = i + 1
    while j < len(signal_dates) and signal_dates[j] <= exit_date:
        j += 1
    i = j

tdf = pd.DataFrame(trades)
n = len(tdf)
print(f"\nTrades: {n}   Final capital: {capital:.9f}")

ok1 = np.allclose(tdf["units"], tdf["notional_invested"] / tdf["entry_px"], atol=TOL)
ok2 = np.allclose(tdf["gross_proceeds"], tdf["units"] * tdf["exit_px"], atol=TOL)
ok3 = np.allclose(tdf["entry_cost"], tdf["cap_before"] * c, atol=TOL)
ok4 = np.allclose(tdf["exit_cost"], tdf["gross_proceeds"] * c, atol=TOL)
ok5 = np.allclose(tdf["capital_after"], tdf["gross_proceeds"] * (1 - c), atol=TOL)
ok6 = np.allclose(tdf["net_ret"], tdf["capital_after"] / tdf["cap_before"] - 1.0, atol=TOL)
ok7 = np.allclose(
    tdf["capital_after"],
    tdf["cap_before"] * (1 + tdf["gross_ret"]) * (1 - c) ** 2,
    atol=TOL,
)
ok8 = (
    np.allclose(
        tdf["cap_before"].iloc[1:].to_numpy(),
        tdf["capital_after"].iloc[:-1].to_numpy(),
        atol=TOL,
    )
    if n > 1
    else True
)

checks = {
    "units = notional_invested / entry_px": ok1,
    "gross_proceeds = units * exit_px": ok2,
    "entry_cost = cap_before * c": ok3,
    "exit_cost = gross_proceeds * c": ok4,
    "capital_after = gross_proceeds*(1-c)": ok5,
    "net_ret = capital_after/cap_before-1": ok6,
    "capital_after = cap_before*(1+gross)*(1-c)^2": ok7,
    "capital chains: cap_after[k] = cap_before[k+1]": ok8,
}
print("\nLEDGER ASSERTIONS (every row)")
for name, passed in checks.items():
    print(f"  {'PASS' if passed else 'FAIL'}  {name}")
assert all(checks.values()), "Ledger assertion failed — stopping."

entry_total = tdf["entry_cost"].sum()
exit_total = tdf["exit_cost"].sum()
total_cost = entry_total + exit_total
recomputed_entry = (tdf["cap_before"] * c).sum()
recomputed_exit = (tdf["units"] * tdf["exit_px"] * c).sum()
recomputed_total = recomputed_entry + recomputed_exit
print(f"\nEntry cost total: {entry_total:.9f}")
print(f"Exit cost total:  {exit_total:.9f}")
print(f"Total cost:       {total_cost:.9f}")
assert abs(total_cost - recomputed_total) < TOL, "Cost recomputation mismatch."

val_days = [date for date in raw_dates if val.index[0] <= date <= val.index[-1]]
pos_map = {}
for trade in trades:
    entry_position = raw_pos[trade["entry_date"]]
    exit_position = raw_pos[trade["exit_date"]]
    for position in range(entry_position, exit_position):
        pos_map[raw_dates[position]] = trade

cash, units = 1.0, 0.0
prev_in, prev_trade = False, None
daily = {}
violations = 0
for date in val_days:
    if date in pos_map:
        trade = pos_map[date]
        if not prev_in or prev_trade is not trade:
            notional = cash * (1.0 - c)
            units = notional / trade["entry_px"]
            cash = 0.0
        equity = cash + units * float(raw.loc[date, "Close"])
        expected = units * float(raw.loc[date, "Close"])
        if abs(equity - expected) > TOL or abs(cash) > TOL:
            violations += 1
        daily[date] = equity
        prev_in, prev_trade = True, trade
    else:
        if prev_in:
            proceeds = units * float(raw.loc[date, "Open"])
            cash = proceeds * (1.0 - c)
            units = 0.0
        daily[date] = cash
        prev_in, prev_trade = False, None

daily = pd.Series(daily)
print(f"\nInvested-day violations: {violations}")
assert violations == 0, "Daily equity assertion failed."

prod = float(np.prod(1.0 + tdf["net_ret"]))
final_daily = float(daily.iloc[-1])
diff = abs(prod - final_daily)
print(f"Product of (1+net_ret): {prod:.9f}")
print(f"Final daily equity:     {final_daily:.9f}")
print(f"Reconciliation diff:    {diff:.3e}")
assert diff < 1e-9, "Reconciliation failed."

cols = [
    "entry_date", "exit_date", "entry_px", "exit_px", "cap_before",
    "notional_invested", "units", "gross_proceeds", "entry_cost",
    "exit_cost", "gross_ret", "net_ret",
]
print("\nLEDGER — first 3 and last 3 trades")
print(tdf.head(3)[cols].to_string(index=False))
print("...")
print(tdf.tail(3)[cols].to_string(index=False))

pos_a = raw_pos[val.index[0]]
pos_b = raw_pos[val.index[-1]]
bh_entry = float(open_px.iloc[pos_a + 1])
bh_exit = float(open_px.iloc[pos_b + 1]) if pos_b + 1 < len(raw_dates) else float(open_px.iloc[pos_b])
bh_ret = bh_exit / bh_entry - 1.0 - 2 * c
print("\nPRIMARY BENCHMARK")
print(f"Strategy total return: {capital - 1:+.4%} ({n} trades)")
print(f"B&H total return:      {bh_ret:+.4%}")
print(f"Difference:            {capital - 1 - bh_ret:+.4%}")
print("Test period used: NO")