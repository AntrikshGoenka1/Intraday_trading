import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Nifty 50 Historical Data.csv')
df['Date']  = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['Price'] = df['Price'].str.replace(',', '').astype(float)
df['Open']  = df['Open'].str.replace(',', '').astype(float)
df['High']  = df['High'].str.replace(',', '').astype(float)
df['Low']   = df['Low'].str.replace(',', '').astype(float)

def parse_volume(v):
    if pd.isna(v):
        return np.nan
    v = str(v).strip().replace(',', '')
    if v.upper().endswith('M'):
        return float(v[:-1]) * 1_000_000
    elif v.upper().endswith('B'):
        return float(v[:-1]) * 1_000_000_000
    elif v.upper().endswith('K'):
        return float(v[:-1]) * 1_000
    else:
        try:   return float(v)
        except: return np.nan

df['Vol.'] = df['Vol.'].apply(parse_volume)
df = df.sort_values('Date').reset_index(drop=True)

start_date = df['Date'].min()
df['Years_Elapsed'] = (df['Date'] - start_date).dt.days / 365.25
P_0  = df['Price'].iloc[0]
cagr = 0.13
df['Trend_Line'] = P_0 * ((1 + cagr) ** df['Years_Elapsed'])

df['Log_Price'] = np.log(df['Price'])
df['Log_Trend'] = np.log(df['Trend_Line'])
df['Residual']  = df['Log_Price'] - df['Log_Trend']

BB_WINDOW       = 30            
BB_ENTRY_MULT   = 0.3           
BB_EXIT_MULT    = 0.3           
EMA_FAST        = 5
EMA_SLOW        = 12
ATR_PERIOD      = 14
ATR_MULTIPLIER  = 1.5           
REGIME_SMA      = 50            
BEAR_ENTRY_MULT = 1.0           
COOLDOWN_DAYS   = 0             

df['Rolling_Mean'] = df['Residual'].rolling(window=BB_WINDOW).mean()
df['Rolling_Std']  = df['Residual'].rolling(window=BB_WINDOW).std()
df['Lower_Band']   = df['Rolling_Mean'] - BB_ENTRY_MULT * df['Rolling_Std']
df['Upper_Band']   = df['Rolling_Mean'] + BB_EXIT_MULT  * df['Rolling_Std']

df['Fast_EMA'] = df['Price'].ewm(span=EMA_FAST, adjust=False).mean()
df['Slow_EMA'] = df['Price'].ewm(span=EMA_SLOW, adjust=False).mean()
df['Prev_Fast_EMA'] = df['Fast_EMA'].shift(1)
df['Prev_Slow_EMA'] = df['Slow_EMA'].shift(1)

df['Prev_Close'] = df['Price'].shift(1)
df['TR'] = np.maximum(
    df['High'] - df['Low'],
    np.maximum(
        (df['High'] - df['Prev_Close']).abs(),
        (df['Low']  - df['Prev_Close']).abs()
    )
)
df['ATR'] = df['TR'].rolling(window=ATR_PERIOD).mean()

df['SMA_Regime']  = df['Price'].rolling(window=REGIME_SMA).mean()
df['Bull_Regime'] = df['Price'] > df['SMA_Regime']
df['Bear_Lower_Band'] = df['Rolling_Mean'] - BEAR_ENTRY_MULT * df['Rolling_Std']

VOL_MA_WINDOW = 20
df['Vol_MA'] = df['Vol.'].rolling(window=VOL_MA_WINDOW).mean()

delta = df['Price'].diff()
gain  = delta.where(delta > 0, 0.0)
loss  = (-delta).where(delta < 0, 0.0)
avg_gain = gain.ewm(com=13, min_periods=14, adjust=False).mean()
avg_loss = loss.ewm(com=13, min_periods=14, adjust=False).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

df['Z_Score'] = (df['Residual'] - df['Rolling_Mean']) / df['Rolling_Std']

signals         = []
current_pos     = 0
oversold_setup  = False
trailing_stop   = -np.inf
entry_price     = 0.0
entry_date      = None
entry_mode      = None          
cooldown_until  = None
trade_log = []

for i, row in df.iterrows():
    date = row['Date']
    warmup_ready = (
        not pd.isna(row['Rolling_Mean']) and
        not pd.isna(row['ATR']) and
        not pd.isna(row['SMA_Regime']) and
        not pd.isna(row['Prev_Fast_EMA'])
    )
    if not warmup_ready:
        signals.append(0)
        continue

    is_bull = row['Bull_Regime']
    active_lower = row['Lower_Band'] if is_bull else row['Bear_Lower_Band']

    if current_pos == 0:
        if cooldown_until is not None and date < cooldown_until:
            signals.append(0)
            continue

        ema_bullish     = row['Fast_EMA'] > row['Slow_EMA']
        fresh_golden_x  = (row['Prev_Fast_EMA'] <= row['Prev_Slow_EMA']) and ema_bullish

        if row['Residual'] < active_lower:
            oversold_setup = True
            
        mode_a_fire = oversold_setup and ema_bullish
        mode_b_fire = (fresh_golden_x and is_bull and not oversold_setup)

        if mode_a_fire or mode_b_fire:
            signals.append(1)
            current_pos     = 1
            oversold_setup  = False
            entry_price     = row['Price']
            entry_date      = date
            trailing_stop   = row['Price'] - ATR_MULTIPLIER * row['ATR']
            entry_mode      = 'A' if mode_a_fire else 'B'
        else:
            signals.append(0)

    elif current_pos == 1:
        oversold_setup = False
        new_stop = row['Price'] - ATR_MULTIPLIER * row['ATR']
        if new_stop > trailing_stop:
            trailing_stop = new_stop

        hit_trailing_stop = row['Price'] <= trailing_stop
        ema_death_cross   = row['Fast_EMA'] < row['Slow_EMA']
        
        z_score_exit = False
        if entry_mode == 'A':
            z_score_exit = row['Residual'] >= row['Upper_Band']
            
        if hit_trailing_stop or ema_death_cross or z_score_exit:
            signals.append(0)
            pnl_pct = ((row['Price'] - entry_price) / entry_price) * 100
            holding_days = (date - entry_date).days
            
            exit_reason = []
            if hit_trailing_stop: exit_reason.append('ATR_STOP')
            if ema_death_cross:   exit_reason.append('EMA_CROSS')
            if z_score_exit:      exit_reason.append('Z_EXIT')
            
            trade_log.append({
                'Entry_Date':   entry_date,
                'Exit_Date':    date,
                'Entry_Price':  round(entry_price, 2),
                'Exit_Price':   round(row['Price'], 2),
                'PnL_Pct':      round(pnl_pct, 2),
                'Holding_Days': holding_days,
                'Exit_Reason':  '+'.join(exit_reason),
                'Entry_Mode':   entry_mode,
                'Regime':       'BULL' if is_bull else 'BEAR'
            })
            
            current_pos    = 0
            trailing_stop  = -np.inf
            entry_price    = 0.0
            entry_mode     = None
            cooldown_until = date + pd.Timedelta(days=COOLDOWN_DAYS)
        else:
            signals.append(1)

df['Signal'] = signals

trades_df = pd.DataFrame(trade_log)
if len(trades_df) > 0:
    trades_df.to_csv('Nifty50_Trade_Log_v4.csv', index=False)

df.to_csv('Nifty50_Enhanced_Strategy_Signals_v4.csv', index=False)