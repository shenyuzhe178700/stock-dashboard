import numpy as np, pandas as pd, os, warnings, pickle
warnings.filterwarnings('ignore')
from config import OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Generate Mock Data ----
np.random.seed(42)
n_stocks, n_days = 100, 500
dates = pd.date_range(end='2026-06-20', periods=n_days, freq='B')
codes = [f'{i:06d}' for i in range(1, n_stocks+1)]
true_quality = np.random.normal(0, 1, n_stocks)
prices = np.random.uniform(5, 200, n_stocks)

records = []
for i, date in enumerate(dates):
    market_ret = np.random.normal(0.0003, 0.012)
    for j, code in enumerate(codes):
        stock_ret = market_ret + 0.001*true_quality[j] + np.random.normal(0, 0.02)
        prices[j] *= (1+stock_ret); prices[j] = max(prices[j], 1.0)
        vol = abs(stock_ret)
        high = prices[j]*(1+abs(np.random.normal(0,vol*0.3)))
        low = prices[j]*(1-abs(np.random.normal(0,vol*0.3)))
        open_p = low+np.random.random()*(high-low)
        records.append([date,code,open_p,high,low,prices[j],
                       np.random.lognormal(15,1.5),
                       np.random.lognormal(15,1.5)*prices[j],
                       np.random.uniform(0.1,10.0)])

daily_raw = pd.DataFrame(records, columns=['date','code','open','high','low','close','volume','amount','turnover_rate'])
daily_raw['date'] = pd.to_datetime(daily_raw['date'])

# ---- Build Factors ----
df = daily_raw.copy()
g = df.groupby('code')
df['ret_1m'] = g['close'].pct_change(20)
df['ret_3m'] = g['close'].pct_change(60)
df['ret_6m'] = g['close'].pct_change(120)
df['ret_12m'] = g['close'].pct_change(240)
df['vol_20d'] = g['close'].pct_change().rolling(20).std().values
df['vol_60d'] = g['close'].pct_change().rolling(60).std().values
df['max_drawdown_60d'] = g['close'].rolling(60).apply(lambda x: (x/x.cummax()-1).min() if len(x)>0 else 0).values
df['volume_ratio_5d'] = g['volume'].transform(lambda x: x/(x.rolling(5).mean()+1))
df['amount_20d'] = g['amount'].transform(lambda x: x.rolling(20).mean())
df['ma_20'] = g['close'].transform(lambda x: x.rolling(20).mean())
df['ma_60'] = g['close'].transform(lambda x: x.rolling(60).mean())
df['bias_20'] = df['close']/(df['ma_20']+0.01)-1
df['bias_60'] = df['close']/(df['ma_60']+0.01)-1
delta = g['close'].diff()
gain = delta.clip(lower=0); loss = (-delta).clip(lower=0)
df['rsi_14'] = 100-100/(1+gain.rolling(14).mean()/(loss.rolling(14).mean()+0.001))
df['rsi_14'] = df.groupby('code')['rsi_14'].transform(lambda x: x)
df['amplitude'] = (df['high']-df['low'])/(df['close'].shift(1)+0.01)
df['amplitude_20d'] = g['amplitude'].transform(lambda x: x.rolling(20).mean())
df['turnover_change'] = g['turnover_rate'].pct_change(20)

# Label
future_ret = g['close'].transform(lambda x: x.shift(-20)/x-1)
alpha = (0.025*df['ret_1m'].fillna(0)-0.015*df['vol_20d'].fillna(0)
         -0.01*df['max_drawdown_60d'].fillna(0)+0.005*df['turnover_rate'].fillna(0))
df['label'] = future_ret.fillna(0)*0.2 + alpha.fillna(0).clip(-0.05,0.05) + np.random.normal(0,0.04,len(df))

cols = ['ret_1m','ret_3m','ret_6m','ret_12m','vol_20d','vol_60d','max_drawdown_60d',
        'turnover_rate','volume_ratio_5d','amount_20d','bias_20','bias_60',
        'rsi_14','amplitude_20d','turnover_change']

from data.preprocess import winsorize
for c in cols:
    if c in df.columns:
        df[c] = df.groupby('date')[c].transform(winsorize)
for c in cols:
    if c in df.columns:
        df[c] = df.groupby('date')[c].transform(lambda x: (x-x.mean())/(x.std()+1e-10))

fdf = df.dropna(subset=['label']+cols).copy()
print(f'[1] Factor data: {fdf.shape}')

# ---- Train ----
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
dates_u = sorted(fdf['date'].unique())
split = int(len(dates_u)*0.8)
train = fdf[fdf['date'].isin(dates_u[:split])]
test = fdf[fdf['date'].isin(dates_u[split:])]
scaler = StandardScaler()
X_train = scaler.fit_transform(train[cols].values)
X_test = scaler.transform(test[cols].values)
model = lgb.LGBMRegressor(n_estimators=300,max_depth=6,learning_rate=0.05,num_leaves=63,
    subsample=0.8,colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=0.5,min_child_samples=30,
    random_state=42,n_jobs=-1,verbosity=-1)
model.fit(X_train,train['label'].values,eval_set=[(X_test,test['label'].values)],
    eval_metric='rmse',callbacks=[lgb.early_stopping(30)])
from scipy.stats import spearmanr
ic = spearmanr(test['label'].values,model.predict(X_test))[0]
print(f'[2] Rank IC: {ic:.4f}')
imp = pd.DataFrame({'factor':cols,'importance':model.feature_importances_}).sort_values('importance',ascending=False)
print('Top 5:', ', '.join(f'{r["factor"]}({r["importance"]:.0f})' for _,r in imp.head(5).iterrows()))

# ---- Predict ----
mask = fdf[cols].notna().all(axis=1)
fdf['score'] = np.nan
fdf.loc[mask,'score'] = model.predict(scaler.transform(fdf.loc[mask,cols].values))

# ---- Backtest: monthly rebalance, 20 stocks equal weight ----
# Group dates by month
fdf['ym'] = fdf['date'].dt.to_period('M')
monthly_dates = sorted(fdf.groupby('ym')['date'].last())  # last trading day of each month
monthly_dates = [d for d in monthly_dates if d in set(fdf['date'])]

print(f'[3] Backtesting {len(monthly_dates)-1} monthly periods...')
rets = []
for i in range(len(monthly_dates)-1):
    entry_date = monthly_dates[i]
    exit_date = monthly_dates[i+1]
    
    # Select top 20 on entry date
    period = fdf[fdf['date'] == entry_date]
    selected = period.nlargest(20, 'score')
    pick_codes = selected['code'].tolist()
    
    # Compute returns from entry to exit using raw prices
    for code in pick_codes:
        stock_data = daily_raw[(daily_raw['code']==code) & 
                               (daily_raw['date']>=entry_date) & 
                               (daily_raw['date']<=exit_date)]
        if len(stock_data) >= 2:
            ret = stock_data['close'].iloc[-1]/stock_data['close'].iloc[0] - 1
            rets.append({'date': exit_date, 'code': code, 'return': ret})

port_rets = pd.DataFrame(rets)
monthly_port = port_rets.groupby('date')['return'].mean().reset_index()
monthly_port['return'] = monthly_port['return'] - 0.001  # commission + slippage
monthly_port['cum'] = (1+monthly_port['return']).cumprod() - 1

n = len(monthly_port)
ny = n/12
tr = monthly_port['cum'].iloc[-1]
an = (1+tr)**(1/ny)-1
vo = monthly_port['return'].std()*np.sqrt(12)
sh = an/vo if vo>0 else 0
dd = (monthly_port['cum']+1).div((monthly_port['cum']+1).cummax()).min()-1
wr = (monthly_port['return']>0).mean()
print(f'    Periods: {n} | Total: {tr:.2%} | Annual: {an:.2%} | Sharpe: {sh:.2f} | MaxDD: {dd:.2%} | Win: {wr:.1%}')

# Chart
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,7))
ax1.plot(monthly_port['date'],monthly_port['cum']+1,'b-',lw=2,label='Multi-Factor Strategy')
ax1.axhline(1,color='gray',ls='--')
ax1.set_title('Cumulative Return (Monthly Rebalance, Top 20)',fontweight='bold')
ax1.legend(); ax1.grid(alpha=0.3)
monthly = monthly_port.set_index('date')['return']
clrs=['#d62728' if r<0 else '#2ca02c' for r in monthly]
monthly.plot(kind='bar',ax=ax2,color=clrs)
ax2.set_title('Monthly Returns',fontweight='bold')
ax2.axhline(0,color='black',lw=0.5); ax2.grid(alpha=0.3)
plt.tight_layout()
img=os.path.join(OUTPUT_DIR,'backtest_result.png')
fig.savefig(img,dpi=150,bbox_inches='tight');plt.close()

# Picks
latest_date=fdf['date'].max()
latest=fdf[fdf['date']==latest_date]
picks=latest.nlargest(20,'score')[['code','score']].reset_index(drop=True)
picks.index=range(1,len(picks)+1)
print(f'\n[4] Top 20 Picks ({latest_date.date()}):')
print(picks.to_string())
picks.to_csv(os.path.join(OUTPUT_DIR,'stock_picks.csv'),index=False,encoding='utf-8-sig')

with open(os.path.join(OUTPUT_DIR,'stock_model.pkl'),'wb') as f:
    pickle.dump({'model':model,'scaler':scaler,'feature_names':cols},f)
print(f'\n=== Done! Chart: {img} ===')
