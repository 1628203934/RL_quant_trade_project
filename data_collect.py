from model_libraries import yf, np


def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Returns'] = data['Close'].pct_change()

    # Technical Indicators
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['RSI'] = compute_rsi(data['Close'], window=14)

    data['MACD'] = compute_macd(data['Close'])
    data['Bollinger_Upper'], data['Bollinger_Lower'] = compute_bollinger_bands(data['Close'], window=20)
    data['ATR'] = compute_atr(data['High'], data['Low'], data['Close'], window=14)

    # Drop rows with missing values
    data.dropna(inplace=True)

    return data


def compute_rsi(series, window):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series):
    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    return macd


def compute_bollinger_bands(series, window):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band


def compute_atr(high, low, close, window):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    tr = high_low.combine(high_close, max).combine(low_close, max)
    atr = tr.rolling(window=window).mean()
    return atr

def get_data():
    return data

start_date = '2018-01-01'
end_date = '2022-01-01'
data = get_stock_data('AAPL', start_date, end_date)
train_data = data[:'2019-12-31']
test_data = data['2020-01-01':]
data.to_csv('train_Stock_data_{}_to_{}.csv'.format(start_date, '2019-12-31'), index=True)
data.to_csv('test_Stock_data_{}_to_{}.csv'.format('2020-01-01', end_date), index=True)
