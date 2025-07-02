import yfinance as yf
import numpy as np

def download_data(symbol, start_date, end_date):
    return yf.download(symbol, start_date, end_date)

def preprocess_data(data):
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Returns'] = data['Close'].pct_change()
    
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    return data

def get_state(data, index):
    return np.array([
        float(data.loc[index, 'Close']),
        float(data.loc[index, 'SMA_5']),
        float(data.loc[index, 'SMA_20']),
        float(data.loc[index, 'Returns'])
    ])