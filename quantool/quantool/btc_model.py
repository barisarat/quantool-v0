import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from ta.trend import SMAIndicator # technical analysis library
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator

from sklearn.neighbors import KNeighborsClassifier # ml learning library
from sklearn.model_selection import train_test_split

def btc_predict(user_input=0):

    price = pd.read_csv('~/Desktop/quantool-django/coins-high-market-cap.csv')
    price.drop('Unnamed: 0', axis=1, inplace=True)
    price['Date'] = pd.to_datetime(price['Date'])
    price = price.set_index('Date')
    # index 16, 24, 28 has low sample thus drop them
    price.drop(['Close-ICPUSDT','Close-SHIBUSDT','Close-CAKEUSDT'], axis=1, inplace=True)

    # STEP 1: BUILDING FEATURES ARRAY
    # Normalized MA diff
    ma_50 = SMAIndicator(price['Close-BTCUSDT'], 50) # takes pandas series input
    ma_50 = ma_50.sma_indicator().values # convert to numpy for ml model
    ma_100 = SMAIndicator(price['Close-BTCUSDT'], 100)
    ma_100 = ma_100.sma_indicator().values
    sma_diff = (ma_50 - ma_100)/ma_50

    # BOLLINGER BANDS
    bollinger = BollingerBands(price['Close-BTCUSDT'])
    middle_b = bollinger.bollinger_mavg().values
    bband_oscilator = (price['Close-BTCUSDT'].values - middle_b)

    #RSI
    rsi = RSIIndicator(price['Close-BTCUSDT'], 14)
    rsi = rsi.rsi().values

    # THIS IS THE FEATURES DATA
    data = np.stack((sma_diff, bband_oscilator, rsi), axis=1)
    btc_indicators = data[99:]

    # STEP 2: BUILDING TARGET ARRAY
    btc_changes = price['Close-BTCUSDT'].pct_change().values
    btc_changes = btc_changes[~np.isnan(btc_changes)]  # shape decreased 1 now (1422,)
                                             # drop the first value from index date
    idx = range(len(btc_changes))
    btc_positions = np.empty((1422,)) # THIS IS THE TARGET DATA

    for i, change in zip(idx, btc_changes):
        if change > 0.0299:
            btc_positions[i] = 1.
        elif change < -0.0299:
            btc_positions[i] = -1.
        else:
            btc_positions[i] = 0.

    btc_positions = btc_positions[98:] # to match the data in features
    # noted that features selected 99, but 1 na was dropped thus 98: selected here
    X = btc_indicators
    y = btc_positions
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=4) # k optimization done in the plot below
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)
    prediction = prediction[user_input]
    if prediction == -1:
        prediction = 'Sell'
    elif prediction == 0:
        prediction = 'Neutral'
    elif prediction == 1:
        prediction = 'Buy'
    else:
        print('Something wrong with predictions!')
    return prediction
