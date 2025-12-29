"""
Stratégies de trading pour le backtesting
"""
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands


def strategy_buy_and_hold(data: pd.DataFrame, initial_capital: float = 10000) -> pd.DataFrame:
    """
    Stratégie Buy and Hold: acheter au début et conserver jusqu'à la fin.
    """
    df = data.copy()
    df['Signal'] = 1  # Toujours en position
    df['Position'] = 1
    
    # Calcul des rendements
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Position'].shift(1) * df['Returns']
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    df['Portfolio_Value'] = initial_capital * df['Cumulative_Returns']
    
    return df


def strategy_sma_crossover(data: pd.DataFrame, short_window: int = 20, long_window: int = 50, 
                           initial_capital: float = 10000) -> pd.DataFrame:
    """
    Stratégie de croisement de moyennes mobiles simples (SMA).
    Achat quand SMA courte croise au-dessus de SMA longue.
    Vente quand SMA courte croise en-dessous de SMA longue.
    """
    df = data.copy()
    
    # Calcul des SMA
    df['SMA_Short'] = SMAIndicator(close=df['Close'], window=short_window).sma_indicator()
    df['SMA_Long'] = SMAIndicator(close=df['Close'], window=long_window).sma_indicator()
    
    # Signaux
    df['Signal'] = 0
    df.loc[df['SMA_Short'] > df['SMA_Long'], 'Signal'] = 1
    df.loc[df['SMA_Short'] < df['SMA_Long'], 'Signal'] = -1
    
    # Position (éviter le look-ahead bias)
    df['Position'] = df['Signal'].shift(1)
    
    # Calcul des rendements
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Position'] * df['Returns']
    df['Strategy_Returns'] = df['Strategy_Returns'].fillna(0)
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    df['Portfolio_Value'] = initial_capital * df['Cumulative_Returns']
    
    return df


def strategy_ema_crossover(data: pd.DataFrame, short_window: int = 12, long_window: int = 26, 
                           initial_capital: float = 10000) -> pd.DataFrame:
    """
    Stratégie de croisement de moyennes mobiles exponentielles (EMA).
    """
    df = data.copy()
    
    # Calcul des EMA
    df['EMA_Short'] = EMAIndicator(close=df['Close'], window=short_window).ema_indicator()
    df['EMA_Long'] = EMAIndicator(close=df['Close'], window=long_window).ema_indicator()
    
    # Signaux
    df['Signal'] = 0
    df.loc[df['EMA_Short'] > df['EMA_Long'], 'Signal'] = 1
    df.loc[df['EMA_Short'] < df['EMA_Long'], 'Signal'] = -1
    
    # Position
    df['Position'] = df['Signal'].shift(1)
    
    # Calcul des rendements
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Position'] * df['Returns']
    df['Strategy_Returns'] = df['Strategy_Returns'].fillna(0)
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    df['Portfolio_Value'] = initial_capital * df['Cumulative_Returns']
    
    return df


def strategy_rsi(data: pd.DataFrame, rsi_window: int = 14, oversold: int = 30, 
                 overbought: int = 70, initial_capital: float = 10000) -> pd.DataFrame:
    """
    Stratégie RSI: Achat quand RSI < oversold, Vente quand RSI > overbought.
    """
    df = data.copy()
    
    # Calcul du RSI
    df['RSI'] = RSIIndicator(close=df['Close'], window=rsi_window).rsi()
    
    # Signaux
    df['Signal'] = 0
    df.loc[df['RSI'] < oversold, 'Signal'] = 1  # Survendu -> Achat
    df.loc[df['RSI'] > overbought, 'Signal'] = -1  # Suracheté -> Vente
    
    # Propager le signal (rester en position jusqu'au prochain signal contraire)
    df['Signal'] = df['Signal'].replace(0, np.nan).ffill().fillna(0)
    
    # Position
    df['Position'] = df['Signal'].shift(1)
    
    # Calcul des rendements
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Position'] * df['Returns']
    df['Strategy_Returns'] = df['Strategy_Returns'].fillna(0)
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    df['Portfolio_Value'] = initial_capital * df['Cumulative_Returns']
    
    return df


def strategy_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9,
                  initial_capital: float = 10000) -> pd.DataFrame:
    """
    Stratégie MACD: Achat quand MACD croise au-dessus de la ligne de signal.
    """
    df = data.copy()
    
    # Calcul du MACD
    macd = MACD(close=df['Close'], window_slow=slow, window_fast=fast, window_sign=signal)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Signaux
    df['Signal'] = 0
    df.loc[df['MACD'] > df['MACD_Signal'], 'Signal'] = 1
    df.loc[df['MACD'] < df['MACD_Signal'], 'Signal'] = -1
    
    # Position
    df['Position'] = df['Signal'].shift(1)
    
    # Calcul des rendements
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Position'] * df['Returns']
    df['Strategy_Returns'] = df['Strategy_Returns'].fillna(0)
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    df['Portfolio_Value'] = initial_capital * df['Cumulative_Returns']
    
    return df


def strategy_bollinger_bands(data: pd.DataFrame, window: int = 20, std_dev: float = 2.0,
                             initial_capital: float = 10000) -> pd.DataFrame:
    """
    Stratégie Bollinger Bands: Achat quand le prix touche la bande inférieure,
    Vente quand le prix touche la bande supérieure.
    """
    df = data.copy()
    
    # Calcul des Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=window, window_dev=std_dev)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()
    
    # Signaux
    df['Signal'] = 0
    df.loc[df['Close'] <= df['BB_Low'], 'Signal'] = 1  # Prix bas -> Achat
    df.loc[df['Close'] >= df['BB_High'], 'Signal'] = -1  # Prix haut -> Vente
    
    # Propager le signal
    df['Signal'] = df['Signal'].replace(0, np.nan).ffill().fillna(0)
    
    # Position
    df['Position'] = df['Signal'].shift(1)
    
    # Calcul des rendements
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Position'] * df['Returns']
    df['Strategy_Returns'] = df['Strategy_Returns'].fillna(0)
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    df['Portfolio_Value'] = initial_capital * df['Cumulative_Returns']
    
    return df


def strategy_stochastic(data: pd.DataFrame, k_window: int = 14, d_window: int = 3,
                        oversold: int = 20, overbought: int = 80,
                        initial_capital: float = 10000) -> pd.DataFrame:
    """
    Stratégie Stochastique: Achat quand %K croise %D en zone de survente.
    """
    df = data.copy()
    
    # Calcul du Stochastique
    stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'],
                                  window=k_window, smooth_window=d_window)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    # Signaux
    df['Signal'] = 0
    # Achat: %K croise au-dessus de %D en zone de survente
    df.loc[(df['Stoch_K'] > df['Stoch_D']) & (df['Stoch_K'] < oversold), 'Signal'] = 1
    # Vente: %K croise en-dessous de %D en zone de surachat
    df.loc[(df['Stoch_K'] < df['Stoch_D']) & (df['Stoch_K'] > overbought), 'Signal'] = -1
    
    # Propager le signal
    df['Signal'] = df['Signal'].replace(0, np.nan).ffill().fillna(0)
    
    # Position
    df['Position'] = df['Signal'].shift(1)
    
    # Calcul des rendements
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Position'] * df['Returns']
    df['Strategy_Returns'] = df['Strategy_Returns'].fillna(0)
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    df['Portfolio_Value'] = initial_capital * df['Cumulative_Returns']
    
    return df


# Dictionnaire des stratégies disponibles
STRATEGIES = {
    "Buy and Hold": strategy_buy_and_hold,
    "SMA Crossover": strategy_sma_crossover,
    "EMA Crossover": strategy_ema_crossover,
    "RSI": strategy_rsi,
    "MACD": strategy_macd,
    "Bollinger Bands": strategy_bollinger_bands,
    "Stochastic": strategy_stochastic,
}
