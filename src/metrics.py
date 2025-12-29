"""
Métriques de performance pour le backtesting
"""
import pandas as pd
import numpy as np
from typing import Dict, Any


def safe_value(value, format_str: str = "{:.2f}", suffix: str = "", prefix: str = "") -> str:
    """
    Formate une valeur de manière sécurisée, retourne 'N/A' si NaN ou infini.
    """
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "N/A"
    try:
        return prefix + format_str.format(value) + suffix
    except:
        return "N/A"


def calculate_metrics(df: pd.DataFrame, risk_free_rate: float = 0.02) -> Dict[str, Any]:
    """
    Calcule les métriques de performance d'une stratégie.
    
    Args:
        df: DataFrame avec les résultats du backtest
        risk_free_rate: Taux sans risque annualisé (défaut: 2%)
    
    Returns:
        Dictionnaire avec toutes les métriques
    """
    # Vérifier que les colonnes nécessaires existent
    if 'Strategy_Returns' not in df.columns or 'Portfolio_Value' not in df.columns:
        return {}
    
    # Supprimer les NaN et les valeurs infinies
    returns = df['Strategy_Returns'].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(returns) == 0:
        return {}
    
    # Nombre de jours de trading par an
    trading_days = 252
    
    # Valeurs du portefeuille (nettoyer les NaN)
    portfolio_values = df['Portfolio_Value'].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(portfolio_values) < 2:
        return {}
    
    initial_value = portfolio_values.iloc[0]
    final_value = portfolio_values.iloc[-1]
    
    # Rendement total
    if initial_value > 0:
        total_return = (final_value / initial_value) - 1
    else:
        total_return = 0
    
    # Rendement annualisé
    n_days = len(df)
    n_years = n_days / trading_days
    if n_years > 0 and total_return > -1:
        annual_return = (1 + total_return) ** (1 / n_years) - 1
    else:
        annual_return = 0
    
    # Volatilité annualisée
    daily_volatility = returns.std()
    if pd.isna(daily_volatility) or daily_volatility == 0:
        annual_volatility = 0
    else:
        annual_volatility = daily_volatility * np.sqrt(trading_days)
    
    # Ratio de Sharpe
    excess_return = annual_return - risk_free_rate
    if annual_volatility > 0:
        sharpe_ratio = excess_return / annual_volatility
    else:
        sharpe_ratio = 0
    
    # Ratio de Sortino (pénalise seulement la volatilité à la baisse)
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 0:
        downside_std = negative_returns.std()
        if pd.notna(downside_std) and downside_std > 0:
            downside_volatility = downside_std * np.sqrt(trading_days)
            sortino_ratio = excess_return / downside_volatility
        else:
            sortino_ratio = 0
    else:
        sortino_ratio = 0
    
    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    drawdown = drawdown.replace([np.inf, -np.inf], np.nan).dropna()
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
    if pd.isna(max_drawdown):
        max_drawdown = 0
    
    # Calmar Ratio
    if max_drawdown != 0 and not pd.isna(max_drawdown):
        calmar_ratio = annual_return / abs(max_drawdown)
    else:
        calmar_ratio = 0
    
    # Ratio de gains/pertes
    winning_trades = returns[returns > 0]
    losing_trades = returns[returns < 0]
    
    win_rate = len(winning_trades) / len(returns) * 100 if len(returns) > 0 else 0
    
    avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
    if pd.isna(avg_win):
        avg_win = 0
    
    avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
    if pd.isna(avg_loss):
        avg_loss = 0
    
    losing_sum = losing_trades.sum()
    winning_sum = winning_trades.sum()
    if losing_sum != 0 and not pd.isna(losing_sum) and not pd.isna(winning_sum):
        profit_factor = abs(winning_sum / losing_sum)
    else:
        profit_factor = 0
    
    # Nombre de trades (changements de position)
    if 'Position' in df.columns:
        position_changes = df['Position'].diff().fillna(0)
        n_trades = int((position_changes != 0).sum())
    else:
        n_trades = 0
    
    return {
        "Valeur Initiale": safe_value(initial_value, "{:,.2f}", " €"),
        "Valeur Finale": safe_value(final_value, "{:,.2f}", " €"),
        "Rendement Total": safe_value(total_return * 100, "{:.2f}", "%"),
        "Rendement Annualisé": safe_value(annual_return * 100, "{:.2f}", "%"),
        "Volatilité Annualisée": safe_value(annual_volatility * 100, "{:.2f}", "%"),
        "Ratio de Sharpe": safe_value(sharpe_ratio, "{:.2f}"),
        "Ratio de Sortino": safe_value(sortino_ratio, "{:.2f}"),
        "Maximum Drawdown": safe_value(max_drawdown * 100, "{:.2f}", "%"),
        "Ratio Calmar": safe_value(calmar_ratio, "{:.2f}"),
        "Taux de Réussite": safe_value(win_rate, "{:.1f}", "%"),
        "Gain Moyen": safe_value(avg_win * 100, "{:.2f}", "%"),
        "Perte Moyenne": safe_value(avg_loss * 100, "{:.2f}", "%"),
        "Profit Factor": safe_value(profit_factor, "{:.2f}"),
        "Nombre de Trades": str(n_trades),
    }


def compare_strategies(results: Dict[str, pd.DataFrame], risk_free_rate: float = 0.02) -> pd.DataFrame:
    """
    Compare les métriques de plusieurs stratégies.
    
    Args:
        results: Dictionnaire {nom_stratégie: DataFrame_résultats}
        risk_free_rate: Taux sans risque
    
    Returns:
        DataFrame avec les métriques comparées
    """
    comparison = {}
    
    for strategy_name, df in results.items():
        metrics = calculate_metrics(df, risk_free_rate)
        comparison[strategy_name] = metrics
    
    return pd.DataFrame(comparison).T
