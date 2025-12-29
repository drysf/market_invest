"""
Application Streamlit de Backtesting d'Investissement
Theme: Dark Mode
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from src.data_loader import get_stock_data, POPULAR_TICKERS
from src.strategies import STRATEGIES
from src.metrics import calculate_metrics, compare_strategies

# Configuration de la page
st.set_page_config(
    page_title="Backtesting Investissement",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalise theme sombre
st.markdown("""
<style>
    /* Theme sombre global */
    .stApp {
        background-color: #000000;
        color: #fafafa;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #fafafa;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        border-bottom: 2px solid #333333;
    }
    
    .metric-card {
        background-color: #000000;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #333333;
    }
    
    .positive {
        color: #00ff88;
    }
    
    .negative {
        color: #ff4444;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 3px solid #FF4B4B;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #fafafa;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #fafafa !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #fafafa;
    }
    
    [data-testid="stMetricLabel"] {
        color: #b0b0b0;
    }
    
    /* Dataframes */
    .stDataFrame {
        background-color: #000000;
    }
    
    /* Dividers */
    hr {
        border-color: #333333;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #000000;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #fafafa;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #000000;
        color: #fafafa;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #000000;
        color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<p class="main-header">SIMULATEUR DE BACKTESTING</p>', unsafe_allow_html=True)

# Sidebar pour les parametres
with st.sidebar:
    st.header("Parametres")
    
    # Selection du ticker
    st.subheader("Selection de l'Actif")
    
    # Mode de selection
    selection_mode = st.radio(
        "Mode de selection",
        ["Liste predefinie", "Saisie manuelle"]
    )
    
    if selection_mode == "Liste predefinie":
        category = st.selectbox(
            "Categorie",
            list(POPULAR_TICKERS.keys())
        )
        ticker_options = POPULAR_TICKERS[category]
        selected_ticker = st.selectbox(
            "Actif",
            list(ticker_options.keys()),
            format_func=lambda x: f"{x} - {ticker_options[x]}"
        )
    else:
        selected_ticker = st.text_input(
            "Symbole du ticker",
            value="AAPL",
            help="Ex: AAPL, MSFT, BTC-USD, ^GSPC"
        ).upper()
    
    # Periode d'analyse
    st.subheader("Periode d'Analyse")
    
    col1, col2 = st.columns(2)
    
    default_end = datetime.now()
    default_start = default_end - timedelta(days=365*3)  # 3 ans par defaut
    
    with col1:
        start_date = st.date_input(
            "Date de debut",
            value=default_start,
            max_value=default_end
        )
    
    with col2:
        end_date = st.date_input(
            "Date de fin",
            value=default_end,
            max_value=default_end
        )
    
    # Capital initial
    st.subheader("Capital")
    initial_capital = st.number_input(
        "Capital initial (EUR)",
        min_value=100,
        max_value=10000000,
        value=10000,
        step=1000
    )
    
    # Selection de la strategie
    st.subheader("Strategie de Trading")
    selected_strategy = st.selectbox(
        "Strategie",
        list(STRATEGIES.keys())
    )
    
    # Parametres specifiques a chaque strategie
    st.subheader("Parametres de la Strategie")
    
    strategy_params = {}
    
    if selected_strategy == "SMA Crossover":
        strategy_params['short_window'] = st.slider("SMA Courte (jours)", 5, 50, 20)
        strategy_params['long_window'] = st.slider("SMA Longue (jours)", 20, 200, 50)
    
    elif selected_strategy == "EMA Crossover":
        strategy_params['short_window'] = st.slider("EMA Courte (jours)", 5, 30, 12)
        strategy_params['long_window'] = st.slider("EMA Longue (jours)", 15, 60, 26)
    
    elif selected_strategy == "RSI":
        strategy_params['rsi_window'] = st.slider("Periode RSI", 5, 30, 14)
        strategy_params['oversold'] = st.slider("Niveau de survente", 10, 40, 30)
        strategy_params['overbought'] = st.slider("Niveau de surachat", 60, 90, 70)
    
    elif selected_strategy == "MACD":
        strategy_params['fast'] = st.slider("Periode rapide", 5, 20, 12)
        strategy_params['slow'] = st.slider("Periode lente", 15, 40, 26)
        strategy_params['signal'] = st.slider("Periode signal", 5, 15, 9)
    
    elif selected_strategy == "Bollinger Bands":
        strategy_params['window'] = st.slider("Periode", 10, 50, 20)
        strategy_params['std_dev'] = st.slider("Ecart-type", 1.0, 3.0, 2.0, 0.1)
    
    elif selected_strategy == "Stochastic":
        strategy_params['k_window'] = st.slider("Periode %K", 5, 30, 14)
        strategy_params['d_window'] = st.slider("Periode %D", 1, 10, 3)
        strategy_params['oversold'] = st.slider("Niveau de survente", 10, 30, 20)
        strategy_params['overbought'] = st.slider("Niveau de surachat", 70, 90, 80)
    
    # Bouton de lancement
    run_backtest = st.button("Lancer le Backtest", type="primary", use_container_width=True)
    
    # Comparaison de strategies
    st.subheader("Comparaison")
    compare_all = st.checkbox("Comparer toutes les strategies")


def apply_dark_theme(fig):
    """Applique le theme sombre a un graphique Plotly"""
    fig.update_layout(
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        font=dict(color='#fafafa'),
        xaxis=dict(
            gridcolor='#333333',
            linecolor='#333333',
            zerolinecolor='#333333',
            tickfont=dict(color='#fafafa')
        ),
        yaxis=dict(
            gridcolor='#333333',
            linecolor='#333333',
            zerolinecolor='#333333',
            tickfont=dict(color='#fafafa')
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fafafa')
        )
    )
    return fig


# Zone principale
if run_backtest or 'data' in st.session_state:
    
    # Chargement des donnees
    with st.spinner(f"Chargement des donnees pour {selected_ticker}..."):
        data, stock_info = get_stock_data(
            selected_ticker,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
    
    if data.empty:
        error_detail = stock_info.get('error', 'symbole invalide ou aucune donnée disponible')
        st.error(f"❌ Impossible de charger les données pour {selected_ticker}")
        st.warning(f"""
        **Causes possibles:**
        - Vérifiez votre connexion Internet
        - Le symbole peut être incorrect
        - Les dates sélectionnées peuvent être invalides
        - Yahoo Finance peut être temporairement indisponible
        
        **Détails:** {error_detail}
        
        **Solutions:**
        - Essayez de mettre à jour yfinance: `pip install --upgrade yfinance`
        - Vérifiez que le symbole est correct (ex: AAPL pour Apple)
        - Essayez avec une autre période
        """)
        st.stop()
    else:
        # Informations sur l'action
        st.subheader(f"{stock_info.get('name', selected_ticker)}")
        
        info_cols = st.columns(4)
        with info_cols[0]:
            st.metric("Secteur", stock_info.get('sector', 'N/A'))
        with info_cols[1]:
            st.metric("Industrie", stock_info.get('industry', 'N/A'))
        with info_cols[2]:
            st.metric("Devise", stock_info.get('currency', 'USD'))
        with info_cols[3]:
            market_cap = stock_info.get('market_cap', 0)
            if market_cap > 0:
                if market_cap >= 1e12:
                    cap_str = f"{market_cap/1e12:.2f}T"
                elif market_cap >= 1e9:
                    cap_str = f"{market_cap/1e9:.2f}B"
                else:
                    cap_str = f"{market_cap/1e6:.2f}M"
                st.metric("Cap. Marche", cap_str)
            else:
                st.metric("Cap. Marche", "N/A")
        
        st.divider()
        
        # Execution du backtest
        if compare_all:
            # Comparer toutes les strategies
            st.subheader("Comparaison de Toutes les Strategies")
            
            results = {}
            for strategy_name, strategy_func in STRATEGIES.items():
                try:
                    result = strategy_func(data.copy(), initial_capital=initial_capital)
                    results[strategy_name] = result
                except Exception as e:
                    st.warning(f"Erreur avec {strategy_name}: {e}")
            
            # Graphique de comparaison
            fig = go.Figure()
            
            colors = ['#00ff88', '#ff4444', '#4488ff', '#ffaa00', '#aa44ff', '#44ffff', '#ff44aa']
            for i, (name, result) in enumerate(results.items()):
                fig.add_trace(go.Scatter(
                    x=result.index,
                    y=result['Portfolio_Value'],
                    mode='lines',
                    name=name,
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
            
            fig.update_layout(
                title="Evolution du Portefeuille - Comparaison des Strategies",
                xaxis_title="Date",
                yaxis_title="Valeur du Portefeuille (EUR)",
                hovermode='x unified',
                height=500
            )
            fig = apply_dark_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau de comparaison
            comparison_df = compare_strategies(results)
            st.subheader("Tableau Comparatif")
            st.dataframe(comparison_df, use_container_width=True)
            
        else:
            # Strategie unique
            strategy_func = STRATEGIES[selected_strategy]
            result = strategy_func(data.copy(), initial_capital=initial_capital, **strategy_params)
            
            # Metriques de performance
            metrics = calculate_metrics(result)
            
            st.subheader("Metriques de Performance")
            
            # Affichage des metriques principales
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                st.metric(
                    "Valeur Finale",
                    metrics.get("Valeur Finale", "N/A"),
                    metrics.get("Rendement Total", "N/A")
                )
            
            with metric_cols[1]:
                st.metric(
                    "Rendement Annualise",
                    metrics.get("Rendement Annualise", "N/A")
                )
            
            with metric_cols[2]:
                st.metric(
                    "Ratio de Sharpe",
                    metrics.get("Ratio de Sharpe", "N/A")
                )
            
            with metric_cols[3]:
                st.metric(
                    "Max Drawdown",
                    metrics.get("Maximum Drawdown", "N/A")
                )
            
            # Metriques secondaires
            metric_cols2 = st.columns(4)
            
            with metric_cols2[0]:
                st.metric("Ratio Sortino", metrics.get("Ratio de Sortino", "N/A"))
            
            with metric_cols2[1]:
                st.metric("Ratio Calmar", metrics.get("Ratio Calmar", "N/A"))
            
            with metric_cols2[2]:
                st.metric("Taux de Reussite", metrics.get("Taux de Reussite", "N/A"))
            
            with metric_cols2[3]:
                st.metric("Nombre de Trades", metrics.get("Nombre de Trades", "N/A"))
            
            # Metriques tertiaires
            with st.expander("Voir toutes les metriques"):
                metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metrique', 'Valeur'])
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # Graphiques
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Portefeuille", "Prix et Signaux", "Indicateurs", "Drawdown", "Analyse"
            ])
            
            with tab1:
                # Graphique principal: Valeur du portefeuille
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=result.index,
                    y=result['Portfolio_Value'],
                    mode='lines',
                    name='Portefeuille',
                    line=dict(color='#00ff88', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0, 255, 136, 0.1)'
                ))
                
                # Ligne de reference (capital initial)
                fig.add_hline(
                    y=initial_capital,
                    line_dash="dash",
                    line_color="#ff4444",
                    annotation_text="Capital Initial",
                    annotation_font_color="#fafafa"
                )
                
                fig.update_layout(
                    title="Evolution du Portefeuille",
                    xaxis_title="Date",
                    yaxis_title="Valeur (EUR)",
                    height=500
                )
                fig = apply_dark_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                
                # Rendements cumules vs Buy and Hold
                fig2 = go.Figure()
                
                # Rendements de la strategie
                fig2.add_trace(go.Scatter(
                    x=result.index,
                    y=result['Cumulative_Returns'],
                    mode='lines',
                    name='Strategie',
                    line=dict(color='#00ff88', width=2)
                ))
                
                # Buy and Hold pour comparaison
                buy_hold_returns = (result['Close'] / result['Close'].iloc[0])
                fig2.add_trace(go.Scatter(
                    x=result.index,
                    y=buy_hold_returns,
                    mode='lines',
                    name='Buy and Hold',
                    line=dict(color='#4488ff', width=2, dash='dash')
                ))
                
                fig2.update_layout(
                    title="Rendements Cumules: Strategie vs Buy and Hold",
                    xaxis_title="Date",
                    yaxis_title="Rendement Cumule",
                    height=400
                )
                fig2 = apply_dark_theme(fig2)
                st.plotly_chart(fig2, use_container_width=True)
            
            with tab2:
                # Graphique chandelier avec signaux
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=("Prix et Indicateurs", "Volume"),
                    row_heights=[0.7, 0.3]
                )
                
                # Chandelier
                fig.add_trace(
                    go.Candlestick(
                        x=result.index,
                        open=result['Open'],
                        high=result['High'],
                        low=result['Low'],
                        close=result['Close'],
                        name='Prix',
                        increasing_line_color='#00ff88',
                        decreasing_line_color='#ff4444'
                    ),
                    row=1, col=1
                )
                
                # Ajout des indicateurs selon la strategie
                if 'SMA_Short' in result.columns:
                    fig.add_trace(
                        go.Scatter(x=result.index, y=result['SMA_Short'], 
                                   mode='lines', name='SMA Courte', 
                                   line=dict(color='#ffaa00', width=1)),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=result.index, y=result['SMA_Long'], 
                                   mode='lines', name='SMA Longue',
                                   line=dict(color='#aa44ff', width=1)),
                        row=1, col=1
                    )
                
                if 'EMA_Short' in result.columns:
                    fig.add_trace(
                        go.Scatter(x=result.index, y=result['EMA_Short'], 
                                   mode='lines', name='EMA Courte',
                                   line=dict(color='#ffaa00', width=1)),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=result.index, y=result['EMA_Long'], 
                                   mode='lines', name='EMA Longue',
                                   line=dict(color='#aa44ff', width=1)),
                        row=1, col=1
                    )
                
                if 'BB_High' in result.columns:
                    fig.add_trace(
                        go.Scatter(x=result.index, y=result['BB_High'], 
                                   mode='lines', name='BB Haute',
                                   line=dict(color='#888888', dash='dash', width=1)),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=result.index, y=result['BB_Low'], 
                                   mode='lines', name='BB Basse',
                                   line=dict(color='#888888', dash='dash', width=1)),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=result.index, y=result['BB_Mid'], 
                                   mode='lines', name='BB Moyenne',
                                   line=dict(color='#888888', width=1)),
                        row=1, col=1
                    )
                
                # Signaux d'achat/vente
                if 'Position' in result.columns:
                    position_changes = result['Position'].diff()
                    buy_signals = result[position_changes > 0]
                    sell_signals = result[position_changes < 0]
                    
                    if len(buy_signals) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=buy_signals.index,
                                y=buy_signals['Close'],
                                mode='markers',
                                name='Achat',
                                marker=dict(symbol='triangle-up', size=12, color='#00ff88')
                            ),
                            row=1, col=1
                        )
                    
                    if len(sell_signals) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=sell_signals.index,
                                y=sell_signals['Close'],
                                mode='markers',
                                name='Vente',
                                marker=dict(symbol='triangle-down', size=12, color='#ff4444')
                            ),
                            row=1, col=1
                        )
                
                # Volume
                colors_vol = ['#00ff88' if result['Close'].iloc[i] >= result['Open'].iloc[i] 
                              else '#ff4444' for i in range(len(result))]
                fig.add_trace(
                    go.Bar(x=result.index, y=result['Volume'], name='Volume',
                           marker_color=colors_vol, opacity=0.7),
                    row=2, col=1
                )
                
                fig.update_layout(
                    height=700,
                    showlegend=True,
                    xaxis_rangeslider_visible=False
                )
                fig = apply_dark_theme(fig)
                fig.update_xaxes(gridcolor='#333333', row=1, col=1)
                fig.update_xaxes(gridcolor='#333333', row=2, col=1)
                fig.update_yaxes(gridcolor='#333333', row=1, col=1)
                fig.update_yaxes(gridcolor='#333333', row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Indicateurs techniques
                if selected_strategy == "RSI" and 'RSI' in result.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=result.index, y=result['RSI'],
                        mode='lines', name='RSI',
                        line=dict(color='#4488ff', width=2)
                    ))
                    fig.add_hline(y=70, line_dash="dash", line_color="#ff4444",
                                  annotation_text="Surachat (70)")
                    fig.add_hline(y=30, line_dash="dash", line_color="#00ff88",
                                  annotation_text="Survente (30)")
                    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,68,68,0.1)", line_width=0)
                    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,255,136,0.1)", line_width=0)
                    fig.update_layout(title="RSI (Relative Strength Index)", height=400)
                    fig = apply_dark_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif selected_strategy == "MACD" and 'MACD' in result.columns:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                        subplot_titles=("MACD", "Histogramme"),
                                        row_heights=[0.6, 0.4])
                    
                    fig.add_trace(go.Scatter(
                        x=result.index, y=result['MACD'],
                        mode='lines', name='MACD',
                        line=dict(color='#4488ff', width=2)
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=result.index, y=result['MACD_Signal'],
                        mode='lines', name='Signal',
                        line=dict(color='#ffaa00', width=2)
                    ), row=1, col=1)
                    
                    colors = ['#00ff88' if v >= 0 else '#ff4444' for v in result['MACD_Hist']]
                    fig.add_trace(go.Bar(
                        x=result.index, y=result['MACD_Hist'],
                        name='Histogramme', marker_color=colors
                    ), row=2, col=1)
                    
                    fig.update_layout(title="MACD (Moving Average Convergence Divergence)", height=500)
                    fig = apply_dark_theme(fig)
                    fig.update_xaxes(gridcolor='#333333')
                    fig.update_yaxes(gridcolor='#333333')
                    st.plotly_chart(fig, use_container_width=True)
                
                elif selected_strategy == "Stochastic" and 'Stoch_K' in result.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=result.index, y=result['Stoch_K'],
                        mode='lines', name='%K',
                        line=dict(color='#4488ff', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=result.index, y=result['Stoch_D'],
                        mode='lines', name='%D',
                        line=dict(color='#ffaa00', width=2)
                    ))
                    fig.add_hline(y=80, line_dash="dash", line_color="#ff4444")
                    fig.add_hline(y=20, line_dash="dash", line_color="#00ff88")
                    fig.add_hrect(y0=80, y1=100, fillcolor="rgba(255,68,68,0.1)", line_width=0)
                    fig.add_hrect(y0=0, y1=20, fillcolor="rgba(0,255,136,0.1)", line_width=0)
                    fig.update_layout(title="Oscillateur Stochastique", height=400)
                    fig = apply_dark_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    # Afficher plusieurs indicateurs par defaut
                    st.info("Indicateurs techniques generaux")
                    
                    # RSI
                    from ta.momentum import RSIIndicator
                    rsi = RSIIndicator(close=result['Close'], window=14).rsi()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=result.index, y=rsi,
                        mode='lines', name='RSI (14)',
                        line=dict(color='#4488ff', width=2)
                    ))
                    fig.add_hline(y=70, line_dash="dash", line_color="#ff4444")
                    fig.add_hline(y=30, line_dash="dash", line_color="#00ff88")
                    fig.update_layout(title="RSI (14 periodes)", height=300)
                    fig = apply_dark_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                # Graphique du Drawdown
                cumulative = (1 + result['Strategy_Returns']).cumprod()
                running_max = cumulative.cummax()
                drawdown = (cumulative - running_max) / running_max * 100
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=result.index,
                    y=drawdown,
                    fill='tozeroy',
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='#ff4444', width=1),
                    fillcolor='rgba(255, 68, 68, 0.3)'
                ))
                
                fig.update_layout(
                    title="Drawdown du Portefeuille",
                    xaxis_title="Date",
                    yaxis_title="Drawdown (%)",
                    height=400
                )
                fig = apply_dark_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                
                # Underwater chart (temps passe sous le pic)
                fig2 = go.Figure()
                
                fig2.add_trace(go.Scatter(
                    x=result.index,
                    y=cumulative,
                    mode='lines',
                    name='Valeur Normalisee',
                    line=dict(color='#00ff88', width=2)
                ))
                
                fig2.add_trace(go.Scatter(
                    x=result.index,
                    y=running_max,
                    mode='lines',
                    name='Plus Haut',
                    line=dict(color='#4488ff', width=1, dash='dash')
                ))
                
                fig2.update_layout(
                    title="Valeur Normalisee vs Plus Haut Historique",
                    xaxis_title="Date",
                    yaxis_title="Valeur",
                    height=400
                )
                fig2 = apply_dark_theme(fig2)
                st.plotly_chart(fig2, use_container_width=True)
            
            with tab5:
                # Distribution des rendements
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = go.Figure()
                    returns_pct = result['Strategy_Returns'].dropna() * 100
                    
                    fig.add_trace(go.Histogram(
                        x=returns_pct,
                        nbinsx=50,
                        name='Rendements',
                        marker_color='#4488ff',
                        opacity=0.7
                    ))
                    
                    mean_return = returns_pct.mean()
                    fig.add_vline(x=mean_return, line_dash="dash", line_color="#00ff88",
                                  annotation_text=f"Moyenne: {mean_return:.3f}%",
                                  annotation_font_color="#fafafa")
                    fig.add_vline(x=0, line_dash="solid", line_color="#888888")
                    
                    fig.update_layout(
                        title="Distribution des Rendements Journaliers",
                        xaxis_title="Rendement (%)",
                        yaxis_title="Frequence",
                        height=400
                    )
                    fig = apply_dark_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Rendements mensuels heatmap
                    monthly_returns = result['Strategy_Returns'].resample('ME').apply(
                        lambda x: (1 + x).prod() - 1
                    ) * 100
                    
                    # Creer un DataFrame pour le heatmap
                    monthly_df = pd.DataFrame({
                        'Annee': monthly_returns.index.year,
                        'Mois': monthly_returns.index.month,
                        'Rendement': monthly_returns.values
                    })
                    
                    pivot_table = monthly_df.pivot_table(
                        values='Rendement', 
                        index='Annee', 
                        columns='Mois',
                        aggfunc='mean'
                    )
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=pivot_table.values,
                        x=['Jan', 'Fev', 'Mar', 'Avr', 'Mai', 'Jun', 
                           'Jul', 'Aou', 'Sep', 'Oct', 'Nov', 'Dec'][:pivot_table.shape[1]],
                        y=pivot_table.index,
                        colorscale=[[0, '#ff4444'], [0.5, '#333333'], [1, '#00ff88']],
                        zmid=0,
                        text=np.round(pivot_table.values, 1),
                        texttemplate='%{text}%',
                        textfont={"size": 10, "color": "#fafafa"},
                        hovertemplate='Annee: %{y}<br>Mois: %{x}<br>Rendement: %{z:.2f}%<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title="Rendements Mensuels (%)",
                        height=400
                    )
                    fig = apply_dark_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Rolling Sharpe Ratio
                st.subheader("Metriques Glissantes")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Rolling volatility
                    rolling_vol = result['Strategy_Returns'].rolling(window=30).std() * np.sqrt(252) * 100
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=result.index,
                        y=rolling_vol,
                        mode='lines',
                        name='Volatilite 30j',
                        line=dict(color='#ffaa00', width=2)
                    ))
                    
                    fig.update_layout(
                        title="Volatilite Annualisee Glissante (30 jours)",
                        xaxis_title="Date",
                        yaxis_title="Volatilite (%)",
                        height=350
                    )
                    fig = apply_dark_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Rolling Sharpe
                    rolling_return = result['Strategy_Returns'].rolling(window=60).mean() * 252
                    rolling_std = result['Strategy_Returns'].rolling(window=60).std() * np.sqrt(252)
                    rolling_sharpe = (rolling_return - 0.02) / rolling_std
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=result.index,
                        y=rolling_sharpe,
                        mode='lines',
                        name='Sharpe 60j',
                        line=dict(color='#aa44ff', width=2)
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="#888888")
                    fig.add_hline(y=1, line_dash="dash", line_color="#00ff88", 
                                  annotation_text="Bon (>1)")
                    fig.add_hline(y=-1, line_dash="dash", line_color="#ff4444",
                                  annotation_text="Mauvais (<-1)")
                    
                    fig.update_layout(
                        title="Ratio de Sharpe Glissant (60 jours)",
                        xaxis_title="Date",
                        yaxis_title="Ratio de Sharpe",
                        height=350
                    )
                    fig = apply_dark_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)

