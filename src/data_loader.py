"""
Module pour charger les données financières via yfinance
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple, List


def get_stock_data(ticker: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, dict]:
    """
    Télécharge les données historiques d'une action via yfinance.
    
    Args:
        ticker: Symbole de l'action (ex: 'AAPL', 'MSFT')
        start_date: Date de début (format 'YYYY-MM-DD')
        end_date: Date de fin (format 'YYYY-MM-DD')
    
    Returns:
        Tuple (DataFrame des données, dict avec infos de l'action)
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Téléchargement avec timeout et retry
        data = stock.history(start=start_date, end=end_date, timeout=10)
        
        if data.empty:
            print(f"Aucune donnée retournée pour {ticker} entre {start_date} et {end_date}")
            return pd.DataFrame(), {'error': 'no_data'}
        
        # Nettoyer les données
        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
        data = data.set_index('Date')
        
        # Infos sur l'action
        try:
            info = stock.info
            stock_info = {
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
            }
        except Exception as e:
            print(f"Impossible de récupérer les infos: {e}")
            stock_info = {'name': ticker}
        
        return data, stock_info
    
    except Exception as e:
        error_msg = str(e)
        print(f"Erreur lors du téléchargement des données pour {ticker}: {error_msg}")
        return pd.DataFrame(), {'error': error_msg}


def get_multiple_stocks(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Télécharge les données de plusieurs actions.
    
    Args:
        tickers: Liste des symboles
        start_date: Date de début
        end_date: Date de fin
    
    Returns:
        DataFrame avec les prix de clôture de toutes les actions
    """
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Close']
        return data
    except Exception as e:
        print(f"Erreur: {e}")
        return pd.DataFrame()


def search_ticker(query: str) -> List[dict]:
    """
    Recherche un ticker par nom ou symbole.
    
    Args:
        query: Terme de recherche
    
    Returns:
        Liste de résultats
    """
    try:
        ticker = yf.Ticker(query)
        info = ticker.info
        if info and 'symbol' in info:
            return [{
                'symbol': info.get('symbol', query),
                'name': info.get('longName', 'N/A'),
                'exchange': info.get('exchange', 'N/A'),
            }]
    except:
        pass
    return []


# Tickers populaires pré-définis (liste étendue)
POPULAR_TICKERS = {
    "Actions US - Tech": {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc. (Google)",
        "AMZN": "Amazon.com Inc.",
        "META": "Meta Platforms Inc. (Facebook)",
        "TSLA": "Tesla Inc.",
        "NVDA": "NVIDIA Corporation",
        "AMD": "Advanced Micro Devices",
        "INTC": "Intel Corporation",
        "CRM": "Salesforce Inc.",
        "ORCL": "Oracle Corporation",
        "ADBE": "Adobe Inc.",
        "NFLX": "Netflix Inc.",
        "CSCO": "Cisco Systems",
        "AVGO": "Broadcom Inc.",
        "QCOM": "Qualcomm Inc.",
        "IBM": "IBM Corporation",
        "NOW": "ServiceNow Inc.",
        "UBER": "Uber Technologies",
        "ABNB": "Airbnb Inc.",
    },
    "Actions US - Finance": {
        "JPM": "JPMorgan Chase & Co.",
        "BAC": "Bank of America",
        "WFC": "Wells Fargo",
        "GS": "Goldman Sachs",
        "MS": "Morgan Stanley",
        "C": "Citigroup",
        "AXP": "American Express",
        "V": "Visa Inc.",
        "MA": "Mastercard Inc.",
        "PYPL": "PayPal Holdings",
        "BLK": "BlackRock Inc.",
        "SCHW": "Charles Schwab",
        "COF": "Capital One",
        "USB": "U.S. Bancorp",
    },
    "Actions US - Sante": {
        "JNJ": "Johnson & Johnson",
        "UNH": "UnitedHealth Group",
        "PFE": "Pfizer Inc.",
        "ABBV": "AbbVie Inc.",
        "MRK": "Merck & Co.",
        "LLY": "Eli Lilly",
        "TMO": "Thermo Fisher Scientific",
        "ABT": "Abbott Laboratories",
        "DHR": "Danaher Corporation",
        "BMY": "Bristol-Myers Squibb",
        "AMGN": "Amgen Inc.",
        "GILD": "Gilead Sciences",
        "MRNA": "Moderna Inc.",
        "CVS": "CVS Health",
    },
    "Actions US - Consommation": {
        "WMT": "Walmart Inc.",
        "PG": "Procter & Gamble",
        "KO": "Coca-Cola Company",
        "PEP": "PepsiCo Inc.",
        "COST": "Costco Wholesale",
        "MCD": "McDonald's Corp.",
        "NKE": "Nike Inc.",
        "SBUX": "Starbucks Corp.",
        "HD": "Home Depot",
        "LOW": "Lowe's Companies",
        "TGT": "Target Corporation",
        "DIS": "Walt Disney Co.",
        "CMCSA": "Comcast Corporation",
    },
    "Actions US - Energie et Industrie": {
        "XOM": "Exxon Mobil",
        "CVX": "Chevron Corporation",
        "COP": "ConocoPhillips",
        "SLB": "Schlumberger",
        "OXY": "Occidental Petroleum",
        "BA": "Boeing Company",
        "CAT": "Caterpillar Inc.",
        "GE": "General Electric",
        "HON": "Honeywell International",
        "UPS": "United Parcel Service",
        "FDX": "FedEx Corporation",
        "LMT": "Lockheed Martin",
        "RTX": "RTX Corporation (Raytheon)",
        "DE": "Deere & Company",
    },
    "Actions Francaises (CAC 40)": {
        "MC.PA": "LVMH",
        "OR.PA": "L'Oreal",
        "TTE.PA": "TotalEnergies",
        "SAN.PA": "Sanofi",
        "AIR.PA": "Airbus",
        "SU.PA": "Schneider Electric",
        "AI.PA": "Air Liquide",
        "BNP.PA": "BNP Paribas",
        "ACA.PA": "Credit Agricole",
        "CS.PA": "AXA",
        "DG.PA": "Vinci",
        "CAP.PA": "Capgemini",
        "RI.PA": "Pernod Ricard",
        "KER.PA": "Kering",
        "HO.PA": "Thales",
        "ORA.PA": "Orange",
        "VIE.PA": "Veolia",
        "EN.PA": "Bouygues",
        "DSY.PA": "Dassault Systemes",
        "STM.PA": "STMicroelectronics",
    },
    "Actions Allemandes (DAX)": {
        "SAP.DE": "SAP SE",
        "SIE.DE": "Siemens AG",
        "ALV.DE": "Allianz SE",
        "DTE.DE": "Deutsche Telekom",
        "BAS.DE": "BASF SE",
        "MBG.DE": "Mercedes-Benz",
        "BMW.DE": "BMW AG",
        "VOW3.DE": "Volkswagen AG",
        "ADS.DE": "Adidas AG",
        "MUV2.DE": "Munich Re",
        "DBK.DE": "Deutsche Bank",
        "IFX.DE": "Infineon Technologies",
        "DHL.DE": "Deutsche Post DHL",
        "RWE.DE": "RWE AG",
    },
    "Actions Britanniques (FTSE)": {
        "SHEL.L": "Shell PLC",
        "BP.L": "BP PLC",
        "HSBA.L": "HSBC Holdings",
        "ULVR.L": "Unilever",
        "AZN.L": "AstraZeneca",
        "GSK.L": "GSK PLC",
        "RIO.L": "Rio Tinto",
        "GLEN.L": "Glencore",
        "LLOY.L": "Lloyds Banking",
        "BARC.L": "Barclays",
        "VOD.L": "Vodafone",
        "BA.L": "BAE Systems",
    },
    "Indices Mondiaux": {
        "^GSPC": "S&P 500 (USA)",
        "^DJI": "Dow Jones Industrial (USA)",
        "^IXIC": "NASDAQ Composite (USA)",
        "^RUT": "Russell 2000 (USA)",
        "^VIX": "VIX Volatilite (USA)",
        "^FCHI": "CAC 40 (France)",
        "^GDAXI": "DAX (Allemagne)",
        "^FTSE": "FTSE 100 (UK)",
        "^STOXX50E": "Euro Stoxx 50",
        "^N225": "Nikkei 225 (Japon)",
        "^HSI": "Hang Seng (Hong Kong)",
        "000001.SS": "Shanghai Composite (Chine)",
        "^AXJO": "ASX 200 (Australie)",
        "^BVSP": "Bovespa (Bresil)",
        "^GSPTSE": "S&P/TSX (Canada)",
    },
    "Crypto-monnaies": {
        "BTC-USD": "Bitcoin (BTC)",
        "ETH-USD": "Ethereum (ETH)",
        "BNB-USD": "Binance Coin (BNB)",
        "XRP-USD": "Ripple (XRP)",
        "SOL-USD": "Solana (SOL)",
        "ADA-USD": "Cardano (ADA)",
        "DOGE-USD": "Dogecoin (DOGE)",
        "DOT-USD": "Polkadot (DOT)",
        "MATIC-USD": "Polygon (MATIC)",
        "LTC-USD": "Litecoin (LTC)",
        "SHIB-USD": "Shiba Inu (SHIB)",
        "AVAX-USD": "Avalanche (AVAX)",
        "LINK-USD": "Chainlink (LINK)",
        "ATOM-USD": "Cosmos (ATOM)",
        "UNI-USD": "Uniswap (UNI)",
        "XLM-USD": "Stellar (XLM)",
        "ALGO-USD": "Algorand (ALGO)",
        "VET-USD": "VeChain (VET)",
        "FTM-USD": "Fantom (FTM)",
        "NEAR-USD": "NEAR Protocol (NEAR)",
    },
    "ETFs Populaires": {
        "SPY": "SPDR S&P 500 ETF",
        "QQQ": "Invesco QQQ (NASDAQ 100)",
        "VTI": "Vanguard Total Stock Market",
        "VOO": "Vanguard S&P 500",
        "IWM": "iShares Russell 2000",
        "VGK": "Vanguard FTSE Europe",
        "EFA": "iShares MSCI EAFE",
        "VWO": "Vanguard Emerging Markets",
        "EEM": "iShares MSCI Emerging Markets",
        "GLD": "SPDR Gold Shares",
        "SLV": "iShares Silver Trust",
        "USO": "United States Oil Fund",
        "XLF": "Financial Select Sector SPDR",
        "XLK": "Technology Select Sector SPDR",
        "XLE": "Energy Select Sector SPDR",
        "XLV": "Health Care Select Sector SPDR",
        "ARKK": "ARK Innovation ETF",
        "ARKG": "ARK Genomic Revolution ETF",
        "VNQ": "Vanguard Real Estate ETF",
        "TLT": "iShares 20+ Year Treasury Bond",
    },
    "Matieres Premieres": {
        "GC=F": "Or (Gold Futures)",
        "SI=F": "Argent (Silver Futures)",
        "CL=F": "Petrole Brut WTI",
        "BZ=F": "Petrole Brent",
        "NG=F": "Gaz Naturel",
        "HG=F": "Cuivre",
        "PL=F": "Platine",
        "PA=F": "Palladium",
        "ZC=F": "Mais (Corn)",
        "ZW=F": "Ble (Wheat)",
        "ZS=F": "Soja (Soybean)",
        "KC=F": "Cafe",
        "CC=F": "Cacao",
        "CT=F": "Coton",
    },
    "Devises (Forex)": {
        "EURUSD=X": "EUR/USD",
        "GBPUSD=X": "GBP/USD",
        "USDJPY=X": "USD/JPY",
        "USDCHF=X": "USD/CHF",
        "AUDUSD=X": "AUD/USD",
        "USDCAD=X": "USD/CAD",
        "NZDUSD=X": "NZD/USD",
        "EURGBP=X": "EUR/GBP",
        "EURJPY=X": "EUR/JPY",
        "GBPJPY=X": "GBP/JPY",
        "EURCHF=X": "EUR/CHF",
        "USDMXN=X": "USD/MXN",
        "USDBRL=X": "USD/BRL",
        "USDCNY=X": "USD/CNY",
    },
}
