import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
import time
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight AI | Where Investor Meets Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1117 50%, #0a0e1a 100%);
        color: #e6edf3;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #21262d;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #58a6ff !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #388bfd 0%, #58a6ff 100%);
        box-shadow: 0 0 20px rgba(56,139,253,0.4);
        transform: translateY(-2px);
    }
    
    /* Info boxes */
    .metric-card {
        background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    /* Bull/Bear indicators */
    .bull { color: #3fb950; font-weight: bold; }
    .bear { color: #f85149; font-weight: bold; }
    .neutral { color: #d29922; font-weight: bold; }
    
    /* Divider */
    hr { border-color: #21262d; }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        color: #e6edf3;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #161b22;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #8b949e;
        border-radius: 8px;
    }
    
    .stTabs [aria-selected="true"] {
        background: #1f6feb;
        color: white;
    }

    /* Warning/Success boxes */
    .success-box {
        background: linear-gradient(135deg, #0d1117 0%, #1a2a1a 100%);
        border: 1px solid #3fb950;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #0d1117 0%, #2a2a1a 100%);
        border: 1px solid #d29922;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .danger-box {
        background: linear-gradient(135deg, #0d1117 0%, #2a1a1a 100%);
        border: 1px solid #f85149;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
ASSETS = {
    "📈 FTSE 100": "^FTSE",
    "🏦 HSBC": "HSBA.L",
    "🛢️ BP": "BP.L",
    "💊 AstraZeneca": "AZN.L",
    "🧴 Unilever": "ULVR.L",
    "✈️ Rolls-Royce": "RR.L",
    "💱 GBP/USD": "GBPUSD=X",
    "🥇 Gold": "GC=F",
    "₿ Bitcoin": "BTC-USD",
    "Ξ Ethereum": "ETH-USD"
}

COLORS = {
    'primary': '#58a6ff',
    'success': '#3fb950',
    'warning': '#d29922',
    'danger': '#f85149',
    'purple': '#bc8cff',
    'teal': '#39d353',
    'orange': '#e3b341',
    'grid': '#21262d',
    'bg': '#0d1117',
    'card': '#161b22'
}

# ─── DATA FUNCTIONS ───────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_data(ticker, period="2y"):
    """Fetch financial data from Yahoo Finance"""
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            return None
        # Flatten multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.dropna()
        return data
    except Exception as e:
        return None

def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators"""
    df = df.copy()
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    volume = df['Volume'].squeeze() if 'Volume' in df.columns else pd.Series(index=df.index)

    # Moving Averages
    df['MA_20'] = close.rolling(window=20).mean()
    df['MA_50'] = close.rolling(window=50).mean()
    df['MA_200'] = close.rolling(window=200).mean()

    # Bollinger Bands
    rolling_std = close.rolling(window=20).std()
    df['BB_Upper'] = df['MA_20'] + (rolling_std * 2)
    df['BB_Lower'] = df['MA_20'] - (rolling_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['MA_20']

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Volatility
    df['Returns'] = close.pct_change()
    df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252) * 100

    # Volume indicators
    if not volume.empty and volume.sum() > 0:
        df['Volume_MA'] = volume.rolling(window=20).mean()
        df['Volume_Ratio'] = volume / df['Volume_MA']

    # ATR
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    df['ATR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

    # Momentum
    df['Momentum'] = close / close.shift(10) - 1

    return df

def prepare_lstm_data(data, seq_length=60):
    """Prepare data for LSTM model"""
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    split = int(len(X) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test, scaler, scaled_data

@st.cache_resource
def build_lstm_model(model_type='LSTM', seq_length=60):
    """Build LSTM or GRU model"""
    model = Sequential()

    if model_type == 'LSTM':
        model.add(LSTM(128, return_sequences=True, input_shape=(seq_length, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(0.2))
    elif model_type == 'GRU':
        model.add(GRU(128, return_sequences=True, input_shape=(seq_length, 1)))
        model.add(Dropout(0.2))
        model.add(GRU(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(32, return_sequences=False))
        model.add(Dropout(0.2))
    else:  # Bidirectional LSTM
        model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(seq_length, 1)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(32)))
        model.add(Dropout(0.2))

    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')
    return model

def calculate_var(returns, confidence=0.95):
    """Calculate Value at Risk"""
    if len(returns.dropna()) < 10:
        return 0, 0, 0
    var = np.percentile(returns.dropna(), (1 - confidence) * 100)
    cvar = returns.dropna()[returns.dropna() <= var].mean()
    daily_var = abs(var)
    annual_var = daily_var * np.sqrt(252)
    return var, cvar, annual_var

def detect_market_regime(data, n_clusters=3):
    """Detect market regimes using K-Means + PCA"""
    df = data.copy()
    features = []

    close = df['Close'].squeeze()
    returns = close.pct_change().dropna()

    if len(returns) < 50:
        return None, None, None

    # Feature engineering for clustering
    window = min(20, len(returns) // 3)
    feature_df = pd.DataFrame(index=returns.index)
    feature_df['returns'] = returns
    feature_df['volatility'] = returns.rolling(window).std()
    feature_df['momentum'] = close.pct_change(window)
    feature_df['volume_ratio'] = 1.0  # default

    if 'Volume' in df.columns:
        vol = df['Volume'].squeeze()
        vol_ma = vol.rolling(window).mean()
        feature_df['volume_ratio'] = (vol / vol_ma.replace(0, np.nan)).fillna(1)

    feature_df = feature_df.dropna()

    if len(feature_df) < n_clusters * 10:
        return None, None, None

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_df)

    # PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)

    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    return labels, features_pca, feature_df.index

def detect_anomalies(data, eps=0.5, min_samples=5):
    """Detect price anomalies using DBSCAN"""
    close = data['Close'].squeeze()
    returns = close.pct_change().dropna()

    if len(returns) < 20:
        return None, None

    window = min(20, len(returns) // 3)
    vol = returns.rolling(window).std()
    feature_df = pd.DataFrame({'returns': returns, 'volatility': vol}).dropna()

    if len(feature_df) < 10:
        return None, None

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_df)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features_scaled)

    anomaly_dates = feature_df.index[labels == -1]
    return labels, anomaly_dates

def forecast_prices(model, scaler, scaled_data, forecast_days=30, seq_length=60):
    """Generate price forecast"""
    last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)
    predictions = []

    current_sequence = last_sequence.copy()
    for _ in range(forecast_days):
        pred = model.predict(current_sequence, verbose=0)[0, 0]
        predictions.append(pred)
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = pred

    predictions = np.array(predictions).reshape(-1, 1)
    predictions_actual = scaler.inverse_transform(predictions)
    return predictions_actual.flatten()

# ─── CHART FUNCTIONS ──────────────────────────────────────────────────────────
def create_candlestick_chart(data, ticker_name, show_indicators=True):
    """Create Bloomberg-style candlestick chart"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{ticker_name} Price', 'Volume', 'RSI')
    )

    close = data['Close'].squeeze()
    open_ = data['Open'].squeeze()
    high = data['High'].squeeze()
    low = data['Low'].squeeze()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=open_,
        high=high,
        low=low,
        close=close,
        name='Price',
        increasing_line_color=COLORS['success'],
        decreasing_line_color=COLORS['danger'],
        increasing_fillcolor=COLORS['success'],
        decreasing_fillcolor=COLORS['danger']
    ), row=1, col=1)

    if show_indicators:
        # Moving Averages
        if 'MA_20' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['MA_20'].squeeze(),
                line=dict(color=COLORS['primary'], width=1.5),
                name='MA 20', opacity=0.8
            ), row=1, col=1)

        if 'MA_50' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['MA_50'].squeeze(),
                line=dict(color=COLORS['orange'], width=1.5),
                name='MA 50', opacity=0.8
            ), row=1, col=1)

        # Bollinger Bands
        if 'BB_Upper' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['BB_Upper'].squeeze(),
                line=dict(color=COLORS['purple'], width=1, dash='dash'),
                name='BB Upper', opacity=0.6
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=data.index, y=data['BB_Lower'].squeeze(),
                line=dict(color=COLORS['purple'], width=1, dash='dash'),
                fill='tonexty', fillcolor='rgba(188,140,255,0.05)',
                name='BB Lower', opacity=0.6
            ), row=1, col=1)

    # Volume
    if 'Volume' in data.columns:
        volume = data['Volume'].squeeze()
        colors = [COLORS['success'] if c >= o else COLORS['danger']
                  for c, o in zip(close, open_)]
        fig.add_trace(go.Bar(
            x=data.index, y=volume,
            marker_color=colors, name='Volume', opacity=0.6
        ), row=2, col=1)

    # RSI
    if 'RSI' in data.columns:
        rsi = data['RSI'].squeeze()
        fig.add_trace(go.Scatter(
            x=data.index, y=rsi,
            line=dict(color=COLORS['primary'], width=1.5),
            name='RSI'
        ), row=3, col=1)

        fig.add_hline(y=70, line_dash="dash", line_color=COLORS['danger'],
                      opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=COLORS['success'],
                      opacity=0.5, row=3, col=1)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,17,23,0.8)',
        font=dict(color='#e6edf3', family='Inter'),
        height=600,
        showlegend=True,
        legend=dict(
            bgcolor='rgba(22,27,34,0.8)',
            bordercolor='#21262d',
            borderwidth=1
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis3=dict(gridcolor=COLORS['grid']),
        yaxis=dict(gridcolor=COLORS['grid']),
        yaxis2=dict(gridcolor=COLORS['grid']),
        yaxis3=dict(gridcolor=COLORS['grid'], range=[0, 100])
    )

    return fig

def create_forecast_chart(data, forecast, forecast_days, ticker_name):
    """Create price forecast chart"""
    last_date = data.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=forecast_days,
        freq='B'
    )

    close = data['Close'].squeeze()
    last_90 = data.tail(90)
    last_90_close = last_90['Close'].squeeze()

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=last_90.index,
        y=last_90_close,
        line=dict(color=COLORS['primary'], width=2),
        name='Historical Price',
        mode='lines'
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast,
        line=dict(color=COLORS['success'], width=2.5, dash='dot'),
        name=f'{forecast_days}-Day Forecast',
        mode='lines'
    ))

    # Confidence interval
    std = close.pct_change().std()
    upper_bound = forecast * (1 + std * 2)
    lower_bound = forecast * (1 - std * 2)

    fig.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates[::-1]),
        y=list(upper_bound) + list(lower_bound[::-1]),
        fill='toself',
        fillcolor='rgba(63,185,80,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval'
    ))

    # Vertical line at forecast start
    fig.add_vline(
        x=last_date,
        line_dash="dash",
        line_color=COLORS['warning'],
        opacity=0.7
    )

    fig.add_annotation(
        x=last_date,
        y=close.iloc[-1],
        text="Forecast Start",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS['warning'],
        font=dict(color=COLORS['warning'])
    )

    price_change = ((forecast[-1] - close.iloc[-1]) / close.iloc[-1]) * 100
    direction = "📈" if price_change > 0 else "📉"

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,17,23,0.8)',
        font=dict(color='#e6edf3'),
        height=450,
        title=dict(
            text=f"{direction} {ticker_name} — {forecast_days}-Day AI Forecast | {price_change:+.2f}% Predicted Change",
            font=dict(color=COLORS['primary'], size=16)
        ),
        yaxis=dict(gridcolor=COLORS['grid']),
        xaxis=dict(gridcolor=COLORS['grid']),
        legend=dict(bgcolor='rgba(22,27,34,0.8)', bordercolor='#21262d', borderwidth=1),
        margin=dict(l=10, r=10, t=60, b=10)
    )

    return fig

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: #58a6ff; font-size: 28px; margin: 0;'>⚡ FinSight AI</h1>
        <p style='color: #8b949e; font-size: 13px; margin: 5px 0;'>Where Investor Meets Intelligence</p>
        <div style='background: linear-gradient(90deg, #1f6feb, #388bfd); height: 2px; border-radius: 2px; margin: 10px 0;'></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 Navigation")
    page = st.radio(
        "",
        ["🏠 Market Overview",
         "📈 Price Forecasting",
         "🔍 Anomaly Detection",
         "🎭 Market Regimes",
         "⚠️ Risk Engine",
         "📊 Technical Analysis",
         "🧠 Deep Learning Lab",
         "🔒 FCA & GDPR"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    selected_asset = st.selectbox(
        "Select Asset",
        list(ASSETS.keys()),
        index=0
    )

    period = st.selectbox(
        "Data Period",
        ["6mo", "1y", "2y", "5y"],
        index=2
    )

    forecast_horizon = st.select_slider(
        "Forecast Horizon (Days)",
        options=[7, 14, 30, 60, 90],
        value=30
    )

    model_type = st.selectbox(
        "AI Model",
        ["LSTM", "GRU", "Bidirectional LSTM"],
        index=0
    )

    st.markdown("---")

    # Market status
    now = datetime.now()
    market_open = 8 <= now.hour < 16 and now.weekday() < 5
    status_color = "#3fb950" if market_open else "#f85149"
    status_text = "OPEN" if market_open else "CLOSED"

    st.markdown(f"""
    <div style='background: #161b22; border: 1px solid #21262d; border-radius: 10px; padding: 15px; text-align: center;'>
        <p style='color: #8b949e; margin: 0; font-size: 12px;'>LONDON MARKET</p>
        <p style='color: {status_color}; margin: 5px 0; font-size: 18px; font-weight: bold;'>● {status_text}</p>
        <p style='color: #8b949e; margin: 0; font-size: 11px;'>{now.strftime("%d %b %Y %H:%M")}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center;'>
        <p style='color: #8b949e; font-size: 11px;'>Built by Mrithik Das Raz</p>
        <p style='color: #8b949e; font-size: 11px;'>🇬🇧 UK GDPR Compliant | FCA Aware</p>
        <p style='color: #f85149; font-size: 10px;'>⚠️ Not Financial Advice</p>
    </div>
    """, unsafe_allow_html=True)

# ─── LOAD DATA ───────────────────────────────────────────────────────────────
ticker = ASSETS[selected_asset]

with st.spinner(f"⚡ Fetching {selected_asset} data..."):
    raw_data = fetch_data(ticker, period)

if raw_data is None or raw_data.empty:
    st.error(f"❌ Could not fetch data for {selected_asset}. Please try another asset or check your internet connection.")
    st.stop()

data = calculate_technical_indicators(raw_data.copy())
close = data['Close'].squeeze()

# ─── PAGE: MARKET OVERVIEW ────────────────────────────────────────────────────
if page == "🏠 Market Overview":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #161b22, #1c2128); border: 1px solid #21262d; 
                border-radius: 15px; padding: 25px; margin-bottom: 25px;'>
        <h1 style='color: #58a6ff; margin: 0; font-size: 32px;'>⚡ FinSight AI</h1>
        <p style='color: #8b949e; margin: 5px 0 0 0; font-size: 16px;'>
            Where Investor Meets Intelligence | AI-Powered Financial Market Intelligence Platform
        </p>
        <div style='display: flex; gap: 10px; margin-top: 15px; flex-wrap: wrap;'>
            <span style='background: #1f6feb22; border: 1px solid #1f6feb; border-radius: 20px; 
                        padding: 4px 12px; color: #58a6ff; font-size: 12px;'>🧠 LSTM + GRU Forecasting</span>
            <span style='background: #3fb95022; border: 1px solid #3fb950; border-radius: 20px; 
                        padding: 4px 12px; color: #3fb950; font-size: 12px;'>📊 K-Means Regime Detection</span>
            <span style='background: #f8514922; border: 1px solid #f85149; border-radius: 20px; 
                        padding: 4px 12px; color: #f85149; font-size: 12px;'>🔍 DBSCAN Anomaly Detection</span>
            <span style='background: #bc8cff22; border: 1px solid #bc8cff; border-radius: 20px; 
                        padding: 4px 12px; color: #bc8cff; font-size: 12px;'>📉 PCA Dimensionality Reduction</span>
            <span style='background: #d2992222; border: 1px solid #d29922; border-radius: 20px; 
                        padding: 4px 12px; color: #d29922; font-size: 12px;'>⚠️ VaR Risk Engine</span>
            <span style='background: #39d35322; border: 1px solid #39d353; border-radius: 20px; 
                        padding: 4px 12px; color: #39d353; font-size: 12px;'>🇬🇧 FCA + UK GDPR Compliant</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    current_price = float(close.iloc[-1])
    prev_price = float(close.iloc[-2])
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100

    week_ago = float(close.iloc[-6]) if len(close) > 5 else prev_price
    month_ago = float(close.iloc[-22]) if len(close) > 21 else prev_price
    year_ago = float(close.iloc[-252]) if len(close) > 251 else prev_price

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Current Price",
            f"{'£' if '.L' in ticker else '$'}{current_price:,.2f}",
            f"{price_change_pct:+.2f}% today"
        )
    with col2:
        week_chg = ((current_price - week_ago) / week_ago) * 100
        st.metric("1 Week", f"{week_chg:+.2f}%",
                  "↑ Gaining" if week_chg > 0 else "↓ Losing")
    with col3:
        month_chg = ((current_price - month_ago) / month_ago) * 100
        st.metric("1 Month", f"{month_chg:+.2f}%",
                  "↑ Bullish" if month_chg > 0 else "↓ Bearish")
    with col4:
        year_chg = ((current_price - year_ago) / year_ago) * 100
        st.metric("1 Year", f"{year_chg:+.2f}%",
                  "↑ Strong" if year_chg > 0 else "↓ Weak")
    with col5:
        if 'Volatility' in data.columns:
            vol = float(data['Volatility'].dropna().iloc[-1])
            vol_level = "🔴 High" if vol > 30 else "🟡 Medium" if vol > 15 else "🟢 Low"
            st.metric("Annualised Vol", f"{vol:.1f}%", vol_level)

    st.markdown("---")

    # Main chart
    st.subheader(f"📊 {selected_asset} — Market View")
    fig = create_candlestick_chart(data.tail(180), selected_asset)
    st.plotly_chart(fig, use_container_width=True)

    # Multi-asset overview
    st.markdown("---")
    st.subheader("🌍 Market Dashboard — All Assets")

    market_data = []
    with st.spinner("Loading all market data..."):
        for asset_name, asset_ticker in list(ASSETS.items())[:8]:
            try:
                d = fetch_data(asset_ticker, "5d")
                if d is not None and len(d) >= 2:
                    c = d['Close'].squeeze()
                    cp = float(c.iloc[-1])
                    pp = float(c.iloc[-2])
                    chg = ((cp - pp) / pp) * 100
                    market_data.append({
                        'Asset': asset_name,
                        'Price': cp,
                        'Change %': chg,
                        'Signal': '🟢 BUY' if chg > 0.5 else '🔴 SELL' if chg < -0.5 else '🟡 HOLD'
                    })
            except:
                pass

    if market_data:
        df_market = pd.DataFrame(market_data)
        df_market['Price'] = df_market['Price'].apply(lambda x: f"{x:,.2f}")
        df_market['Change %'] = df_market['Change %'].apply(lambda x: f"{x:+.2f}%")
        st.dataframe(df_market, use_container_width=True, hide_index=True)

# ─── PAGE: PRICE FORECASTING ──────────────────────────────────────────────────
elif page == "📈 Price Forecasting":
    st.markdown(f"## 📈 AI Price Forecasting — {selected_asset}")
    st.markdown(f"**Model:** {model_type} | **Horizon:** {forecast_horizon} days | **Training Data:** {period}")

    st.markdown("""
    <div class='warning-box'>
        <p style='color: #d29922; margin: 0;'>⚠️ <strong>Important Disclaimer:</strong> 
        These forecasts are generated by AI models for educational and research purposes only. 
        They do not constitute financial advice. Past performance does not guarantee future results. 
        Always consult a qualified financial advisor before making investment decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    if len(data) < 100:
        st.error("❌ Insufficient data for forecasting. Please select a longer period.")
        st.stop()

    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown("### ⚙️ Model Config")
        seq_length = st.slider("Sequence Length", 30, 120, 60)
        epochs = st.slider("Training Epochs", 10, 100, 50)
        batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)

        st.markdown("### 📊 Model Architecture")
        if model_type == "LSTM":
            st.info("**LSTM** — Long Short-Term Memory\nBest for long-term patterns")
        elif model_type == "GRU":
            st.info("**GRU** — Gated Recurrent Unit\nFaster, similar accuracy to LSTM")
        else:
            st.info("**Bi-LSTM** — Bidirectional\nCaptures forward & backward patterns")

    with col1:
        if st.button(f"⚡ Train {model_type} & Forecast {forecast_horizon} Days", type="primary"):
            with st.spinner(f"🧠 Training {model_type} model on {len(data)} data points..."):
                try:
                    X_train, X_test, y_train, y_test, scaler, scaled_data = \
                        prepare_lstm_data(data, seq_length)

                    model = build_lstm_model(model_type, seq_length)

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    history_loss = []
                    for epoch in range(epochs):
                        h = model.fit(
                            X_train, y_train,
                            batch_size=batch_size,
                            epochs=1,
                            validation_split=0.1,
                            verbose=0
                        )
                        history_loss.append(h.history['loss'][0])
                        progress = (epoch + 1) / epochs
                        progress_bar.progress(progress)
                        status_text.text(f"Epoch {epoch+1}/{epochs} | Loss: {h.history['loss'][0]:.6f}")

                    progress_bar.empty()
                    status_text.empty()

                    # Generate forecast
                    forecast = forecast_prices(model, scaler, scaled_data, forecast_horizon, seq_length)

                    # Store in session state
                    st.session_state['forecast'] = forecast
                    st.session_state['model_trained'] = True
                    st.session_state['history_loss'] = history_loss

                    st.success(f"✅ {model_type} model trained successfully!")

                except Exception as e:
                    st.error(f"❌ Training error: {str(e)}")

        # Show forecast if available
        if st.session_state.get('model_trained') and st.session_state.get('forecast') is not None:
            forecast = st.session_state['forecast']

            # Forecast chart
            fig = create_forecast_chart(data, forecast, forecast_horizon, selected_asset)
            st.plotly_chart(fig, use_container_width=True)

            # Forecast metrics
            current_price = float(close.iloc[-1])
            predicted_end = float(forecast[-1])
            predicted_change = ((predicted_end - current_price) / current_price) * 100
            predicted_high = float(np.max(forecast))
            predicted_low = float(np.min(forecast))

            c1, c2, c3, c4 = st.columns(4)
            currency = '£' if '.L' in ticker else '$'
            with c1:
                st.metric("Current Price", f"{currency}{current_price:,.2f}")
            with c2:
                st.metric(f"Day {forecast_horizon} Target",
                          f"{currency}{predicted_end:,.2f}",
                          f"{predicted_change:+.2f}%")
            with c3:
                st.metric("Forecast High", f"{currency}{predicted_high:,.2f}")
            with c4:
                st.metric("Forecast Low", f"{currency}{predicted_low:,.2f}")

            # Training loss chart
            if st.session_state.get('history_loss'):
                st.markdown("### 📉 Training Loss Curve")
                loss_fig = px.line(
                    y=st.session_state['history_loss'],
                    title="Model Training Loss (Lower = Better)",
                    labels={'x': 'Epoch', 'y': 'Loss'},
                    template='plotly_dark'
                )
                loss_fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(13,17,23,0.8)',
                    height=300
                )
                st.plotly_chart(loss_fig, use_container_width=True)

            # Forecast table
            last_date = data.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_horizon, freq='B'
            )
            forecast_df = pd.DataFrame({
                'Date': forecast_dates.strftime('%d %b %Y'),
                'Predicted Price': [f"{currency}{p:,.2f}" for p in forecast],
                'Change from Today': [f"{((p - current_price)/current_price)*100:+.2f}%" for p in forecast],
                'Direction': ['📈' if p > current_price else '📉' for p in forecast]
            })
            st.markdown("### 📋 Forecast Table")
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        else:
            st.info("👆 Configure your model settings and click **Train & Forecast** to begin.")

            # Show historical data while waiting
            fig = create_candlestick_chart(data.tail(120), selected_asset)
            st.plotly_chart(fig, use_container_width=True)

# ─── PAGE: ANOMALY DETECTION ──────────────────────────────────────────────────
elif page == "🔍 Anomaly Detection":
    st.markdown(f"## 🔍 Anomaly Detection — {selected_asset}")
    st.markdown("**Algorithm:** DBSCAN (Density-Based Spatial Clustering of Applications with Noise)")

    st.markdown("""
    <div class='metric-card'>
        <h4 style='color: #58a6ff;'>What is DBSCAN Anomaly Detection?</h4>
        <p style='color: #8b949e;'>DBSCAN identifies anomalies as data points that do not fit into any cluster — 
        these are labelled as noise (outliers). In financial markets, these represent unusual price movements 
        that deviate significantly from normal behaviour — potential flash crashes, manipulation events, 
        or major news impacts.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        eps = st.slider("DBSCAN Epsilon", 0.1, 2.0, 0.5, 0.1,
                        help="Maximum distance between points in a cluster")
        min_samples = st.slider("Min Samples", 2, 20, 5,
                                help="Minimum points to form a cluster")

    labels, anomaly_dates = detect_anomalies(data, eps, min_samples)

    if labels is not None:
        n_anomalies = sum(labels == -1)
        n_normal = sum(labels != -1)

        with col1:
            st.markdown("---")
            st.metric("🔴 Anomalies Found", n_anomalies)
            st.metric("🟢 Normal Points", n_normal)
            anomaly_rate = (n_anomalies / len(labels)) * 100
            st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")

        with col2:
            fig = go.Figure()

            close_series = close.squeeze()

            # Normal prices
            fig.add_trace(go.Scatter(
                x=data.index,
                y=close_series,
                mode='lines',
                line=dict(color=COLORS['primary'], width=1.5),
                name='Price',
                opacity=0.8
            ))

            # Anomalies
            if len(anomaly_dates) > 0:
                anomaly_prices = close_series[close_series.index.isin(anomaly_dates)]
                fig.add_trace(go.Scatter(
                    x=anomaly_dates,
                    y=anomaly_prices,
                    mode='markers',
                    marker=dict(
                        color=COLORS['danger'],
                        size=10,
                        symbol='x',
                        line=dict(color='white', width=1)
                    ),
                    name=f'🔴 Anomalies ({n_anomalies})'
                ))

            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(13,17,23,0.8)',
                height=400,
                title=f"DBSCAN Anomaly Detection — {n_anomalies} anomalies detected",
                font=dict(color='#e6edf3'),
                yaxis=dict(gridcolor=COLORS['grid']),
                xaxis=dict(gridcolor=COLORS['grid']),
                margin=dict(l=10, r=10, t=50, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        # Returns scatter with anomalies
        st.markdown("### 📊 Returns Distribution with Anomalies")
        returns = close.pct_change().dropna()
        vol = returns.rolling(20).std().dropna()
        common_idx = returns.index.intersection(vol.index)
        returns_common = returns[common_idx]
        vol_common = vol[common_idx]

        if len(anomaly_dates) > 0:
            anomaly_mask = common_idx.isin(anomaly_dates)
            normal_mask = ~anomaly_mask

            scatter_fig = go.Figure()
            scatter_fig.add_trace(go.Scatter(
                x=vol_common[normal_mask],
                y=returns_common[normal_mask],
                mode='markers',
                marker=dict(color=COLORS['primary'], size=4, opacity=0.5),
                name='Normal'
            ))
            scatter_fig.add_trace(go.Scatter(
                x=vol_common[anomaly_mask],
                y=returns_common[anomaly_mask],
                mode='markers',
                marker=dict(color=COLORS['danger'], size=10, symbol='x'),
                name='Anomaly'
            ))

            scatter_fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(13,17,23,0.8)',
                height=350,
                title="Volatility vs Returns — Anomaly Scatter",
                xaxis_title="Rolling Volatility",
                yaxis_title="Daily Returns",
                font=dict(color='#e6edf3'),
                margin=dict(l=10, r=10, t=50, b=10)
            )
            st.plotly_chart(scatter_fig, use_container_width=True)

        # Anomaly table
        if len(anomaly_dates) > 0:
            st.markdown("### 📋 Detected Anomalies")
            anomaly_df = pd.DataFrame({
                'Date': anomaly_dates.strftime('%d %b %Y'),
                'Price': [f"{float(close[d]):,.2f}" for d in anomaly_dates if d in close.index],
                'Daily Return': [f"{float(returns[d])*100:+.2f}%" for d in anomaly_dates if d in returns.index],
                'Severity': ['🔴 High' if abs(float(returns.get(d, 0))) > 0.03 else '🟡 Medium'
                             for d in anomaly_dates if d in returns.index]
            })
            st.dataframe(anomaly_df, use_container_width=True, hide_index=True)

# ─── PAGE: MARKET REGIMES ─────────────────────────────────────────────────────
elif page == "🎭 Market Regimes":
    st.markdown(f"## 🎭 Market Regime Detection — {selected_asset}")
    st.markdown("**Algorithms:** K-Means Clustering + PCA Dimensionality Reduction")

    st.markdown("""
    <div class='metric-card'>
        <h4 style='color: #58a6ff;'>What are Market Regimes?</h4>
        <p style='color: #8b949e;'>Markets operate in distinct regimes — Bull (upward trending), 
        Bear (downward trending), and Sideways (consolidating). K-Means clustering identifies these 
        regimes automatically from price patterns, volatility, and momentum. PCA reduces the high-dimensional 
        feature space to 2D for visualisation while preserving variance.</p>
    </div>
    """, unsafe_allow_html=True)

    n_clusters = st.slider("Number of Market Regimes", 2, 5, 3)

    labels, features_pca, feature_dates = detect_market_regime(data, n_clusters)

    if labels is not None:
        regime_colors = [COLORS['success'], COLORS['danger'], COLORS['warning'],
                         COLORS['purple'], COLORS['primary']]
        regime_names = ['Bull Market 📈', 'Bear Market 📉', 'Sideways/Consolidation ↔️',
                        'High Volatility ⚡', 'Recovery Phase 🔄']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🗺️ PCA Regime Map")
            pca_fig = go.Figure()
            for i in range(n_clusters):
                mask = labels == i
                pca_fig.add_trace(go.Scatter(
                    x=features_pca[mask, 0],
                    y=features_pca[mask, 1],
                    mode='markers',
                    marker=dict(color=regime_colors[i], size=6, opacity=0.7),
                    name=regime_names[i] if i < len(regime_names) else f'Regime {i+1}'
                ))

            pca_fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(13,17,23,0.8)',
                height=400,
                title="Market Regimes in PCA Space",
                xaxis_title="Principal Component 1",
                yaxis_title="Principal Component 2",
                font=dict(color='#e6edf3'),
                margin=dict(l=10, r=10, t=50, b=10)
            )
            st.plotly_chart(pca_fig, use_container_width=True)

        with col2:
            st.markdown("### 📊 Regime Distribution")
            regime_counts = pd.Series(labels).value_counts().sort_index()
            pie_fig = px.pie(
                values=regime_counts.values,
                names=[regime_names[i] if i < len(regime_names) else f'Regime {i+1}'
                       for i in regime_counts.index],
                color_discrete_sequence=regime_colors[:n_clusters],
                template='plotly_dark'
            )
            pie_fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                height=400,
                font=dict(color='#e6edf3'),
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(pie_fig, use_container_width=True)

        # Price chart with regime overlay
        st.markdown("### 📈 Price Chart with Regime Overlay")
        regime_fig = go.Figure()

        close_series = close.squeeze()
        regime_fig.add_trace(go.Scatter(
            x=data.index,
            y=close_series,
            mode='lines',
            line=dict(color='#8b949e', width=1),
            name='Price',
            opacity=0.5
        ))

        if feature_dates is not None and len(feature_dates) == len(labels):
            for i in range(n_clusters):
                mask = labels == i
                regime_dates = feature_dates[mask]
                regime_prices = close_series[close_series.index.isin(regime_dates)]
                regime_fig.add_trace(go.Scatter(
                    x=regime_dates,
                    y=regime_prices,
                    mode='markers',
                    marker=dict(color=regime_colors[i], size=5, opacity=0.8),
                    name=regime_names[i] if i < len(regime_names) else f'Regime {i+1}'
                ))

        regime_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,17,23,0.8)',
            height=400,
            title=f"Market Regimes Over Time — {selected_asset}",
            font=dict(color='#e6edf3'),
            yaxis=dict(gridcolor=COLORS['grid']),
            xaxis=dict(gridcolor=COLORS['grid']),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(regime_fig, use_container_width=True)

        # Current regime
        current_regime = labels[-1]
        current_regime_name = regime_names[current_regime] if current_regime < len(regime_names) else f'Regime {current_regime+1}'
        st.markdown(f"""
        <div class='success-box'>
            <h4 style='color: #3fb950; margin: 0;'>Current Market Regime</h4>
            <p style='color: #e6edf3; font-size: 24px; margin: 10px 0;'>{current_regime_name}</p>
            <p style='color: #8b949e; margin: 0;'>Based on recent price action, volatility, and momentum analysis</p>
        </div>
        """, unsafe_allow_html=True)

# ─── PAGE: RISK ENGINE ────────────────────────────────────────────────────────
elif page == "⚠️ Risk Engine":
    st.markdown(f"## ⚠️ Risk Engine — {selected_asset}")

    returns = close.pct_change().dropna()

    if len(returns) < 30:
        st.error("❌ Insufficient data for risk analysis.")
        st.stop()

    # VaR calculations
    var_95, cvar_95, annual_var_95 = calculate_var(returns, 0.95)
    var_99, cvar_99, annual_var_99 = calculate_var(returns, 0.99)

    current_price = float(close.iloc[-1])
    currency = '£' if '.L' in ticker else '$'

    # Risk metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("VaR (95%)", f"{abs(var_95)*100:.2f}%",
                  f"{currency}{abs(var_95)*current_price:,.0f} per unit")
    with col2:
        st.metric("VaR (99%)", f"{abs(var_99)*100:.2f}%",
                  f"{currency}{abs(var_99)*current_price:,.0f} per unit")
    with col3:
        st.metric("CVaR (95%)", f"{abs(cvar_95)*100:.2f}%",
                  "Expected Shortfall")
    with col4:
        annual_vol = float(returns.std()) * np.sqrt(252) * 100
        st.metric("Annual Volatility", f"{annual_vol:.1f}%",
                  "🔴 High" if annual_vol > 30 else "🟡 Medium" if annual_vol > 15 else "🟢 Low")

    st.markdown("---")

    # Portfolio simulator
    st.markdown("### 💼 Portfolio Risk Simulator")
    col1, col2, col3 = st.columns(3)
    with col1:
        investment = st.number_input(
            f"Investment Amount ({currency})",
            min_value=100.0, max_value=1000000.0, value=10000.0, step=500.0
        )
    with col2:
        holding_period = st.selectbox("Holding Period", ["1 Day", "1 Week", "1 Month", "3 Months"])
    with col3:
        confidence_level = st.selectbox("Confidence Level", ["95%", "99%"], index=0)

    period_multiplier = {"1 Day": 1, "1 Week": 5, "1 Month": 21, "3 Months": 63}[holding_period]
    conf = 0.95 if confidence_level == "95%" else 0.99
    var_val, cvar_val, _ = calculate_var(returns, conf)

    portfolio_var = abs(var_val) * investment * np.sqrt(period_multiplier)
    portfolio_cvar = abs(cvar_val) * investment * np.sqrt(period_multiplier)
    max_loss = investment * abs(returns.min())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='danger-box'>
            <h4 style='color: #f85149;'>Value at Risk ({confidence_level})</h4>
            <p style='color: #e6edf3; font-size: 28px; font-weight: bold;'>{currency}{portfolio_var:,.0f}</p>
            <p style='color: #8b949e;'>Maximum expected loss over {holding_period} with {confidence_level} confidence</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='warning-box'>
            <h4 style='color: #d29922;'>Expected Shortfall</h4>
            <p style='color: #e6edf3; font-size: 28px; font-weight: bold;'>{currency}{portfolio_cvar:,.0f}</p>
            <p style='color: #8b949e;'>Average loss when VaR is exceeded (tail risk)</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <h4 style='color: #58a6ff;'>Historical Max Loss</h4>
            <p style='color: #e6edf3; font-size: 28px; font-weight: bold;'>{currency}{max_loss:,.0f}</p>
            <p style='color: #8b949e;'>Worst single-day loss in history (on {currency}{investment:,.0f} investment)</p>
        </div>
        """, unsafe_allow_html=True)

    # Returns distribution
    st.markdown("### 📊 Returns Distribution")
    hist_fig = go.Figure()

    hist_fig.add_trace(go.Histogram(
        x=returns * 100,
        nbinsx=80,
        marker_color=COLORS['primary'],
        opacity=0.7,
        name='Daily Returns'
    ))

    # VaR lines
    hist_fig.add_vline(x=var_95 * 100, line_dash="dash",
                       line_color=COLORS['warning'],
                       annotation_text="VaR 95%",
                       annotation_font_color=COLORS['warning'])
    hist_fig.add_vline(x=var_99 * 100, line_dash="dash",
                       line_color=COLORS['danger'],
                       annotation_text="VaR 99%",
                       annotation_font_color=COLORS['danger'])

    hist_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,17,23,0.8)',
        height=350,
        title="Daily Returns Distribution with VaR Lines",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        font=dict(color='#e6edf3'),
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    # Rolling VaR
    st.markdown("### 📈 Rolling 30-Day VaR")
    rolling_var = returns.rolling(30).quantile(0.05) * 100

    rolling_fig = go.Figure()
    rolling_fig.add_trace(go.Scatter(
        x=data.index,
        y=rolling_var,
        fill='tozeroy',
        fillcolor='rgba(248,81,73,0.2)',
        line=dict(color=COLORS['danger'], width=1.5),
        name='Rolling VaR (95%)'
    ))

    rolling_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,17,23,0.8)',
        height=300,
        title="Rolling 30-Day Value at Risk (95%)",
        font=dict(color='#e6edf3'),
        yaxis=dict(gridcolor=COLORS['grid'], title="VaR (%)"),
        xaxis=dict(gridcolor=COLORS['grid']),
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(rolling_fig, use_container_width=True)

# ─── PAGE: TECHNICAL ANALYSIS ─────────────────────────────────────────────────
elif page == "📊 Technical Analysis":
    st.markdown(f"## 📊 Technical Analysis — {selected_asset}")

    tabs = st.tabs(["📈 Price & MAs", "📉 MACD", "💪 RSI", "🎯 Bollinger Bands", "📊 Volatility"])

    with tabs[0]:
        fig = create_candlestick_chart(data.tail(252), selected_asset, show_indicators=True)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        if 'MACD' in data.columns:
            macd_fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                     vertical_spacing=0.05, row_heights=[0.6, 0.4])
            close_series = close.squeeze()
            macd_fig.add_trace(go.Scatter(
                x=data.index, y=close_series,
                line=dict(color=COLORS['primary'], width=1.5), name='Price'
            ), row=1, col=1)

            macd = data['MACD'].squeeze()
            macd_signal = data['MACD_Signal'].squeeze()
            macd_hist = data['MACD_Hist'].squeeze()

            macd_fig.add_trace(go.Scatter(
                x=data.index, y=macd,
                line=dict(color=COLORS['primary'], width=1.5), name='MACD'
            ), row=2, col=1)
            macd_fig.add_trace(go.Scatter(
                x=data.index, y=macd_signal,
                line=dict(color=COLORS['warning'], width=1.5), name='Signal'
            ), row=2, col=1)

            colors_hist = [COLORS['success'] if v >= 0 else COLORS['danger'] for v in macd_hist]
            macd_fig.add_trace(go.Bar(
                x=data.index, y=macd_hist,
                marker_color=colors_hist, name='Histogram', opacity=0.7
            ), row=2, col=1)

            macd_fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(13,17,23,0.8)',
                height=500, font=dict(color='#e6edf3'),
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(macd_fig, use_container_width=True)

    with tabs[2]:
        if 'RSI' in data.columns:
            rsi_series = data['RSI'].squeeze()
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(
                x=data.index, y=rsi_series,
                line=dict(color=COLORS['primary'], width=2), name='RSI',
                fill='tozeroy', fillcolor='rgba(88,166,255,0.1)'
            ))
            rsi_fig.add_hline(y=70, line_dash="dash", line_color=COLORS['danger'],
                              annotation_text="Overbought (70)")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color=COLORS['success'],
                              annotation_text="Oversold (30)")
            rsi_fig.add_hline(y=50, line_dash="dot", line_color='#8b949e', opacity=0.5)

            current_rsi = float(rsi_series.dropna().iloc[-1])
            rsi_signal = "🔴 OVERBOUGHT" if current_rsi > 70 else "🟢 OVERSOLD" if current_rsi < 30 else "🟡 NEUTRAL"
            st.metric(f"Current RSI: {current_rsi:.1f}", rsi_signal)

            rsi_fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(13,17,23,0.8)',
                height=400, font=dict(color='#e6edf3'),
                yaxis=dict(gridcolor=COLORS['grid'], range=[0, 100]),
                xaxis=dict(gridcolor=COLORS['grid']),
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(rsi_fig, use_container_width=True)

    with tabs[3]:
        if 'BB_Upper' in data.columns:
            bb_fig = go.Figure()
            close_series = close.squeeze()
            bb_upper = data['BB_Upper'].squeeze()
            bb_lower = data['BB_Lower'].squeeze()
            ma_20 = data['MA_20'].squeeze()

            bb_fig.add_trace(go.Scatter(
                x=data.index, y=bb_upper,
                line=dict(color=COLORS['purple'], width=1, dash='dash'),
                name='Upper Band'
            ))
            bb_fig.add_trace(go.Scatter(
                x=data.index, y=bb_lower,
                line=dict(color=COLORS['purple'], width=1, dash='dash'),
                fill='tonexty', fillcolor='rgba(188,140,255,0.08)',
                name='Lower Band'
            ))
            bb_fig.add_trace(go.Scatter(
                x=data.index, y=ma_20,
                line=dict(color=COLORS['warning'], width=1.5),
                name='MA 20'
            ))
            bb_fig.add_trace(go.Scatter(
                x=data.index, y=close_series,
                line=dict(color=COLORS['primary'], width=2),
                name='Price'
            ))

            bb_fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(13,17,23,0.8)',
                height=450, font=dict(color='#e6edf3'),
                yaxis=dict(gridcolor=COLORS['grid']),
                xaxis=dict(gridcolor=COLORS['grid']),
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(bb_fig, use_container_width=True)

    with tabs[4]:
        if 'Volatility' in data.columns:
            vol_series = data['Volatility'].dropna().squeeze()
            vol_fig = go.Figure()
            vol_fig.add_trace(go.Scatter(
                x=vol_series.index, y=vol_series,
                fill='tozeroy', fillcolor='rgba(210,153,34,0.2)',
                line=dict(color=COLORS['warning'], width=2),
                name='Annualised Volatility (%)'
            ))

            avg_vol = float(vol_series.mean())
            vol_fig.add_hline(y=avg_vol, line_dash="dash",
                              line_color=COLORS['primary'],
                              annotation_text=f"Average: {avg_vol:.1f}%")

            vol_fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(13,17,23,0.8)',
                height=400,
                title="Rolling 20-Day Annualised Volatility",
                font=dict(color='#e6edf3'),
                yaxis=dict(gridcolor=COLORS['grid'], title="Volatility (%)"),
                xaxis=dict(gridcolor=COLORS['grid']),
                margin=dict(l=10, r=10, t=50, b=10)
            )
            st.plotly_chart(vol_fig, use_container_width=True)

# ─── PAGE: DEEP LEARNING LAB ──────────────────────────────────────────────────
elif page == "🧠 Deep Learning Lab":
    st.markdown("## 🧠 Deep Learning Lab — Model Architecture Explorer")

    tabs = st.tabs(["🏗️ LSTM Architecture", "🔄 GRU Architecture",
                    "↔️ Bidirectional LSTM", "📚 How It Works"])

    with tabs[0]:
        st.markdown("""
        ### LSTM — Long Short-Term Memory
        
        LSTM networks are designed to learn long-term dependencies in sequential data. 
        They solve the vanishing gradient problem through their gating mechanisms.
        
        **Architecture in FinSight AI:**
        ```
        Input Layer    → (seq_length, 1) — price sequence
        LSTM Layer 1   → 128 units, return_sequences=True
        Dropout        → 0.2 (prevents overfitting)
        LSTM Layer 2   → 64 units, return_sequences=True  
        Dropout        → 0.2
        LSTM Layer 3   → 32 units, return_sequences=False
        Dropout        → 0.2
        Dense Layer    → 16 units, ReLU activation
        Output Layer   → 1 unit (price prediction)
        ```
        
        **Key Components:**
        - **Forget Gate:** Decides what information to discard from cell state
        - **Input Gate:** Decides what new information to store
        - **Output Gate:** Decides what to output based on cell state
        - **Cell State:** Long-term memory highway
        
        **Why LSTM for Finance?**
        - Financial data has long-term dependencies (market cycles)
        - Price patterns from months ago can influence current prices
        - LSTM remembers important patterns and forgets noise
        """)

    with tabs[1]:
        st.markdown("""
        ### GRU — Gated Recurrent Unit
        
        GRU is a simplified version of LSTM with fewer parameters, 
        making it faster to train while achieving similar performance.
        
        **Architecture in FinSight AI:**
        ```
        Input Layer    → (seq_length, 1)
        GRU Layer 1    → 128 units, return_sequences=True
        Dropout        → 0.2
        GRU Layer 2    → 64 units, return_sequences=True
        Dropout        → 0.2
        GRU Layer 3    → 32 units, return_sequences=False
        Dropout        → 0.2
        Dense Layer    → 16 units, ReLU
        Output Layer   → 1 unit
        ```
        
        **GRU vs LSTM:**
        - GRU combines forget and input gates into a single update gate
        - Fewer parameters → faster training
        - Often comparable accuracy to LSTM
        - Better for shorter sequences
        """)

    with tabs[2]:
        st.markdown("""
        ### Bidirectional LSTM
        
        Processes sequences both forward AND backward, capturing patterns 
        from both past and future context.
        
        **Architecture in FinSight AI:**
        ```
        Input Layer          → (seq_length, 1)
        Bidirectional LSTM 1 → 64 units each direction (128 total)
        Dropout              → 0.2
        Bidirectional LSTM 2 → 32 units each direction (64 total)
        Dropout              → 0.2
        Dense Layer          → 16 units, ReLU
        Output Layer         → 1 unit
        ```
        
        **When to use Bidirectional:**
        - When future context helps understand current state
        - Time-series analysis where patterns exist in both directions
        - Generally slightly higher accuracy than standard LSTM
        """)

    with tabs[3]:
        st.markdown("""
        ### 📚 How FinSight AI's Forecasting Works
        
        **Step 1: Data Collection**
        - Real-time data from Yahoo Finance API
        - FTSE 100, UK stocks, GBP/USD, Gold, Crypto
        
        **Step 2: Preprocessing**
        - MinMaxScaler normalises prices to [0, 1]
        - Creates sliding window sequences of length 60
        - 85% training, 15% testing split
        
        **Step 3: Model Training**
        - LSTM/GRU learns price patterns
        - Adam optimiser with learning rate 0.001
        - Huber loss (robust to outliers)
        - Early stopping prevents overfitting
        
        **Step 4: Forecasting**
        - Model predicts next price from last 60 days
        - Prediction fed back as input for next step
        - Process repeats for forecast_days iterations
        
        **Step 5: Uncertainty Quantification**
        - Confidence intervals based on historical volatility
        - 95% confidence band shown around forecast
        
        **Limitations & Disclaimers:**
        - Cannot predict black swan events
        - Performance varies by market conditions
        - Not financial advice
        - Past patterns may not repeat
        """)

# ─── PAGE: FCA & GDPR ─────────────────────────────────────────────────────────
elif page == "🔒 FCA & GDPR":
    st.markdown("## 🔒 FCA & UK GDPR Compliance Framework")

    tabs = st.tabs(["🏛️ FCA Compliance", "🔒 UK GDPR", "⚠️ Disclaimers", "📋 About"])

    with tabs[0]:
        st.markdown("""
        ### 🏛️ Financial Conduct Authority (FCA) — Compliance Framework
        
        **FinSight AI operates with full awareness of FCA regulations governing AI in financial services.**
        
        #### FCA AI Principles Applied:
        
        **1. Safety and Soundness**
        - All model outputs are clearly labelled as AI-generated
        - Uncertainty quantification provided for all forecasts
        - Multiple model options to avoid single-point-of-failure
        
        **2. Consumer Protection**
        - Clear disclaimers on every forecast page
        - Risk warnings prominently displayed
        - VaR and CVaR clearly explained in plain English
        
        **3. Market Integrity**
        - No automated trading recommendations
        - All outputs require human review and decision-making
        - Anomaly detection supports market surveillance, not manipulation
        
        **4. Transparency**
        - Full model architecture documented in Deep Learning Lab
        - Training methodology explained
        - Limitations clearly stated
        
        **5. Accountability**
        - Human oversight required for all investment decisions
        - Clear audit trail of forecasts
        - Model version tracked
        
        #### MiFID II Awareness:
        - Investment research classification considered
        - No personalised investment recommendations made
        - Risk disclosure requirements met
        
        #### Senior Managers & Certification Regime (SM&CR):
        - Clear accountability structure documented
        - Model risk management framework in place
        - Regular model validation recommended
        """)

    with tabs[1]:
        st.markdown("""
        ### 🔒 UK GDPR Compliance — Data Protection Framework
        
        **FinSight AI is designed with Privacy by Design (Article 25 UK GDPR)**
        
        #### Data Processed by FinSight AI:
        
        | Data Type | Source | Personal Data? | Retention |
        |-----------|--------|----------------|-----------|
        | Market prices | Yahoo Finance API | No | Session only |
        | Trading volumes | Yahoo Finance API | No | Session only |
        | User inputs | Browser session | Minimal | Session only |
        | Model predictions | Generated locally | No | Session only |
        
        #### Article 5 — Data Processing Principles:
        
        **Lawfulness:** All market data obtained from public APIs under their terms of service
        
        **Purpose Limitation:** Data used exclusively for market analysis and visualisation
        
        **Data Minimisation:** Only price/volume data fetched — no personal financial data
        
        **Accuracy:** Real-time data from authoritative source (Yahoo Finance)
        
        **Storage Limitation:** No data persisted beyond the browser session
        
        **Security:** All processing occurs locally — no data transmitted to third parties beyond Yahoo Finance API
        
        #### Privacy by Design (Article 25):
        - ✅ No personal data collected
        - ✅ No user accounts required
        - ✅ No data stored beyond session
        - ✅ No third-party tracking
        - ✅ All AI processing happens locally
        
        #### Your Rights:
        Since no personal data is processed, standard data subject rights (access, erasure, portability) 
        are not applicable. However, users can clear all session data by refreshing the browser.
        """)

    with tabs[2]:
        st.markdown("""
        ### ⚠️ Important Disclaimers
        """)

        st.error("""
        **NOT FINANCIAL ADVICE**
        
        FinSight AI is an educational and research tool only. Nothing in this application 
        constitutes financial advice, investment recommendations, or solicitation to buy or sell 
        any financial instrument. All forecasts are generated by AI models and carry significant 
        uncertainty.
        """)

        st.warning("""
        **INVESTMENT RISK WARNING**
        
        Investing in financial markets involves risk. You may lose some or all of your invested capital. 
        Past performance of AI models does not guarantee future results. Market conditions can change 
        rapidly in ways that AI models cannot predict. Always consult a qualified financial adviser 
        before making investment decisions.
        """)

        st.info("""
        **AI MODEL LIMITATIONS**
        
        - LSTM/GRU models are trained on historical data and cannot predict unforeseen events
        - Model accuracy varies by market conditions and asset class
        - Forecasts become less reliable further into the future
        - Anomaly detection may produce false positives/negatives
        - Market regime detection is probabilistic, not deterministic
        """)

    with tabs[3]:
        st.markdown("""
        ### 📋 About FinSight AI
        
        **FinSight AI — Where Investor Meets Intelligence**
        
        Built as part of a professional AI & Data Science portfolio targeting the UK market.
        
        #### Technical Stack:
        - **Deep Learning:** TensorFlow 2.x, Keras — LSTM, GRU, Bidirectional LSTM
        - **Machine Learning:** scikit-learn — K-Means, DBSCAN, PCA
        - **Data:** Yahoo Finance API (yfinance) — real-time market data
        - **Visualisation:** Plotly — interactive Bloomberg-style charts
        - **Framework:** Streamlit — production-ready web application
        - **Risk:** Custom VaR/CVaR engine
        
        #### Curriculum Coverage:
        - ✅ Module 8: RNNs, LSTMs, GRUs
        - ✅ Module 8: Time-Series Forecasting
        - ✅ Module 7: K-Means Clustering
        - ✅ Module 7: DBSCAN Anomaly Detection
        - ✅ Module 7: PCA Dimensionality Reduction
        - ✅ Module 7: Unsupervised Learning
        
        #### Built by:
        **Mrithik Das Raz** | MSc Computer Science (Network Engineering)
        University of Greenwich | IBM Certified Data Analytics Professional
        
        🔗 GitHub: github.com/Moharaz01
        💼 LinkedIn: linkedin.com/in/mdrmrithik01
        """)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #8b949e; padding: 20px;'>
    <p style='font-size: 14px;'>⚡ <strong style='color: #58a6ff;'>FinSight AI</strong> — Where Investor Meets Intelligence</p>
    <p style='font-size: 12px;'>Built by Mrithik Das Raz | MSc Computer Science | IBM Certified | UK GDPR Compliant | FCA Aware</p>
    <p style='font-size: 11px; color: #f85149;'>⚠️ For educational and research purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)