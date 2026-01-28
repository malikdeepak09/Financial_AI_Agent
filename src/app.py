import os
import re
import logging
import streamlit as st
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.models.anthropic import Claude
from agno.tools.websearch import WebSearchTools

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")

# --------------------------------------------------
# ASSET MAPPING
# --------------------------------------------------

ASSET_MAPPING = {
    "AAPL": "AAPL", "APPLE": "AAPL",
    "TSLA": "TSLA", "TESLA": "TSLA",
    "NVDA": "NVDA", "NVIDIA": "NVDA",
    "MSFT": "MSFT", "MICROSOFT": "MSFT",
    "GOOGL": "GOOGL", "GOOGLE": "GOOGL",
    "AMZN": "AMZN", "AMAZON": "AMZN",
    "META": "META", "FACEBOOK": "META",
    "BTC": "BTC-USD", "BITCOIN": "BTC-USD",
    "ETH": "ETH-USD", "ETHEREUM": "ETH-USD",
    "SPY": "SPY", "S&P": "SPY",
    "QQQ": "QQQ", "NASDAQ": "QQQ",
}

# --------------------------------------------------
# FAST INTENT CLASSIFIER (CACHED)
# --------------------------------------------------

@st.cache_data(ttl=300)
def classify_query_intent(query: str):
    q = query.lower()
    intent = {
        "type": "general",
        "tickers": [],
        "confidence": 0.3
    }

    price_kw = ["price", "cost", "trading", "worth"]
    compare_kw = ["vs", "compare", "versus"]
    analysis_kw = ["analysis", "outlook", "review"]
    news_kw = ["news", "latest", "update"]

    words = re.findall(r"[A-Za-z]+", query.upper())
    for w in words:
        if w in ASSET_MAPPING:
            intent["tickers"].append(ASSET_MAPPING[w])
            intent["confidence"] += 0.3

    if any(k in q for k in price_kw):
        intent["type"] = "price"
    elif any(k in q for k in compare_kw):
        intent["type"] = "comparison"
    elif any(k in q for k in analysis_kw):
        intent["type"] = "analysis"
    elif any(k in q for k in news_kw):
        intent["type"] = "news"

    return intent

# --------------------------------------------------
# ULTRA-FAST MARKET DATA (NO LLM)
# --------------------------------------------------

@st.cache_data(ttl=30)
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.fast_info

    return {
        "symbol": ticker,
        "price": info.last_price,
        "previous_close": info.previous_close,
        "day_high": info.day_high,
        "day_low": info.day_low,
        "volume": info.last_volume,
        "market_cap": info.market_cap,
    }

def fetch_multiple(tickers):
    with ThreadPoolExecutor() as executor:
        return list(executor.map(get_stock_data, tickers))

# --------------------------------------------------
# LLM (ONLY FOR FINAL EXPLANATION)
# --------------------------------------------------

def create_analysis_agent(model_choice):
    if model_choice == "claude-sonnet-4-5":
        model = Claude(id=model_choice, api_key=ANTHROPIC_API_KEY, temperature=0.1)
    else:
        model = OpenAIResponses(
            id=model_choice,
            api_key=OPENAI_API_KEY,
            temperature=0.1,
        )

    return Agent(
        name="Financial Analyst",
        model=model,
        instructions=dedent("""
        You are a professional financial analyst.

        Convert structured market data into a clear explanation.
        Be concise, insightful, and friendly.

        Format:
        ## 📊 Summary
        ## 💰 Key Metrics
        ## 📈 Interpretation

        End with date.
        """),
        markdown=True
    )

@st.cache_resource
def get_web_agent(model_choice):
    model = OpenAIResponses(id=model_choice, api_key=OPENAI_API_KEY, temperature=0.1)
    return Agent(
        name="Web Agent",
        model=model,
        tools=[WebSearchTools(enable_search=True, enable_news=True)],
        markdown=True
    )

# --------------------------------------------------
# CORE LOGIC (FAST)
# --------------------------------------------------

def process_query(query, model_choice):
    intent = classify_query_intent(query)
    tickers = intent["tickers"]

    placeholder = st.empty()
    placeholder.info("⚡ Fetching live market data...")

    # ---- PRICE (INSTANT)
    if intent["type"] == "price" and len(tickers) == 1:
        data = get_stock_data(tickers[0])
        placeholder.success("✅ Live price loaded")

        st.metric(
            label=data["symbol"],
            value=f"${data['price']:.2f}",
            delta=f"{data['price'] - data['previous_close']:.2f}"
        )
        return

    # ---- COMPARISON / ANALYSIS
    if tickers:
        data = fetch_multiple(tickers[:3])
        placeholder.success("✅ Market data loaded")

        agent = create_analysis_agent(model_choice)
        response = agent.run(f"User Query: {query}\n\nMarket Data:\n{data}")
        st.markdown(response.content)
        return

    # ---- NEWS / GENERAL (ONLY NOW WEB SEARCH)
    placeholder.info("🌐 Searching latest market news...")
    web_agent = get_web_agent(model_choice)
    web_results = web_agent.run(query).content

    agent = create_analysis_agent(model_choice)
    final = agent.run(f"User Query: {query}\n\nWeb Results:\n{web_results}")
    st.markdown(final.content)

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

def main():
    st.set_page_config("📈 Fast Financial AI", layout="wide")

    st.title("📈 Fast Financial AI Agent")
    st.caption("⚡ Optimized for speed & great UX")

    with st.sidebar:
        model_choice = st.radio(
            "Model",
            ["claude-sonnet-4-5", "gpt-5.2"],
            index=0
        )

        st.markdown("### 💡 Examples")
        for q in [
            "Apple stock price",
            "Compare TSLA vs NVDA",
            "Bitcoin price today",
            "Tech stocks news"
        ]:
            if st.button(q, use_container_width=True):
                st.session_state.query = q

    query = st.text_input(
        "Ask a financial question",
        value=st.session_state.get("query", ""),
        placeholder="e.g. AAPL price, Compare TSLA vs NVDA"
    )

    if st.button("🚀 Get Answer", type="primary") and query:
        process_query(query, model_choice)
        st.caption(f"🕒 Updated {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
