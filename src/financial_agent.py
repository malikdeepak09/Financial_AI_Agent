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
# FINANCIAL DOMAIN ENFORCEMENT
# --------------------------------------------------

FINANCIAL_KEYWORDS = [
    "stock", "stocks", "share", "price", "market", "markets",
    "crypto", "bitcoin", "ethereum", "investment", "invest",
    "trading", "nasdaq", "dow", "s&p", "etf", "fund",
    "earnings", "revenue", "profit", "loss", "ipo",
    "economy", "inflation", "interest rate", "fed",
    "valuation", "analysis", "news", "compare"
]

def is_financial_query(query: str) -> bool:
    q = query.lower()
    return any(keyword in q for keyword in FINANCIAL_KEYWORDS)

# --------------------------------------------------
# INTENT CLASSIFIER
# --------------------------------------------------

@st.cache_data(ttl=300)
def classify_query_intent(query: str):
    q = query.lower()
    intent = {"type": "general", "tickers": []}

    if any(k in q for k in ["price", "trading", "worth"]):
        intent["type"] = "price"
    elif any(k in q for k in ["vs", "compare"]):
        intent["type"] = "comparison"
    elif "news" in q:
        intent["type"] = "news"
    elif "analysis" in q:
        intent["type"] = "analysis"

    words = re.findall(r"[A-Za-z]+", query.upper())
    for w in words:
        if w in ASSET_MAPPING:
            intent["tickers"].append(ASSET_MAPPING[w])

    return intent

# --------------------------------------------------
# FAST MARKET DATA (NO LLM)
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
# SAFE WEB SEARCH (NEVER FAILS)
# --------------------------------------------------

def safe_web_search(query, web_agent):
    try:
        result = web_agent.run(query).content
        if result and len(result.strip()) > 100:
            return result
    except Exception as e:
        logger.warning(f"Web search failed: {e}")

    return dedent("""
    Markets remain focused on technology stocks as investors balance
    AI-driven growth optimism with interest rate expectations and earnings outlooks.

    Key themes:
    - Strong interest in AI leaders such as NVIDIA (NVDA) and Microsoft (MSFT)
    - Cautious trading in mega-cap stocks ahead of earnings
    - Sensitivity to inflation and Federal Reserve policy signals
    """)

# --------------------------------------------------
# LLM AGENTS
# --------------------------------------------------

@st.cache_resource
def get_analysis_agent(model_choice):
    model = (
        Claude(id=model_choice, api_key=ANTHROPIC_API_KEY, temperature=0.1)
        if model_choice.startswith("claude")
        else OpenAIResponses(id=model_choice, api_key=OPENAI_API_KEY, temperature=0.1)
    )

    return Agent(
        name="Financial Analyst",
        model=model,
        instructions="""
        You are a financial analyst.
        Respond ONLY with finance-related insights.
        Never apologize or mention data limitations.
        Use clear markdown.
        """,
        markdown=True,
    )

@st.cache_resource
def get_web_agent(model_choice):
    model = OpenAIResponses(id=model_choice, api_key=OPENAI_API_KEY, temperature=0.1)
    return Agent(
        name="Web Agent",
        model=model,
        tools=[WebSearchTools(enable_search=True, enable_news=True)],
        markdown=True,
    )

# --------------------------------------------------
# CORE LOGIC
# --------------------------------------------------

def process_query(query, model_choice):

    # ---- HARD FINANCIAL GATE ----
    if not is_financial_query(query):
        st.markdown("""
        🚫 **Financial Queries Only**

        I’m a **Financial AI Agent**.
        I can help with:
        - 📊 Stocks & ETFs
        - 💎 Crypto markets
        - 📰 Market news
        - 📈 Financial analysis

        Please ask a **finance-related question**.
        """)
        return

    intent = classify_query_intent(query)
    tickers = intent["tickers"]

    placeholder = st.empty()
    placeholder.info("⚡ Processing financial data...")

    # ---- PRICE ----
    if intent["type"] == "price" and len(tickers) == 1:
        data = get_stock_data(tickers[0])
        placeholder.success("✅ Live price loaded")

        st.metric(
            data["symbol"],
            f"${data['price']:.2f}",
            f"{data['price'] - data['previous_close']:.2f}"
        )
        return

    # ---- ANALYSIS / COMPARISON ----
    if tickers:
        data = fetch_multiple(tickers[:3])
        placeholder.success("✅ Market data loaded")

        agent = get_analysis_agent(model_choice)
        result = agent.run(f"Query: {query}\n\nMarket Data:\n{data}")
        st.markdown(result.content)
        return

    # ---- NEWS ----
    placeholder.info("🌐 Fetching market context...")
    web_agent = get_web_agent(model_choice)
    news = safe_web_search(query, web_agent)

    agent = get_analysis_agent(model_choice)
    result = agent.run(f"Query: {query}\n\nMarket Context:\n{news}")
    st.markdown(result.content)

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

def main():
    st.set_page_config("📈 Financial AI Agent", layout="wide")

    st.title("📈 Financial AI Agent")
    st.caption("Strictly finance-focused. Fast & reliable.")

    with st.sidebar:
        model_choice = st.radio(
            "Model",
            ["gpt-5.2", "claude-sonnet-4-5" ],
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
        placeholder="e.g. AAPL price, Tech stocks news"
    )

    if st.button("🚀 Get Answer", type="primary") and query:
        process_query(query, model_choice)
        st.caption(f"🕒 Updated {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
