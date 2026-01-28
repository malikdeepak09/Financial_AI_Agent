"""
Test Suite for Financial AI Agent
Run with: pytest test_financial_agent.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)




# Import functions from the main app
#
try:
    from src.financial_agent import (
    is_financial_query,
    classify_query_intent,
    get_stock_data,
    fetch_multiple,
    safe_web_search,
    ASSET_MAPPING,
    FINANCIAL_KEYWORDS
)

except ImportError:
    pytest.skip("financial_agent.py not found", allow_module_level=True)


# ============================================================
# TEST DATA
# ============================================================

VALID_FINANCIAL_QUERIES = [
    "Apple stock price",
    "What's TSLA worth?",
    "Bitcoin price today",
    "Compare NVDA vs AMD",
    "Tech stocks news",
    "S&P 500 analysis",
    "Crypto market update",
    "AAPL earnings report",
    "Interest rate decision",
    "Market trends"
]

NON_FINANCIAL_QUERIES = [
    "Hello",
    "What's the weather?",
    "How to cook pasta",
    "Who won the game?",
    "Movie recommendations",
    "Best restaurants nearby",
    "Python programming help",
    "What's 2+2?"
]

TICKER_TEST_CASES = [
    ("Apple stock price", ["AAPL"]),
    ("TSLA vs NVDA", ["TSLA", "NVDA"]),
    ("Bitcoin today", ["BTC-USD"]),
    ("Microsoft and Google", ["MSFT", "GOOGL"]),
    ("SPY analysis", ["SPY"]),
    ("What about AAPL?", ["AAPL"]),
]


# ============================================================
# TEST: FINANCIAL QUERY DETECTION
# ============================================================

class TestFinancialQueryDetection:
    """Test the financial query filtering"""
    
    def test_valid_financial_queries(self):
        """Valid financial queries should return True"""
        for query in VALID_FINANCIAL_QUERIES:
            assert is_financial_query(query), f"Failed for: {query}"
    
    def test_non_financial_queries(self):
        """Non-financial queries should return False"""
        for query in NON_FINANCIAL_QUERIES:
            assert not is_financial_query(query), f"Failed for: {query}"
    
    def test_case_insensitive(self):
        """Should work regardless of case"""
        assert is_financial_query("STOCK PRICE")
        assert is_financial_query("stock price")
        assert is_financial_query("StOcK PrIcE")
    
    def test_partial_matches(self):
        """Should detect keywords in larger sentences"""
        assert is_financial_query("I want to know the stock market status")
        assert is_financial_query("Tell me about cryptocurrency investments")
    
    def test_empty_query(self):
        """Empty query should return False"""
        assert not is_financial_query("")
        assert not is_financial_query("   ")


# ============================================================
# TEST: INTENT CLASSIFICATION
# ============================================================

class TestIntentClassification:
    """Test query intent classification"""
    
    def test_price_intent(self):
        """Should detect price queries"""
        result = classify_query_intent("Apple stock price")
        assert result["type"] == "price"
        assert "AAPL" in result["tickers"]
    
    def test_comparison_intent(self):
        """Should detect comparison queries"""
        result = classify_query_intent("Compare TSLA vs NVDA")
        assert result["type"] == "comparison"
        assert "TSLA" in result["tickers"]
        assert "NVDA" in result["tickers"]
    
    def test_news_intent(self):
        """Should detect news queries"""
        result = classify_query_intent("Tech stocks news")
        assert result["type"] == "news"
    
    def test_analysis_intent(self):
        """Should detect analysis queries"""
        result = classify_query_intent("AAPL analysis")
        assert result["type"] == "analysis"
        assert "AAPL" in result["tickers"]
    
    def test_general_intent(self):
        """Should default to general for unclear queries"""
        result = classify_query_intent("What's happening in markets?")
        assert result["type"] == "general"
    
    @pytest.mark.parametrize("query,expected_tickers", TICKER_TEST_CASES)
    def test_ticker_extraction(self, query, expected_tickers):
        """Should correctly extract tickers"""
        result = classify_query_intent(query)
        for ticker in expected_tickers:
            assert ticker in result["tickers"], f"Expected {ticker} in {result['tickers']}"


# ============================================================
# TEST: ASSET MAPPING
# ============================================================

class TestAssetMapping:
    """Test the asset mapping dictionary"""
    
    def test_stock_mappings(self):
        """Should have correct stock mappings"""
        assert ASSET_MAPPING["AAPL"] == "AAPL"
        assert ASSET_MAPPING["APPLE"] == "AAPL"
        assert ASSET_MAPPING["TSLA"] == "TSLA"
        assert ASSET_MAPPING["TESLA"] == "TSLA"
    
    def test_crypto_mappings(self):
        """Should have correct crypto mappings"""
        assert ASSET_MAPPING["BTC"] == "BTC-USD"
        assert ASSET_MAPPING["BITCOIN"] == "BTC-USD"
        assert ASSET_MAPPING["ETH"] == "ETH-USD"
        assert ASSET_MAPPING["ETHEREUM"] == "ETH-USD"
    
    def test_index_mappings(self):
        """Should have correct index mappings"""
        assert ASSET_MAPPING["SPY"] == "SPY"
        assert ASSET_MAPPING["S&P"] == "SPY"
        assert ASSET_MAPPING["QQQ"] == "QQQ"
        assert ASSET_MAPPING["NASDAQ"] == "QQQ"
    
    def test_case_sensitivity(self):
        """Mapping keys should be uppercase"""
        for key in ASSET_MAPPING.keys():
            assert key.isupper(), f"Key {key} should be uppercase"


# ============================================================
# TEST: STOCK DATA FETCHING
# ============================================================

class TestStockDataFetching:
    """Test stock data retrieval"""
    
    @patch('yfinance.Ticker')
    def test_get_stock_data_success(self, mock_ticker):
        """Should successfully fetch stock data"""
        # Mock yfinance response
        mock_fast_info = Mock()
        mock_fast_info.last_price = 150.25
        mock_fast_info.previous_close = 148.50
        mock_fast_info.day_high = 152.00
        mock_fast_info.day_low = 149.00
        mock_fast_info.last_volume = 50000000
        mock_fast_info.market_cap = 2500000000000
        
        mock_ticker.return_value.fast_info = mock_fast_info
        
        result = get_stock_data("AAPL")
        
        assert result["symbol"] == "AAPL"
        assert result["price"] == 150.25
        assert result["previous_close"] == 148.50
        assert result["day_high"] == 152.00
        assert result["day_low"] == 149.00
        assert result["volume"] == 50000000
        assert result["market_cap"] == 2500000000000
    
    @patch('yfinance.Ticker')
    def test_get_stock_data_structure(self, mock_ticker):
        """Should return data with correct structure"""
        mock_fast_info = Mock()
        mock_fast_info.last_price = 100.0
        mock_fast_info.previous_close = 99.0
        mock_fast_info.day_high = 101.0
        mock_fast_info.day_low = 98.0
        mock_fast_info.last_volume = 1000000
        mock_fast_info.market_cap = 1000000000
        
        mock_ticker.return_value.fast_info = mock_fast_info
        
        result = get_stock_data("TEST")
        
        # Check all required keys exist
        required_keys = ["symbol", "price", "previous_close", "day_high", 
                        "day_low", "volume", "market_cap"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
    
    @patch('yfinance.Ticker')
    def test_fetch_multiple_tickers(self, mock_ticker):
        """Should fetch data for multiple tickers"""
        mock_fast_info = Mock()
        mock_fast_info.last_price = 100.0
        mock_fast_info.previous_close = 99.0
        mock_fast_info.day_high = 101.0
        mock_fast_info.day_low = 98.0
        mock_fast_info.last_volume = 1000000
        mock_fast_info.market_cap = 1000000000
        
        mock_ticker.return_value.fast_info = mock_fast_info
        
        tickers = ["AAPL", "TSLA", "NVDA"]
        results = fetch_multiple(tickers)
        
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)


# ============================================================
# TEST: WEB SEARCH SAFETY
# ============================================================

class TestWebSearch:
    """Test web search functionality"""
    
    def test_safe_web_search_success(self):
        """Should return web search results on success"""
        mock_agent = Mock()
        mock_result = Mock()
        mock_result.content = "This is a long enough market analysis result that exceeds 100 characters and should be returned successfully by the function."
        mock_agent.run.return_value = mock_result
        
        result = safe_web_search("market news", mock_agent)
        
        assert result == mock_result.content
        mock_agent.run.assert_called_once_with("market news")
    
    def test_safe_web_search_failure(self):
        """Should return fallback on failure"""
        mock_agent = Mock()
        mock_agent.run.side_effect = Exception("Network error")
        
        result = safe_web_search("market news", mock_agent)
        
        # Should return fallback text
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Markets" in result or "markets" in result
    
    def test_safe_web_search_short_response(self):
        """Should return fallback for very short responses"""
        mock_agent = Mock()
        mock_result = Mock()
        mock_result.content = "Short"  # Less than 100 chars
        mock_agent.run.return_value = mock_result
        
        result = safe_web_search("market news", mock_agent)
        
        # Should return fallback instead
        assert "Markets" in result or "markets" in result


# ============================================================
# TEST: EDGE CASES
# ============================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_empty_ticker_list(self):
        """Should handle empty ticker list"""
        result = classify_query_intent("What's happening?")
        assert result["tickers"] == []
    
    def test_multiple_same_ticker(self):
        """Should handle duplicate tickers"""
        result = classify_query_intent("AAPL AAPL AAPL")
        # Should contain AAPL (may or may not deduplicate)
        assert "AAPL" in result["tickers"]
    
    def test_mixed_case_tickers(self):
        """Should handle mixed case ticker symbols"""
        result = classify_query_intent("aapl vs TsLa")
        assert "AAPL" in result["tickers"]
        assert "TSLA" in result["tickers"]
    
    def test_special_characters(self):
        """Should handle special characters gracefully"""
        queries = [
            "What's AAPL's price?",
            "TSLA vs. NVDA",
            "S&P 500 index",
            "BTC/USD price"
        ]
        for query in queries:
            result = classify_query_intent(query)
            assert "tickers" in result
    
    def test_very_long_query(self):
        """Should handle very long queries"""
        long_query = "I want to know " + "really " * 50 + "about AAPL stock price"
        result = classify_query_intent(long_query)
        assert "AAPL" in result["tickers"]


# ============================================================
# TEST: INTEGRATION SCENARIOS
# ============================================================

class TestIntegrationScenarios:
    """Test realistic user scenarios"""
    
    def test_scenario_quick_price_check(self):
        """User asks for quick price"""
        query = "AAPL"
        
        # Should be financial
        assert is_financial_query(query)
        
        # Should detect intent
        intent = classify_query_intent(query)
        assert "AAPL" in intent["tickers"]
    
    def test_scenario_company_comparison(self):
        """User compares two companies"""
        query = "Compare Apple vs Microsoft"
        
        assert is_financial_query(query)
        
        intent = classify_query_intent(query)
        assert intent["type"] == "comparison"
        assert "AAPL" in intent["tickers"]
        assert "MSFT" in intent["tickers"]
    
    def test_scenario_market_news(self):
        """User asks for market news"""
        query = "What's the latest tech stocks news?"
        
        assert is_financial_query(query)
        
        intent = classify_query_intent(query)
        assert intent["type"] == "news"
    
    def test_scenario_crypto_price(self):
        """User asks for crypto price"""
        query = "Bitcoin price today"
        
        assert is_financial_query(query)
        
        intent = classify_query_intent(query)
        assert intent["type"] == "price"
        assert "BTC-USD" in intent["tickers"]
    
    def test_scenario_non_financial_rejection(self):
        """User asks non-financial question"""
        query = "What's the weather today?"
        
        assert not is_financial_query(query)


# ============================================================
# TEST: PERFORMANCE & CACHING
# ============================================================

class TestPerformance:
    """Test performance-related aspects"""
    
    def test_classification_consistency(self):
        """Same query should return consistent results"""
        query = "Apple stock price"
        
        result1 = classify_query_intent(query)
        result2 = classify_query_intent(query)
        
        assert result1 == result2
    
    def test_financial_check_speed(self):
        """Financial check should be fast"""
        import time
        
        start = time.time()
        for _ in range(1000):
            is_financial_query("stock price")
        elapsed = time.time() - start
        
        # Should complete 1000 checks in under 1 second
        assert elapsed < 1.0


# ============================================================
# PYTEST FIXTURES
# ============================================================

@pytest.fixture
def mock_streamlit():
    """Mock streamlit components"""
    with patch('streamlit.cache_data'):
        with patch('streamlit.cache_resource'):
            yield


@pytest.fixture
def sample_stock_data():
    """Provide sample stock data for testing"""
    return {
        "symbol": "AAPL",
        "price": 150.25,
        "previous_close": 148.50,
        "day_high": 152.00,
        "day_low": 149.00,
        "volume": 50000000,
        "market_cap": 2500000000000
    }


# ============================================================
# RUN CONFIGURATION
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])