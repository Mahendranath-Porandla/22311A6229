import os
from flask import Flask, jsonify, request
import requests
from datetime import datetime, timedelta, timezone
from collections import namedtuple
import math
from typing import List, Dict, Tuple, Optional, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)


TEST_SERVER_BASE_URL = os.environ.get("TEST_SERVER_BASE_URL", "http://20.244.56.144/evaluation-service")
STOCK_API_BEARER_TOKEN = os.environ.get("STOCK_API_BEARER_TOKEN")
MIN_API_CALL_INTERVAL_SECONDS = int(os.environ.get("MIN_API_CALL_INTERVAL_SECONDS", 10)) 
MAX_CACHE_AGE_HOURS = int(os.environ.get("MAX_CACHE_AGE_HOURS", 2)) 
API_TIMEOUT_SECONDS = float(os.environ.get("API_TIMEOUT_SECONDS", 5.0)) 
TIME_ALIGNMENT_TOLERANCE_SECONDS = int(os.environ.get("TIME_ALIGNMENT_TOLERANCE_SECONDS", 10)) 


PricePoint = namedtuple('PricePoint', ['timestamp', 'price'])
STOCK_DATA_CACHE: Dict[str, List[PricePoint]] = {} 
LAST_API_CALL_ATTEMPT_TICKER: Dict[str, datetime] = {} 


def parse_api_timestamp(timestamp_str: str) -> datetime:
    """Parses the API's timestamp string into a UTC datetime object."""
    if '.' in timestamp_str and 'Z' == timestamp_str[-1]:
        parts = timestamp_str[:-1].split('.')
        if len(parts) == 2:
            fractional_seconds = parts[1]
            # Ensure microseconds are exactly 6 digits
            fractional_seconds = (fractional_seconds + '000000')[:6]
            timestamp_str = f"{parts[0]}.{fractional_seconds}Z"
    
    try:
        # datetime.fromisoformat handles 'Z' correctly in Python 3.7+
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except ValueError:
        # Fallback for older Python or slightly different formats
        dt = datetime.strptime(timestamp_str.split('.')[0], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc) # Ensure it's UTC

# --- Test Server API Interaction ---
def fetch_from_test_server(ticker: str, minutes: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetches stock data from the test server."""
    endpoint = f"{TEST_SERVER_BASE_URL}/stocks/{ticker}"
    params = {'minutes': minutes} if minutes is not None else {}
    headers = {"Authorization": f"Bearer {STOCK_API_BEARER_TOKEN}"} if STOCK_API_BEARER_TOKEN else {}

    if not STOCK_API_BEARER_TOKEN:
        app.logger.error("CRITICAL: STOCK_API_BEARER_TOKEN is not set. API calls will fail.")
        return []

    app.logger.info(f"Fetching from test server: {ticker}, minutes: {minutes}")
    try:
        response = requests.get(endpoint, params=params, headers=headers, timeout=API_TIMEOUT_SECONDS)
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if isinstance(data, dict) and "stock" in data: # Single stock response
            return [data["stock"]]
        elif isinstance(data, list): # History response
            return data
        else:
            app.logger.warning(f"Unexpected data format from {endpoint} for {ticker}. Data: {str(data)[:200]}")
            return []
    except requests.exceptions.Timeout:
        app.logger.error(f"Timeout fetching data for {ticker} from {endpoint}")
    except requests.exceptions.HTTPError as e:
        app.logger.error(f"HTTP error {e.response.status_code} for {ticker} from {endpoint}. Response: {e.response.text[:200]}")
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Request exception for {ticker} from {endpoint}: {e}")
    except ValueError: # Includes JSONDecodeError
        app.logger.error(f"Could not decode JSON for {ticker} from {endpoint}. Response: {response.text[:200] if 'response' in locals() else 'N/A'}")
    return []


def get_and_update_stock_data(ticker: str, minutes_history_needed: int) -> List[PricePoint]:
    """Retrieves stock data, utilizing cache and fetching from API if necessary."""
    now_utc = datetime.now(timezone.utc)
    
    should_fetch_from_api = True
    if ticker in LAST_API_CALL_ATTEMPT_TICKER and STOCK_DATA_CACHE.get(ticker):
        if now_utc - LAST_API_CALL_ATTEMPT_TICKER[ticker] < timedelta(seconds=MIN_API_CALL_INTERVAL_SECONDS):
            app.logger.info(f"Skipping API call for {ticker} due to rate limit. Using existing cache.")
            should_fetch_from_api = False
    
    raw_api_data = []
    if should_fetch_from_api:
        LAST_API_CALL_ATTEMPT_TICKER[ticker] = now_utc
        raw_api_data = fetch_from_test_server(ticker, minutes_history_needed)
    
    newly_fetched_points = []
    for item in raw_api_data:
        try:
            price = float(item.get("price"))
            ts_str = item.get("lastUpdatedAt")
            if price is not None and ts_str:
                dt_obj = parse_api_timestamp(ts_str)
                newly_fetched_points.append(PricePoint(dt_obj, price))
        except (ValueError, TypeError, AttributeError) as e:
            app.logger.warning(f"Skipping invalid data item {item} for {ticker}: {e}")

    if newly_fetched_points:
        if ticker not in STOCK_DATA_CACHE:
            STOCK_DATA_CACHE[ticker] = []
        
        existing_timestamps_in_cache = {pp.timestamp for pp in STOCK_DATA_CACHE[ticker]}
        unique_new_points_added_count = 0
        for pp in newly_fetched_points:
            if pp.timestamp not in existing_timestamps_in_cache:
                STOCK_DATA_CACHE[ticker].append(pp)
                existing_timestamps_in_cache.add(pp.timestamp)
                unique_new_points_added_count += 1
        
        if unique_new_points_added_count > 0:
            STOCK_DATA_CACHE[ticker].sort(key=lambda x: x.timestamp) # Sort oldest to newest
            app.logger.info(f"Updated cache for {ticker} with {unique_new_points_added_count} new unique points.")
        
        # Prune cache
        max_cache_age_delta = timedelta(hours=MAX_CACHE_AGE_HOURS)
        original_len = len(STOCK_DATA_CACHE[ticker])
        STOCK_DATA_CACHE[ticker] = [
            pp for pp in STOCK_DATA_CACHE[ticker] if now_utc - pp.timestamp <= max_cache_age_delta
        ]
        if len(STOCK_DATA_CACHE[ticker]) < original_len:
            app.logger.info(f"Pruned {original_len - len(STOCK_DATA_CACHE[ticker])} old entries from {ticker} cache.")

    window_start_time = now_utc - timedelta(minutes=minutes_history_needed)
    relevant_price_points = [
        pp for pp in STOCK_DATA_CACHE.get(ticker, []) if pp.timestamp >= window_start_time
    ]
    
    if not relevant_price_points:
         app.logger.info(f"No relevant data points found for {ticker} in the last {minutes_history_needed} minutes (API fetched: {should_fetch_from_api}).")
    return relevant_price_points


def calculate_mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None

def calculate_std_dev(values: List[float], mean_val: Optional[float] = None) -> Optional[float]:
    n = len(values)
    if n < 2: return None
    mean_val = mean_val if mean_val is not None else calculate_mean(values)
    if mean_val is None: return None
    variance = sum([(x - mean_val) ** 2 for x in values]) / (n - 1)
    return math.sqrt(variance)

def calculate_covariance(values1: List[float], values2: List[float], mean1: float, mean2: float) -> Optional[float]:
    n = len(values1)
    if n != len(values2) or n < 2: return None
    sum_dev_products = sum([(values1[i] - mean1) * (values2[i] - mean2) for i in range(n)])
    return sum_dev_products / (n - 1)

def calculate_pearson_correlation(values1: List[float], values2: List[float]) -> Optional[float]:
    n = len(values1)
    if n != len(values2) or n < 2: return None
    mean1, mean2 = calculate_mean(values1), calculate_mean(values2)
    if mean1 is None or mean2 is None: return None
    std_dev1, std_dev2 = calculate_std_dev(values1, mean1), calculate_std_dev(values2, mean2)
    if std_dev1 is None or std_dev2 is None or std_dev1 == 0 or std_dev2 == 0: return None
    covariance = calculate_covariance(values1, values2, mean1, mean2)
    return covariance / (std_dev1 * std_dev2) if covariance is not None else None


@app.route('/stocks/<string:ticker>', methods=['GET'])
def get_average_stock_price(ticker_symbol: str):
    minutes_str = request.args.get('minutes')
    aggregation_type = request.args.get('aggregation', 'average') # Default

    if not minutes_str:
        return jsonify({"error": "Missing 'minutes' query parameter"}), 400
    try:
        minutes = int(minutes_str)
        if minutes <= 0: raise ValueError("Minutes must be positive")
    except ValueError:
        return jsonify({"error": "Invalid 'minutes' parameter. Must be a positive integer."}), 400

    if aggregation_type != 'average':
        return jsonify({"error": "Unsupported aggregation type. Only 'average' is supported."}), 400

    ticker_symbol = ticker_symbol.upper()
    price_history_points = get_and_update_stock_data(ticker_symbol, minutes)

    if not price_history_points:
        return jsonify({"error": f"No data for {ticker_symbol} in last {minutes} min or API issue."}), 404

    prices = [pp.price for pp in price_history_points]
    avg_price = calculate_mean(prices)

    response_history = sorted(
        [{"price": pp.price, "lastUpdatedAt": pp.timestamp.isoformat().replace('+00:00', 'Z')} for pp in price_history_points],
        key=lambda x: x["lastUpdatedAt"],
        reverse=True # Newest first
    )
    return jsonify({"averageStockPrice": avg_price, "priceHistory": response_history})

@app.route('/stockcorrelation', methods=['GET'])
def get_stock_correlation():
    minutes_str = request.args.get('minutes')
    tickers_list = request.args.getlist('ticker')

    if not minutes_str:
        return jsonify({"error": "Missing 'minutes' query parameter"}), 400
    try:
        minutes = int(minutes_str)
        if minutes <= 0: raise ValueError("Minutes must be positive")
    except ValueError:
        return jsonify({"error": "Invalid 'minutes' parameter (positive integer required)."}), 400

    if len(tickers_list) != 2:
        return jsonify({"error": "Exactly two 'ticker' parameters are required."}), 400

    ticker1_symbol, ticker2_symbol = tickers_list[0].upper(), tickers_list[1].upper()

    history1_points = get_and_update_stock_data(ticker1_symbol, minutes)
    history2_points = get_and_update_stock_data(ticker2_symbol, minutes)

    if not history1_points or not history2_points:
        t1_status = "Data" if history1_points else "No data"
        t2_status = "Data" if history2_points else "No data"
        return jsonify({"error": f"Insufficient data. {ticker1_symbol}: {t1_status}. {ticker2_symbol}: {t2_status} (Last {minutes} min)."}), 404

    avg_price1 = calculate_mean([pp.price for pp in history1_points])
    avg_price2 = calculate_mean([pp.price for pp in history2_points])
    
    aligned_prices1, aligned_prices2 = [], []
    temp_history1 = sorted(history1_points, key=lambda p: p.timestamp)
    temp_history2 = sorted(history2_points, key=lambda p: p.timestamp)
    
    ptr1, ptr2 = 0, 0
    alignment_tolerance = timedelta(seconds=TIME_ALIGNMENT_TOLERANCE_SECONDS)
    while ptr1 < len(temp_history1) and ptr2 < len(temp_history2):
        pp1, pp2 = temp_history1[ptr1], temp_history2[ptr2]
        if abs(pp1.timestamp - pp2.timestamp) <= alignment_tolerance:
            aligned_prices1.append(pp1.price)
            aligned_prices2.append(pp2.price)
            ptr1 += 1; ptr2 += 1
        elif pp1.timestamp < pp2.timestamp:
            ptr1 += 1
        else:
            ptr2 += 1
            
    correlation_value = None
    if len(aligned_prices1) < 2:
        app.logger.warning(f"Less than 2 aligned points ({len(aligned_prices1)}) for {ticker1_symbol} & {ticker2_symbol} correlation.")
    else:
        correlation_value = calculate_pearson_correlation(aligned_prices1, aligned_prices2)

    def format_history(points):
        return sorted(
            [{"price": p.price, "lastUpdatedAt": p.timestamp.isoformat().replace('+00:00', 'Z')} for p in points],
            key=lambda x: x["lastUpdatedAt"],
            reverse=True
        )

    return jsonify({
        "correlation": correlation_value,
        "stocks": {
            ticker1_symbol: {"averagePrice": avg_price1, "priceHistory": format_history(history1_points)},
            ticker2_symbol: {"averagePrice": avg_price2, "priceHistory": format_history(history2_points)}
        }
    })

if __name__ == '__main__':
    if not STOCK_API_BEARER_TOKEN:
        app.logger.critical("STOCK_API_BEARER_TOKEN is not defined. Test server calls will fail if auth is required.")
    
    # Use app.logger for Flask's built-in logger
    app.logger.info(f"Starting Stock Aggregation Microservice...")
    app.logger.info(f"Test Server API Token is {'SET' if STOCK_API_BEARER_TOKEN else 'NOT SET'}.")
    app.run(host='0.0.0.0', port=9877, debug=False)