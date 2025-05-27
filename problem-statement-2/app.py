import os
from flask import Flask, jsonify, request
import requests
from datetime import datetime, timedelta, timezone
# import pytz # pytz is not strictly necessary if using datetime.timezone.utc consistently
from collections import namedtuple
import math
from typing import List, Dict, Tuple, Optional, Any
from dotenv import load_dotenv # <<< ADD THIS IMPORT

# Load environment variables from .env file
load_dotenv() # <<< ADD THIS CALL

app = Flask(__name__)

# --- Configuration ---
TEST_SERVER_BASE_URL = "http://20.244.56.144/evaluation-service"
# Get the token from the environment variable (loaded from .env)
STOCK_API_BEARER_TOKEN = os.environ.get("STOCK_API_BEARER_TOKEN") # <<< USE THIS

# --- In-memory Cache ---
PricePoint = namedtuple('PricePoint', ['timestamp', 'price'])
STOCK_DATA_CACHE: Dict[str, List[PricePoint]] = {}
LAST_SUCCESSFUL_FETCH_INFO: Dict[Tuple[str, int], datetime] = {}
MIN_API_CALL_INTERVAL_SECONDS = 10 # Reduced for more frequent updates if needed, but be mindful of API limits
LAST_API_CALL_ATTEMPT_TICKER: Dict[str, datetime] = {}


# --- Datetime Utilities ---
def parse_api_timestamp(timestamp_str: str) -> datetime:
    """Parses the API's timestamp string into a UTC datetime object."""
    if '.' in timestamp_str and 'Z' == timestamp_str[-1]:
        parts = timestamp_str[:-1].split('.')
        if len(parts) == 2:
            fractional_seconds = parts[1]
            if len(fractional_seconds) > 6:
                fractional_seconds = fractional_seconds[:6]
            elif len(fractional_seconds) < 6:
                fractional_seconds = fractional_seconds.ljust(6, '0')
            timestamp_str = f"{parts[0]}.{fractional_seconds}Z"
    
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except ValueError:
        dt = datetime.strptime(timestamp_str.split('.')[0], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        print(f"Warning: Used fallback strptime for timestamp {timestamp_str}")
    return dt.astimezone(timezone.utc)


# --- Test Server API Interaction ---
def fetch_from_test_server(ticker: str, minutes: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Fetches stock data from the test server.
    Returns a list of price entries or an empty list on error.
    """
    endpoint = f"{TEST_SERVER_BASE_URL}/stocks/{ticker}"
    params = {}
    if minutes is not None:
        params['minutes'] = minutes
    
    headers = {}
    if STOCK_API_BEARER_TOKEN: # <<< CHECK AND USE THE TOKEN
        headers["Authorization"] = f"Bearer {STOCK_API_BEARER_TOKEN}"
    else:
        print("CRITICAL: STOCK_API_BEARER_TOKEN is not set. API calls will likely fail.")
        # Depending on how strict, you might want to return [] here immediately
        # or let it try and fail if the API is sometimes open.
        # For this problem, it's clear a token is needed.
        return []


    print(f"Fetching from test server: {ticker}, minutes: {minutes} (Token used: {'Yes' if STOCK_API_BEARER_TOKEN else 'No'})")
    try:
        response = requests.get(endpoint, params=params, headers=headers, timeout=5.0) # Increased timeout slightly

        # More detailed error logging for 401
        if response.status_code == 401:
            print(f"DEBUG: Received 401 Unauthorized for {ticker}. Response: {response.text[:200]}")
        
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict) and "stock" in data:
            return [data["stock"]]
        elif isinstance(data, list):
            return data
        else:
            print(f"Warning: Unexpected data format from {endpoint} for {ticker}. Data: {str(data)[:200]}")
            return []
    except requests.exceptions.Timeout:
        print(f"Error: Timeout fetching data for {ticker} from {endpoint}")
    except requests.exceptions.HTTPError as e:
        print(f"Error: HTTP error {e.response.status_code} for {ticker} from {endpoint}. Response: {e.response.text[:200]}")
    except requests.exceptions.RequestException as e:
        print(f"Error: Request exception for {ticker} from {endpoint}: {e}")
    except ValueError: 
        print(f"Error: Could not decode JSON for {ticker} from {endpoint}. Response text: {response.text[:200] if 'response' in locals() else 'N/A'}")
    return []

# --- Cache Management & Data Retrieval ---
def get_and_update_stock_data(ticker: str, minutes_history_needed: int) -> List[PricePoint]:
    now_utc = datetime.now(timezone.utc)
    
    # Check if we can skip API call based on rate limit
    # Always attempt to fetch if cache is empty for the ticker
    should_fetch_from_api = True
    if ticker in LAST_API_CALL_ATTEMPT_TICKER and STOCK_DATA_CACHE.get(ticker): # Only rate limit if we have *some* cached data
        if now_utc - LAST_API_CALL_ATTEMPT_TICKER[ticker] < timedelta(seconds=MIN_API_CALL_INTERVAL_SECONDS):
            print(f"Skipping API call for {ticker} due to rate limit. Using existing cache.")
            should_fetch_from_api = False
    
    raw_api_data = []
    if should_fetch_from_api:
        LAST_API_CALL_ATTEMPT_TICKER[ticker] = now_utc # Update attempt time before calling
        raw_api_data = fetch_from_test_server(ticker, minutes_history_needed)
    
    newly_fetched_points = []
    for item in raw_api_data:
        try:
            price = float(item.get("price"))
            ts_str = item.get("lastUpdatedAt")
            if price is not None and ts_str:
                dt_obj = parse_api_timestamp(ts_str)
                newly_fetched_points.append(PricePoint(dt_obj, price))
        except (ValueError, TypeError, AttributeError) as e: # Added AttributeError for safety
            print(f"Warning: Skipping invalid data item {item}: {e}")

    if newly_fetched_points: # Only update cache if new, valid points were fetched
        if ticker not in STOCK_DATA_CACHE:
            STOCK_DATA_CACHE[ticker] = []
        
        existing_timestamps_in_cache = {pp.timestamp for pp in STOCK_DATA_CACHE[ticker]}
        unique_new_points_added = 0
        for pp in newly_fetched_points:
            if pp.timestamp not in existing_timestamps_in_cache:
                STOCK_DATA_CACHE[ticker].append(pp)
                existing_timestamps_in_cache.add(pp.timestamp)
                unique_new_points_added +=1
        
        if unique_new_points_added > 0:
            STOCK_DATA_CACHE[ticker].sort(key=lambda x: x.timestamp)
            print(f"Updated cache for {ticker} with {unique_new_points_added} new unique points.")
        
        # Prune very old data from cache (e.g., older than 2 hours)
        # This should ideally be done less frequently or on a schedule if memory is an issue
        # For now, doing it after successful fetch.
        max_cache_age = timedelta(hours=2) 
        current_cache_len = len(STOCK_DATA_CACHE[ticker])
        STOCK_DATA_CACHE[ticker] = [
            pp for pp in STOCK_DATA_CACHE[ticker] if now_utc - pp.timestamp <= max_cache_age
        ]
        if len(STOCK_DATA_CACHE[ticker]) < current_cache_len:
            print(f"Pruned old data from {ticker} cache. Size before: {current_cache_len}, after: {len(STOCK_DATA_CACHE[ticker])}")


    window_start_time = now_utc - timedelta(minutes=minutes_history_needed)
    relevant_price_points = [
        pp for pp in STOCK_DATA_CACHE.get(ticker, []) if pp.timestamp >= window_start_time
    ]
    
    # If after attempting fetch (or using cache), we still have no relevant points, log it.
    if not relevant_price_points and should_fetch_from_api :
        print(f"Info: No relevant data points found for {ticker} in the last {minutes_history_needed} minutes after API check.")
    elif not relevant_price_points and not should_fetch_from_api:
         print(f"Info: No relevant data points for {ticker} from cache (API call skipped due to rate limit).")

    return relevant_price_points

# --- Mathematical Utilities (Unchanged) ---
def calculate_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)

def calculate_std_dev(values: List[float], mean_val: Optional[float] = None) -> Optional[float]:
    if len(values) < 2:
        return None
    if mean_val is None:
        mean_val = calculate_mean(values)
        if mean_val is None: return None
    variance = sum([(x - mean_val) ** 2 for x in values]) / (len(values) - 1)
    return math.sqrt(variance)

def calculate_covariance(values1: List[float], values2: List[float], mean1: float, mean2: float) -> Optional[float]:
    if len(values1) != len(values2) or len(values1) < 2:
        return None
    sum_products_of_deviations = sum([(values1[i] - mean1) * (values2[i] - mean2) for i in range(len(values1))])
    return sum_products_of_deviations / (len(values1) - 1)

def calculate_pearson_correlation(values1: List[float], values2: List[float]) -> Optional[float]:
    if len(values1) != len(values2) or len(values1) < 2:
        return None
    mean1 = calculate_mean(values1)
    mean2 = calculate_mean(values2)
    if mean1 is None or mean2 is None: return None
    std_dev1 = calculate_std_dev(values1, mean1)
    std_dev2 = calculate_std_dev(values2, mean2)
    if std_dev1 is None or std_dev2 is None or std_dev1 == 0 or std_dev2 == 0:
        return None
    covariance = calculate_covariance(values1, values2, mean1, mean2)
    if covariance is None: return None
    return covariance / (std_dev1 * std_dev2)

# --- API Endpoints (Logic mostly unchanged, relies on data fetching) ---
@app.route('/stocks/<string:ticker>', methods=['GET'])
def get_average_stock_price(ticker: str):
    minutes_str = request.args.get('minutes')
    aggregation_type = request.args.get('aggregation', 'average')

    if not minutes_str:
        return jsonify({"error": "Missing 'minutes' query parameter"}), 400
    try:
        minutes = int(minutes_str)
        if minutes <= 0:
            raise ValueError("Minutes must be positive")
    except ValueError:
        return jsonify({"error": "Invalid 'minutes' parameter. Must be a positive integer."}), 400

    if aggregation_type != 'average':
        return jsonify({"error": "Unsupported aggregation type. Only 'average' is supported."}), 400

    ticker = ticker.upper()
    # Ensure token is loaded for data fetching
    if not STOCK_API_BEARER_TOKEN:
         print("Warning: /stocks/ticker endpoint called but STOCK_API_BEARER_TOKEN is not configured.")
         # Behavior if token is missing (e.g., return error, or let fetch_from_test_server handle it)
         # For now, let it proceed and fetch_from_test_server will return empty if token is missing and required.

    price_history_points = get_and_update_stock_data(ticker, minutes)

    if not price_history_points:
        return jsonify({
            "error": f"No data available for {ticker} in the last {minutes} minutes or ticker not found/API issue."
        }), 404

    prices = [pp.price for pp in price_history_points]
    avg_price = calculate_mean(prices)

    response_history = [
        {"price": pp.price, "lastUpdatedAt": pp.timestamp.isoformat().replace('+00:00', 'Z')}
        for pp in price_history_points
    ]
    response_history.sort(key=lambda x: x["lastUpdatedAt"], reverse=True)

    return jsonify({
        "averageStockPrice": avg_price,
        "priceHistory": response_history
    })

@app.route('/stockcorrelation', methods=['GET'])
def get_stock_correlation():
    minutes_str = request.args.get('minutes')
    tickers = request.args.getlist('ticker')

    if not minutes_str:
        return jsonify({"error": "Missing 'minutes' query parameter"}), 400
    try:
        minutes = int(minutes_str)
        if minutes <= 0:
            raise ValueError("Minutes must be positive")
    except ValueError:
        return jsonify({"error": "Invalid 'minutes' parameter. Must be a positive integer."}), 400

    if len(tickers) != 2:
        return jsonify({"error": "Exactly two 'ticker' parameters are required for correlation."}), 400

    ticker1_symbol = tickers[0].upper()
    ticker2_symbol = tickers[1].upper()
    
    if not STOCK_API_BEARER_TOKEN:
         print("Warning: /stockcorrelation endpoint called but STOCK_API_BEARER_TOKEN is not configured.")

    history1_points = get_and_update_stock_data(ticker1_symbol, minutes)
    history2_points = get_and_update_stock_data(ticker2_symbol, minutes)

    if not history1_points or not history2_points:
        # Add more context to the error message
        t1_status = "Data found" if history1_points else "No data"
        t2_status = "Data found" if history2_points else "No data"
        return jsonify({
            "error": f"Insufficient data for correlation. {ticker1_symbol}: {t1_status}. {ticker2_symbol}: {t2_status}. (Last {minutes} minutes)."
        }), 404

    prices1 = [pp.price for pp in history1_points]
    prices2 = [pp.price for pp in history2_points]
    
    avg_price1 = calculate_mean(prices1)
    avg_price2 = calculate_mean(prices2)
    
    aligned_prices1: List[float] = []
    aligned_prices2: List[float] = []
    
    TIME_ALIGNMENT_TOLERANCE = timedelta(seconds=10) 

    # Ensure histories are sorted for alignment logic
    # get_and_update_stock_data should return filtered points from a sorted master list.
    # If they are not guaranteed sorted by timestamp from get_and_update_stock_data for the window, sort them here.
    temp_history1 = sorted(history1_points, key=lambda p: p.timestamp)
    temp_history2 = sorted(history2_points, key=lambda p: p.timestamp)

    ptr1, ptr2 = 0, 0
    while ptr1 < len(temp_history1) and ptr2 < len(temp_history2):
        pp1 = temp_history1[ptr1]
        pp2 = temp_history2[ptr2]

        if abs(pp1.timestamp - pp2.timestamp) <= TIME_ALIGNMENT_TOLERANCE:
            aligned_prices1.append(pp1.price)
            aligned_prices2.append(pp2.price)
            ptr1 += 1
            ptr2 += 1
        elif pp1.timestamp < pp2.timestamp:
            ptr1 += 1
        else:
            ptr2 += 1
            
    correlation = None
    if len(aligned_prices1) < 2:
        print(f"Warning: Less than 2 aligned data points ({len(aligned_prices1)}) for correlation between {ticker1_symbol} and {ticker2_symbol}.")
    else:
        correlation = calculate_pearson_correlation(aligned_prices1, aligned_prices2)

    response_history1 = [
        {"price": pp.price, "lastUpdatedAt": pp.timestamp.isoformat().replace('+00:00', 'Z')}
        for pp in history1_points # Use original full history for response
    ]
    response_history1.sort(key=lambda x: x["lastUpdatedAt"], reverse=True)

    response_history2 = [
        {"price": pp.price, "lastUpdatedAt": pp.timestamp.isoformat().replace('+00:00', 'Z')}
        for pp in history2_points # Use original full history for response
    ]
    response_history2.sort(key=lambda x: x["lastUpdatedAt"], reverse=True)

    return jsonify({
        "correlation": correlation,
        "stocks": {
            ticker1_symbol: {
                "averagePrice": avg_price1,
                "priceHistory": response_history1
            },
            ticker2_symbol: {
                "averagePrice": avg_price2,
                "priceHistory": response_history2
            }
        }
    })


if __name__ == '__main__':
    if not STOCK_API_BEARER_TOKEN: # <<< ADD A CHECK AT STARTUP
        print("CRITICAL: STOCK_API_BEARER_TOKEN is not defined in your .env file or environment.")
        print("The application may not function correctly as the test server requires authentication.")
        # exit(1) # Optionally exit if token is absolutely mandatory for any operation
    
    print(f"Starting Stock Aggregation Microservice...")
    print(f"Test Server API Token is {'SET' if STOCK_API_BEARER_TOKEN else 'NOT SET'}.")
    app.run(host='0.0.0.0', port=9877, debug=True)