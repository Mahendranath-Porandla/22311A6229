import os
from flask import Flask, jsonify, request
import requests
from collections import deque
import time
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)

#Configuration 
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", 10))
BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
THIRD_PARTY_BASE_URL = os.environ.get("THIRD_PARTY_BASE_URL", "http://20.244.56.144/evaluation-service")

ID_MAP = {
    'p': 'primes',
    'f': 'fibo',
    'e': 'even',
    'I': 'rand'
}

stored_numbers_deque = deque()
stored_numbers_set = set()

def fetch_numbers_from_third_party(number_type_id_code: str) -> list:
    """
    Fetches numbers from the third-party server.
    Returns a list of numbers, or an empty list if an error/timeout occurs.
    """
    if not BEARER_TOKEN:
        print("Error: BEARER_TOKEN is not set. Cannot authenticate.")
        return []

    api_endpoint_key = ID_MAP.get(number_type_id_code)
    if not api_endpoint_key:
        return [] 

    url = f"{THIRD_PARTY_BASE_URL}/{api_endpoint_key}"
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    fetched_numbers_list = []

    try:
        # External API call timeout is 0.5s (500ms)
        response = requests.get(url, headers=headers, timeout=0.5)
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if "numbers" in data and isinstance(data["numbers"], list):
            fetched_numbers_list = [num for num in data["numbers"] if isinstance(num, int)]
        else:
            print(f"Warning: 'numbers' key not found or invalid in response from {url}.")
    except requests.exceptions.Timeout:
        print(f"Warning: Timeout (500ms exceeded) fetching numbers from {url}")
    except requests.exceptions.HTTPError as e:
        print(f"Warning: HTTP error {e.response.status_code} from {url}. Response: {e.response.text[:100]}") # Log snippet
    except requests.exceptions.RequestException as e:
        print(f"Warning: Request exception for {url}: {e}")
    except ValueError: # Includes JSONDecodeError
        print(f"Warning: JSON decoding error from {url}.")
    
    return fetched_numbers_list


@app.route('/numbers/<string:number_id>', methods=['GET'])
def get_average_and_numbers(number_id: str):
    request_processing_start_time = time.monotonic()

    if number_id not in ID_MAP:
        return jsonify({"error": "Invalid number ID qualifier. Use 'p', 'f', 'e', or 'I'."}), 400

    if not BEARER_TOKEN: 
        return jsonify({"error": "Server configuration error: Missing API token."}), 500

    window_prev_state = list(stored_numbers_deque)
    raw_fetched_numbers = fetch_numbers_from_third_party(number_id)
    
    unique_incoming_numbers = list(dict.fromkeys(raw_fetched_numbers)) # Preserves order

    for num in unique_incoming_numbers:
        if num not in stored_numbers_set:
            if len(stored_numbers_deque) >= WINDOW_SIZE:
                oldest_num = stored_numbers_deque.popleft()
                stored_numbers_set.remove(oldest_num)
            
            stored_numbers_deque.append(num)
            stored_numbers_set.add(num)

    window_curr_state = list(stored_numbers_deque)
    current_avg = 0.0
    if window_curr_state:
        current_avg = sum(window_curr_state) / len(window_curr_state)
    
    response_data = {
        "windowPrevState": window_prev_state,
        "windowCurrState": window_curr_state,
        "numbers": raw_fetched_numbers,
        "avg": f"{current_avg:.2f}"
    }
    
    elapsed_time_ms = (time.monotonic() - request_processing_start_time) * 1000
    if elapsed_time_ms > 500:
        print(f"Warning: Service response for /numbers/{number_id} took {elapsed_time_ms:.2f} ms, exceeding 500ms limit.")

    return jsonify(response_data), 200

if __name__ == '__main__':
    if not BEARER_TOKEN:
        print("CRITICAL: BEARER_TOKEN is not set in environment variables or .env file.")
        print("The third-party API requires this token for authorization.")
        print("Please ensure it is set correctly.")

    
    print(f"Starting Average Calculator microservice on port 9876 with WINDOW_SIZE={WINDOW_SIZE}...")
    app.run(host='0.0.0.0', port=9876, debug=True) 