# Average Calculator Microservice

A Flask microservice that fetches numbers from a third-party API, stores them in a fixed-size window, and calculates their average.


**Prerequisites:**

*   Python 3.7 or higher installed.
*   `pip` (Python package installer) available.

**Steps to Install:**

1.  **Navigate to the Project Directory:**
    Open your terminal or command prompt and change to the root directory of this microservice project (where the `requirements.txt` file is located).
    ```bash
    cd path/to/your/stock-aggregation-microservice
    ```

2.  **(Recommended) Create and Activate a Virtual Environment:**
    It's highly recommended to use a virtual environment to manage project-specific dependencies, preventing conflicts with other Python projects.

    *   **Create the virtual environment (if it doesn't exist):**
        ```bash
        python3 -m venv venv
        ```
        (You can name it `venv` or something else like `.venv`.)

    *   **Activate the virtual environment:**
        *   **On macOS and Linux:**
            ```bash
            source venv/bin/activate
            ```
        *   **On Windows (Command Prompt):**
            ```bash
            venv\Scripts\activate.bat
            ```
        *   **On Windows (PowerShell):**
            ```bash
            .\venv\Scripts\Activate.ps1
            ```
        Your terminal prompt should change to indicate the virtual environment is active (e.g., `(venv) your-prompt$`).

3.  **Install Packages using `requirements.txt`:**
    With the virtual environment activated, run the following command:
    ```bash
    pip install -r requirements.txt
    ```
    This command will read the `requirements.txt` file and install all the listed packages and their specific versions (or minimum versions).



## API Endpoint

*   `GET /numbers/{number_id}`
    *   `number_id`:
        *   `p`: Prime numbers
        *   `f`: Fibonacci numbers
        *   `e`: Even numbers
        *   `I`: Random numbers

## Setup

1.  **Prerequisites:** Python 3.7+
2.  **Create Virtual Environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure Environment:**
    Create a `.env` file in the root directory with the following content:
    ```env
    BEARER_TOKEN="YOUR_FRESH_API_BEARER_TOKEN"
    WINDOW_SIZE=10 # Optional, defaults to 10
    ```
    Replace `"YOUR_FRESH_API_BEARER_TOKEN"` with a valid token for the test server. These tokens expire quickly.

## Running the Service

```bash
python app.py
```

![Sample Output](screenshot.png)