#  Step 1: Import Required Libraries
import os                # For environment variables and file handling
import time              # For adding delays where needed
import datetime          # To handle timestamps
import numpy as np       # For handling with numpy operations
import pandas as pd      # For working with dataframes
import math              # For working with math related functions
import psycopg2          # PostgreSQL database connection
import logging           # For structured logging
from kiteconnect import KiteConnect  # Zerodha API connection


# Logging Setup
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

logging.info(" Required libraries imported successfully.")

#  Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#  API Credentials
API_KEY = "api_key"  #  Replace with your actual API key
API_SECRET = "secret_key"  # Replace with your actual API secret
ACCESS_TOKEN_FILE = "access_token.txt"

#  Initialize KiteConnect
kite = KiteConnect(api_key=API_KEY)


def get_access_token():
    """
    Checks if the access token exists and is valid. If not, prompts the user to manually enter a new one.
    """
    #  Step 1: Check if access_token.txt exists
    if os.path.exists(ACCESS_TOKEN_FILE):
        with open(ACCESS_TOKEN_FILE, "r") as file:
            access_token = file.read().strip()
            kite.set_access_token(access_token)
            logging.info(" Found existing access token. Attempting authentication...")

            #  Step 2: Validate access token
            try:
                profile = kite.profile()  #  API call to validate token
                logging.info(f"API Authentication Successful! User: {profile['user_name']}")
                return access_token  # Return the valid token
            except Exception as e:
                logging.warning(f" Invalid/Expired Access Token: {e}")
    
    #  Step 3: If token is invalid or file does not exist, ask the user for a new one
    logging.info(" Fetching new access token...")

    request_token_url = kite.login_url()
    logging.info(f" Go to this URL, authorize, and retrieve the request token: {request_token_url}")
    
    request_token = input(" Paste the request token here: ").strip()

    try:
        data = kite.generate_session(request_token, api_secret=API_SECRET)
        access_token = data["access_token"]

        #  Step 4: Save the new access token
        with open(ACCESS_TOKEN_FILE, "w") as file:
            file.write(access_token)

        logging.info(" New access token saved successfully!")
        return access_token
    except Exception as e:
        logging.error(f" Failed to generate access token: {e}")
        return None

#  Get Access Token
access_token = get_access_token()

if access_token:
    logging.info(" API is now authenticated and ready to use!")
else:
    logging.error(" API authentication failed. Please check credentials and try again.")

from psycopg2 import sql

#  Database Configuration
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "admin123"
DB_HOST = "localhost"
DB_PORT = "5432"

#  Connect to PostgreSQL
def connect_to_db():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.autocommit = True  # Enable autocommit mode
        return conn
    except Exception as e:
        logging.error(f" Failed to connect to database: {e}")
        return None
connect_to_db()


def update_current_month_futures():
    """
    Fetch all current month stock futures from Kite, filter them,
    and store in the PostgreSQL table 'current_month_futures'.
    If the table exists, it will be cleared first.
    """
    conn = connect_to_db()
    if conn is None:
        logging.error("Database connection failed. Cannot update futures list.")
        return
    
    cur = conn.cursor()

    # Step 1: Create the table if not exists
    create_table_query = """
    CREATE TABLE IF NOT EXISTS current_month_futures (
        tradingsymbol TEXT PRIMARY KEY,
        instrument_token BIGINT,
        expiry DATE,
        name TEXT
    );
    """
    cur.execute(create_table_query)

    # Step 2: Clear old data
    cur.execute("TRUNCATE TABLE current_month_futures;")
    logging.info("Cleared old futures data.")

    # Step 3: Fetch instruments list from Kite
    try:
        instruments = kite.instruments("NFO")
    except Exception as e:
        logging.error(f"Failed to fetch instruments from Kite: {e}")
        return

    # Step 4: Detect current month expiry (last Thursday logic)
    today = datetime.date.today()
    # Get all expiries from the dump to find the nearest one in current month
    all_expiries = sorted({inst["expiry"] for inst in instruments if inst["segment"] == "NFO-FUT"})
    current_month_expiry = None
    for expiry in all_expiries:
        if expiry.month == today.month and expiry.year == today.year:
            current_month_expiry = expiry
            break

    if current_month_expiry is None:
        logging.error("Could not determine current month expiry.")
        return

    # Step 5: Filter only current month stock futures (exclude NIFTY/BANKNIFTY)
    current_futs = [
        inst for inst in instruments
        if inst["segment"] == "NFO-FUT"
        and inst["expiry"] == current_month_expiry
        and not inst["tradingsymbol"].startswith(("NIFTY", "BANKNIFTY"))
    ]

    logging.info(f"Found {len(current_futs)} current month stock futures.")

    # Step 6: Insert into DB
    insert_query = """
    INSERT INTO current_month_futures (tradingsymbol, instrument_token, expiry, name)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (tradingsymbol) DO UPDATE
    SET instrument_token = EXCLUDED.instrument_token,
        expiry = EXCLUDED.expiry,
        name = EXCLUDED.name;
    """
    for inst in current_futs:
        cur.execute(insert_query, (
            inst["tradingsymbol"],
            inst["instrument_token"],
            inst["expiry"],
            inst["name"]
        ))

    conn.commit()
    logging.info("Updated 'current_month_futures' table with latest contracts.")
    cur.close()
    conn.close()

# Run the function once when script starts
update_current_month_futures()

# List of NSE Market Holidays for 2025
MARKET_HOLIDAYS = {
    "2025-02-26", "2025-03-14", "2025-03-31", "2025-04-10",
    "2025-04-14", "2025-04-18", "2025-05-01", "2025-08-15",
    "2025-08-27", "2025-10-02", "2025-10-21", "2025-10-22",
    "2025-11-05", "2025-12-25"
}

def update_futures_ohlc():
    """
    For all current month futures in 'current_month_futures',
    fetch 50 previous daily candles + (optionally) today's synthetic daily candle (from hourly data),
    and store them in separate OHLC tables (one per contract, now including volume).
    If the OHLC table exists, it will be truncated and rewritten.
    """
    conn = connect_to_db()
    if conn is None:
        logging.error("Database connection failed. Cannot update OHLC tables.")
        return

    cur = conn.cursor()

    # Fetch all current month futures
    cur.execute("SELECT tradingsymbol, instrument_token FROM current_month_futures;")
    futures_list = cur.fetchall()
    logging.info(f"Found {len(futures_list)} futures to update OHLC data.")

    today = datetime.date.today()
    now = datetime.datetime.now().time()

    is_holiday = today.strftime("%Y-%m-%d") in MARKET_HOLIDAYS
    is_weekend = today.weekday() >= 5
    is_after_open = now >= datetime.time(9, 15)

    fetch_today = (not is_holiday) and (not is_weekend) and is_after_open

    for tradingsymbol, token in futures_list:
        try:
            table_name = f"ohlc_{tradingsymbol.lower()}"

            # Ensure table exists with volume column
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                date DATE PRIMARY KEY,
                open NUMERIC,
                high NUMERIC,
                low NUMERIC,
                close NUMERIC,
                volume NUMERIC
            );
            """
            cur.execute(create_table_query)

            # If the table already exists but has no volume column, add it
            cur.execute(f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name = '{table_name}' AND column_name = 'volume'
                    ) THEN
                        ALTER TABLE {table_name} ADD COLUMN volume NUMERIC;
                    END IF;
                END $$;
            """)

            # Clear old data
            cur.execute(f"TRUNCATE TABLE {table_name};")

            # Fetch last 50 daily candles (excluding today)
            to_date = today
            from_date = to_date - datetime.timedelta(days=80)
            daily_candles = kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date - datetime.timedelta(days=1),
                interval="day"
            )
            df_daily = pd.DataFrame(daily_candles)[["date", "open", "high", "low", "close", "volume"]]

            # Filter out holidays and keep last 50 trading days
            df_daily = df_daily[~df_daily["date"].dt.strftime("%Y-%m-%d").isin(MARKET_HOLIDAYS)]
            df_daily = df_daily.tail(50)

            # Optionally add today's synthetic daily candle
            if fetch_today:
                hourly_candles = kite.historical_data(
                    instrument_token=token,
                    from_date=today,
                    to_date=today,
                    interval="60minute"
                )
                df_hourly = pd.DataFrame(hourly_candles)[["date", "open", "high", "low", "close", "volume"]]
                if not df_hourly.empty:
                    synthetic_row = {
                        "date": today,
                        "open": df_hourly.iloc[0]["open"],
                        "high": df_hourly["high"].max(),
                        "low": df_hourly["low"].min(),
                        "close": df_hourly.iloc[-1]["close"],
                        "volume": df_hourly["volume"].sum()
                    }
                    df_daily = pd.concat([df_daily, pd.DataFrame([synthetic_row])], ignore_index=True)

            # Insert data into OHLC table (with volume)
            insert_query = f"""
            INSERT INTO {table_name} (date, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (date) DO UPDATE
            SET open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume;
            """
            for _, row in df_daily.iterrows():
                cur.execute(insert_query, (
                    row["date"], row["open"], row["high"], row["low"], row["close"], row["volume"]
                ))

            conn.commit()
            logging.info(f"Updated OHLC table for {tradingsymbol} with {len(df_daily)} rows (with volume).")

            time.sleep(0.5)  # avoid Kite rate limit

        except Exception as e:
            logging.error(f"Failed to update OHLC for {tradingsymbol}: {e}")
            continue

    cur.close()
    conn.close()
    logging.info("All OHLC tables updated successfully (with volume).")


# update_futures_ohlc()

def calculate_adx_for_table(table_name: str, period: int = 2):
    """
    Calculates ADX, DI+, and DI− using Wilder's smoothing (matching Pine Script)
    and updates them in the given OHLC table (uses 'date' column, handles Decimal types).
    """
    try:
        conn = connect_to_db()
        if not conn:
            logging.error(" DB connection failed for ADX calculation.")
            return

        cur = conn.cursor()

        # Fetch OHLC data (date instead of timestamp)
        cur.execute(f"SELECT date, open, high, low, close FROM {table_name} ORDER BY date ASC;")
        rows = cur.fetchall()
        if not rows:
            logging.warning(f" No data found in table {table_name} for ADX calculation.")
            return

        # Create DataFrame
        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"])

        # Convert Decimal (NUMERIC) to float for math operations
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)

        # Previous period values
        df["prev_close"] = df["close"].shift(1)
        df["prev_high"] = df["high"].shift(1)
        df["prev_low"] = df["low"].shift(1)

        # True Range
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["prev_close"]).abs()
        tr3 = (df["low"] - df["prev_close"]).abs()
        df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = df["high"] - df["prev_high"]
        down_move = df["prev_low"] - df["low"]
        df["+dm"] = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        df["-dm"] = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Initialize smoothing columns
        df["tr_smooth"] = 0.0
        df["+dm_smooth"] = 0.0
        df["-dm_smooth"] = 0.0

        # Initialize first period smoothing sums (match PineScript logic)
        df.loc[period, "tr_smooth"] = df["tr"].iloc[0:period].sum()
        df.loc[period, "+dm_smooth"] = df["+dm"].iloc[0:period].sum()
        df.loc[period, "-dm_smooth"] = df["-dm"].iloc[0:period].sum()

        # Initialize first row (avoid NaN)
        df.loc[0, "tr_smooth"] = df.loc[0, "tr"]
        df.loc[0, "+dm_smooth"] = df.loc[0, "+dm"]
        df.loc[0, "-dm_smooth"] = df.loc[0, "-dm"]

        # Wilder smoothing loop
        for i in range(period + 1, len(df)):
            df.loc[i, "tr_smooth"] = df.loc[i-1, "tr_smooth"] - (df.loc[i-1, "tr_smooth"] / period) + df.loc[i, "tr"]
            df.loc[i, "+dm_smooth"] = df.loc[i-1, "+dm_smooth"] - (df.loc[i-1, "+dm_smooth"] / period) + df.loc[i, "+dm"]
            df.loc[i, "-dm_smooth"] = df.loc[i-1, "-dm_smooth"] - (df.loc[i-1, "-dm_smooth"] / period) + df.loc[i, "-dm"]

        # DI+ and DI−
        df["di_plus"] = 100 * df["+dm_smooth"] / df["tr_smooth"]
        df["di_minus"] = 100 * df["-dm_smooth"] / df["tr_smooth"]

        # DX and ADX
        df["dx"] = 100 * (df["di_plus"] - df["di_minus"]).abs() / (df["di_plus"] + df["di_minus"])
        df["adx"] = df["dx"].rolling(window=period).mean()

        # Ensure table has required columns for ADX
        cur.execute(f"""
            ALTER TABLE {table_name}
            ADD COLUMN IF NOT EXISTS adx NUMERIC,
            ADD COLUMN IF NOT EXISTS di_plus NUMERIC,
            ADD COLUMN IF NOT EXISTS di_minus NUMERIC;
        """)
        conn.commit()

        # Update table with computed values
        for _, row in df.iterrows():
            cur.execute(f"""
                UPDATE {table_name}
                SET 
                    adx = %s,
                    di_plus = %s,
                    di_minus = %s
                WHERE date = %s;
            """, (
                round(row["adx"], 4) if not pd.isna(row["adx"]) else None,
                round(row["di_plus"], 4) if not pd.isna(row["di_plus"]) else None,
                round(row["di_minus"], 4) if not pd.isna(row["di_minus"]) else None,
                row["date"]
            ))

        conn.commit()
        logging.info(f"ADX, DI+, and DI− updated for table {table_name}")

        cur.close()
        conn.close()

    except Exception as e:
        logging.error(f"Error in calculate_adx_for_table({table_name}): {e}")



def run_adx_for_all_futures(period: int = 2):
    """
    Iterates over all current month futures from 'current_month_futures' table,
    and runs the ADX calculation on each OHLC table.
    If the OHLC table doesn't exist, logs a warning and skips it.
    """
    conn = connect_to_db()
    if not conn:
        logging.error("DB connection failed. Cannot run ADX for all futures.")
        return

    cur = conn.cursor()
    cur.execute("SELECT tradingsymbol FROM current_month_futures;")
    futures_list = cur.fetchall()

    for (tradingsymbol,) in futures_list:
        table_name = f"ohlc_{tradingsymbol.lower()}"

        # Check if OHLC table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            );
        """, (table_name,))
        exists = cur.fetchone()[0]

        if exists:
            try:
                logging.info(f"Running ADX calculation for {table_name}...")
                calculate_adx_for_table(table_name, period=period)
            except Exception as e:
                logging.error(f"Failed ADX calc for {table_name}: {e}")
        else:
            logging.warning(f"Table {table_name} does not exist. Skipping.")

    cur.close()
    conn.close()
    logging.info("Completed ADX calculation for all futures.")


# run_adx_for_all_futures(period=2)

def calculate_cboe_for_table(table_name: str, smoothK=3, smoothD=3, lengthRSI=14, lengthStoch=14, lengthcboe=7):
    """
    Calculates the custom CBOE indicator (with volume) for the given OHLC table.
    Ensures all rows remain sorted by date before and after calculation.
    """
    try:
        conn = connect_to_db()
        if not conn:
            logging.error("DB connection failed for CBOE calculation.")
            return

        cur = conn.cursor()

        # Fetch OHLC data sorted by date ASC
        cur.execute(f"SELECT date, close, volume FROM {table_name} ORDER BY date ASC;")
        rows = cur.fetchall()
        if not rows:
            logging.warning(f"No data in {table_name} for CBOE calculation.")
            return

        # Build sorted DataFrame
        df = pd.DataFrame(rows, columns=["date", "close", "volume"])
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df.sort_values("date", inplace=True)  # Ensure sorted

        # --- RSI and Stochastic RSI Calculations ---
        def calculate_rsi_rma(series, length):
            delta = series.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

        # Primary RSI and Stochastic RSI (K & D lines)
        df["rsi1"] = calculate_rsi_rma(df["close"], lengthRSI)
        rsi_min = df["rsi1"].rolling(window=lengthStoch).min()
        rsi_max = df["rsi1"].rolling(window=lengthStoch).max()
        df["stoch_rsi"] = 100 * (df["rsi1"] - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
        df["k"] = df["stoch_rsi"].rolling(window=smoothK).mean()
        df["d"] = df["k"].rolling(window=smoothD).mean()

        # Secondary RSI for CBOE oscillator
        df["rs"] = calculate_rsi_rma(df["close"], lengthcboe)
        df["rh"] = df["rs"].rolling(window=lengthcboe).max()
        df["rl"] = df["rs"].rolling(window=lengthcboe).min()
        df["stch"] = 100 * (df["rs"] - df["rl"]) / (df["rh"] - df["rl"]).replace(0, np.nan)
        df["stch1"] = 100 - df["stch"]

        # Price structure + odds using volume-weighted flow (PineScript logic)
        change = df["close"].diff()
        df["up_input"] = np.where(change > 0, df["close"], 0)
        df["down_input"] = np.where(change < 0, df["close"], 0)
        df["up_volume_price"] = df["volume"] * df["up_input"]
        df["down_volume_price"] = df["volume"] * df["down_input"]

        def f_sum(series, length):
            length = max(1, length)
            return series.rolling(window=length).sum()

        df["upper_s"] = f_sum(df["up_volume_price"], lengthcboe)
        df["lower_s"] = f_sum(df["down_volume_price"], lengthcboe)
        df["R"] = df["upper_s"] / df["lower_s"].replace(0, np.nan)
        df["market_index"] = 100 - (100 / (1 + df["R"]))

        df["_bull_gross"] = df["market_index"]
        df["_bear_gross"] = 100 - df["market_index"]

        df["_price_stagnant"] = (df["_bull_gross"] * df["_bear_gross"]) / 100
        df["_price_bull"] = df["_bull_gross"] - df["_price_stagnant"]
        df["_price_bear"] = df["_bear_gross"] - df["_price_stagnant"]

        df["_coeff_price"] = (df["_price_stagnant"] + df["_price_bull"] + df["_price_bear"]).replace(0, np.nan) / 100
        df["_bull"] = df["_price_bull"] / df["_coeff_price"]
        df["_bear"] = df["_price_bear"] / df["_coeff_price"]
        df["_stagnant"] = df["_price_stagnant"] / df["_coeff_price"]

        # Final weighted odds
        df["_temp_stagnant"] = df["_stagnant"] * (1 + (df["stch1"] / (df["stch"] + df["stch1"])).fillna(0))
        df["_temp_bull"] = df["_bull"] * (1 + (df["stch"] / (df["stch"] + df["stch1"])).fillna(0))
        df["_temp_bear"] = df["_bear"] * (1 + (df["stch1"] / (df["stch"] + df["stch1"])).fillna(0))

        df["_coeff"] = (df["_temp_stagnant"] + df["_temp_bull"] + df["_temp_bear"]).replace(0, np.nan) / 100
        df["_odd_bull"] = df["_temp_bull"] / df["_coeff"]
        df["_odd_bear"] = df["_temp_bear"] / df["_coeff"]
        df["_odd_stagnant"] = df["_temp_stagnant"] / df["_coeff"]

        df.fillna(0, inplace=True)

        # Ensure columns exist
        cur.execute(f"""
            ALTER TABLE {table_name}
            ADD COLUMN IF NOT EXISTS stoch_k NUMERIC,
            ADD COLUMN IF NOT EXISTS stoch_d NUMERIC,
            ADD COLUMN IF NOT EXISTS odd_bull NUMERIC,
            ADD COLUMN IF NOT EXISTS odd_bear NUMERIC,
            ADD COLUMN IF NOT EXISTS odd_stagnant NUMERIC;
        """)
        conn.commit()

        # Sort again before pushing back to DB
        df.sort_values("date", inplace=True)

        # Update DB row by row in sorted order
        for _, row in df.iterrows():
            cur.execute(f"""
                UPDATE {table_name}
                SET 
                    stoch_k = %s,
                    stoch_d = %s,
                    odd_bull = %s,
                    odd_bear = %s,
                    odd_stagnant = %s
                WHERE date = %s;
            """, (
                round(row["k"], 4),
                round(row["d"], 4),
                round(row["_odd_bull"], 4),
                round(row["_odd_bear"], 4),
                round(row["_odd_stagnant"], 4),
                row["date"]
            ))

        # Physically store table sorted by date
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_date_{table_name} ON {table_name}(date);
            CLUSTER {table_name} USING idx_date_{table_name};
            ANALYZE {table_name};
        """)
        conn.commit()

        logging.info(f"CBOE indicators calculated & table {table_name} fully sorted.")

        cur.close()
        conn.close()

    except Exception as e:
        logging.error(f"Error in calculate_cboe_for_table({table_name}): {e}")

def run_cboe_for_all_futures():
    """
    Iterates over all current month futures from 'current_month_futures' table,
    and runs the CBOE indicator calculation for each OHLC table.
    If the OHLC table doesn't exist, logs a warning and skips it.
    """
    conn = connect_to_db()
    if not conn:
        logging.error("DB connection failed. Cannot run CBOE for all futures.")
        return

    cur = conn.cursor()
    cur.execute("SELECT tradingsymbol FROM current_month_futures;")
    futures_list = cur.fetchall()

    updated_count = 0
    skipped_count = 0

    for (tradingsymbol,) in futures_list:
        table_name = f"ohlc_{tradingsymbol.lower()}"

        # Check if OHLC table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            );
        """, (table_name,))
        exists = cur.fetchone()[0]

        if exists:
            try:
                logging.info(f"Running CBOE calculation for {table_name}...")
                calculate_cboe_for_table(table_name)
                updated_count += 1
            except Exception as e:
                logging.error(f"Failed CBOE calc for {table_name}: {e}")
        else:
            logging.warning(f"Table {table_name} does not exist. Skipping.")
            skipped_count += 1

    cur.close()
    conn.close()

    logging.info(f"CBOE calculation completed. Updated: {updated_count} tables, Skipped: {skipped_count} tables.")

# Run the CBOE indicator calculation for all current month futures
# run_cboe_for_all_futures()


import logging
import pandas as pd

import csv

def run_strategy_and_alert():
    """
    Iterates over all current month futures and triggers alerts if:
    - Bullish: (odd_bull[-1] < 15 OR odd_bull[0] < 15) AND 
               (di_plus[-1] < 9 OR di_plus[0] < 9) AND 
               (odd_bull[0] < 15 AND di_plus[0] < 9)
    - Bearish: (odd_bear[-1] < 15 OR odd_bear[0] < 15) AND 
               (di_minus[-1] < 9 OR di_minus[0] < 9) AND 
               (odd_bear[0] < 15 AND di_minus[0] < 9)
    Writes the triggered contract names into alerts.csv (one per line).
    """
    conn = connect_to_db()
    if not conn:
        logging.error("DB connection failed. Cannot run strategy.")
        return

    cur = conn.cursor()
    cur.execute("SELECT tradingsymbol FROM current_month_futures;")
    futures_list = cur.fetchall()

    alerts_triggered = []

    for (tradingsymbol,) in futures_list:
        table_name = f"ohlc_{tradingsymbol.lower()}"
        try:
            # Check if the table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                );
            """, (table_name,))
            exists = cur.fetchone()[0]
            if not exists:
                continue

            # Fetch last 2 rows (latest and latest-1)
            cur.execute(f"""
                SELECT date, odd_bull, odd_bear, di_plus, di_minus
                FROM {table_name}
                ORDER BY date DESC
                LIMIT 2;
            """)
            rows = cur.fetchall()
            if len(rows) < 1:
                continue

            df = pd.DataFrame(rows, columns=["date", "odd_bull", "odd_bear", "di_plus", "di_minus"])
            df = df.astype(float, errors="ignore")

            # Latest and previous
            latest = df.iloc[0]
            prev = df.iloc[1] if len(df) > 1 else None

            # Defaults if prev missing
            prev_odd_bull = prev["odd_bull"] if prev is not None else float('inf')
            prev_di_plus = prev["di_plus"] if prev is not None else float('inf')
            prev_odd_bear = prev["odd_bear"] if prev is not None else float('inf')
            prev_di_minus = prev["di_minus"] if prev is not None else float('inf')

            trigger = False

            # Bullish trigger
            if ((prev_odd_bull < 15 or latest["odd_bull"] < 15) and
                (prev_di_plus < 9 or latest["di_plus"] < 9) and
                (latest["odd_bull"] < 15 and latest["di_plus"] < 9)):
                trigger = True

            # Bearish trigger
            if ((prev_odd_bear < 15 or latest["odd_bear"] < 15) and
                (prev_di_minus < 9 or latest["di_minus"] < 9) and
                (latest["odd_bear"] < 15 and latest["di_minus"] < 9)):
                trigger = True

            if trigger:
                alerts_triggered.append(tradingsymbol)
                logging.info(f"ALERT TRIGGERED for {tradingsymbol}")

        except Exception as e:
            logging.error(f"Error processing {table_name}: {e}")
            continue

    cur.close()
    conn.close()

    # Write alerts (each on new line)
    with open("alerts.csv", "w", newline="") as f:
        for symbol in alerts_triggered:
            f.write(symbol + "\n")

    logging.info(f"Alerts written for {len(alerts_triggered)} contracts (one per line).")

run_strategy_and_alert()
