#  Step 1: Import Required Libraries
import os                # For environment variables and file handling
import time              # For adding delays where needed
import datetime          # To handle timestamps
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
API_KEY = "8re7mjcm2btaozwf"  #  Replace with your actual API key
API_SECRET = "fw8gm7wfeclcic9rlkp0tbzx4h2ss2n1"  # Replace with your actual API secret
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
                return access_token  # ✅Return the valid token
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

        # 🔹 Step 4: Save the new access token
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