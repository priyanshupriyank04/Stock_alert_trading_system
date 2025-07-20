# 📈 Stock Alert Trading System

A Python-based monitoring & alert system for Indian stock futures. It fetches OHLC data using Kite Connect, calculates ADX/DX & a custom CBOE-style indicator, and triggers alerts when specific conditions are met.

---

## Table of Contents

- [Features](#features)  
- [Prerequisites](#prerequisites)  
- [Setup & Installation](#setup--installation)  
- [Configuration](#configuration)  
- [Folder Structure](#folder-structure)  
- [How It Works](#how-it-works)  
  - Data ingestion  
  - Calculating ADX  
  - Calculating CBOE indicator  
  - Strategy & alert generation  
- [Running the System](#running-the-system)  
- [Alerts Output](#alerts-output)  
- [Troubleshooting](#troubleshooting)  
- [Extending & Customizing](#extending--customizing)

---

## Features

- Connects to Zerodha’s Kite Connect API to fetch daily and intraday futures data  
- Loads 50-day historical OHLC (+Volume) tables in PostgreSQL  
- Calculates ADX/DX using Wilder’s method (PineScript-aligned)  
- Computes a custom CBOE-like indicator using RSI + weighted volume flow  
- Triggers alerts when indicator thresholds are met  
- Exports alerts in text format: one contract per line

---

## Prerequisites

- Python 3.9+  
- PostgreSQL  
- Kite Connect API credentials  
- Recommended: virtualenv or Conda environment  

---

## Setup & Installation

```bash
git clone https://github.com/priyanshupriyank04/Stock_alert_trading_system.git
cd Stock_alert_trading_system

python -m venv venv
source venv/bin/activate  # macOS/Linux
.\venv\Scripts\activate  # Windows

pip install -r requirements.txt
