"""Quick connectivity check — prints paper account cash balance."""
from dotenv import load_dotenv
load_dotenv()

import os
from alpaca.trading.client import TradingClient

client = TradingClient(
    api_key=os.environ["ALPACA_API_KEY"],
    secret_key=os.environ["ALPACA_SECRET_KEY"],
    paper=True,
)

account = client.get_account()
print(f"Status:       {account.status}")
print(f"Cash:         ${float(account.cash):,.2f}")
print(f"Portfolio:    ${float(account.portfolio_value):,.2f}")
print(f"Buying power: ${float(account.buying_power):,.2f}")
