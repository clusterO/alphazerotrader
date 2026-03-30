from dydx3 import Client
from web3 import Web3

#
# Access public API endpoints.
#
public_client = Client(
    host='http://localhost:8080',
)
public_client.public.get_markets()
