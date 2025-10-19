import os
from dotenv import load_dotenv

# Clear any existing env vars to force fresh load
if 'TOPSTEPX_API_KEY' in os.environ:
    del os.environ['TOPSTEPX_API_KEY']

load_dotenv(override=True)
key = os.getenv('TOPSTEPX_API_KEY')
print(f'API Key: {repr(key)}')
print(f'Length: {len(key) if key else 0}')
print(f'Has comment: {"#" in key if key else False}')
