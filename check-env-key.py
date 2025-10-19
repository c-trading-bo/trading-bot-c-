import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv('TOPSTEPX_API_KEY')
print(f'API Key: {repr(key)}')
print(f'Length: {len(key) if key else 0}')
print(f'Has comment: {"#" in key if key else False}')
