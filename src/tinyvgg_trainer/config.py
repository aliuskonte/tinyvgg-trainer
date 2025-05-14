from dotenv import load_dotenv
import os

load_dotenv(override=True)

CLEARML_API_KEY = os.getenv('LEARML_API_KEY')
CLEARML_SECRET_KEY = os.getenv('LEARML_SECRET_KEY')
CLEARML_SERVER = os.getenv('LEARML_SERVER')