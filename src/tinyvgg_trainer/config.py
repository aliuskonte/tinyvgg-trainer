from dotenv import load_dotenv
import os

load_dotenv(override=True)

CLEARML = {
    "api_access_key": os.getenv("CLEARML_API_ACCESS_KEY"),
    "api_secret_key": os.getenv("CLEARML_API_SECRET_KEY"),
    "api_host":       os.getenv("CLEARML_API_HOST",    "https://api.clear.ml"),
    "web_host":       os.getenv("CLEARML_WEB_HOST",    "https://app.clear.ml"),
    "files_host":     os.getenv("CLEARML_FILES_HOST",  "https://files.clear.ml"),
}