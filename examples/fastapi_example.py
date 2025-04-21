"""
A minimal FastAPI example using brain-proxy.
Run with:
    uvicorn examples.fastapi_example:app --reload
"""

from fastapi import FastAPI
# Adjust the import path below if brain_proxy is not installed as a package
from brain_proxy import BrainProxy
import dotenv, os

dotenv.load_dotenv()

app = FastAPI()

# Example: instantiate your BrainProxy class (adjust if needed)
brain_proxy = BrainProxy(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

app.include_router(brain_proxy.router, prefix="/v1")    

@app.get("/")
def root():
    # Example usage of brain_proxy; replace with real method as needed
    return {"message": "Hello from FastAPI with brain-proxy!", "proxy_status": str(brain_proxy)}
