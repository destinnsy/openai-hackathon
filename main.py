from dotenv import load_dotenv
load_dotenv()  # This will load the environment variables from the .env file

from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Annotated

from helper.store import store
from helper.retrieve import retrieve
import asyncio

app = FastAPI()

class StoreRequest(BaseModel):
    user_messages: list[str]
    assistant_messages: list[str]

@app.post("/store")
async def store_api(request: StoreRequest):
    asyncio.create_task(store(request.user_messages, request.assistant_messages))
    return {"message": "Successfully stored"}

class RetrieveRequest(BaseModel):
    query: str

@app.post("/retrieve")
async def retrieve_api(request: RetrieveRequest):
    retrieved_content = retrieve(request.query)
    return {"content": retrieved_content}

@app.get("/")
async def root(q: Annotated[str | None, Query(max_length=50)] = None):
    return {"message": "Hello World"}