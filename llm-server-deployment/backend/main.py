
import os
import httpx

from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# Initialize FastAPI
app = FastAPI(
    title="LLM Server Deployment",
    description="FastAPI backend connected to Ollama",
    version="1.0.0"
)


# Configuration
OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434/api/chat"
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "llama3.2"
)


# Request models
class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class ChatResponse(BaseModel):
    answer: str
    model: str


# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "LLM Backend is running"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME
    }


# Chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:

            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "messages": [
                        message.model_dump()
                        for message in request.messages
                    ],
                    "stream": False
                }
            )

            response.raise_for_status()

            result = response.json()

            answer = result["message"]["content"]

            return ChatResponse(
                answer=answer,
                model=MODEL_NAME
            )

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Ollama request timed out"
        )

    except httpx.HTTPStatusError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned an error: {error.response.text}"
        )

    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="Unable to connect to Ollama"
        )
