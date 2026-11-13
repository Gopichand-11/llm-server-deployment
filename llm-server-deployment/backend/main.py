
import os

from typing import Literal

from fastapi import FastAPI, HTTPException
from groq import (
    AsyncGroq,
    APIConnectionError,
    APIStatusError,
    RateLimitError
)
from pydantic import BaseModel


app = FastAPI(
    title="LLM Server Deployment",
    description="FastAPI backend connected to Groq",
    version="1.0.0"
)


# Groq configuration
MODEL_NAME = os.getenv("MODEL_NAME", "")


# Request models
class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class ChatResponse(BaseModel):
    answer: str
    model: str


# Health check
@app.get("/")
async def root():
    return {"message": "LLM Backend is running"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "provider": "Groq",
        "model": MODEL_NAME
    }


# Chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Groq API key is not configured"
        )

    if not MODEL_NAME:
        raise HTTPException(
            status_code=500,
            detail="Groq model is not configured"
        )

    try:
        client = AsyncGroq(api_key=api_key)

        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                message.model_dump()
                for message in request.messages
            ],
            temperature=0.7,
            max_completion_tokens=1024
        )

        answer = completion.choices[0].message.content

        return ChatResponse(
            answer=answer or "",
            model=MODEL_NAME
        )

    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Groq API rate limit or quota exceeded"
        )

    except APIConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Unable to connect to Groq"
        )

    except APIStatusError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Groq API error: {error.status_code}"
        )
