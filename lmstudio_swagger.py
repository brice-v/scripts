# /// script
# dependencies = [
#     "fastapi>=0.104.0",
#     "uvicorn[standard]>=0.24.0",
#     "httpx>=0.25.0",
#     "pydantic>=2.0.0",
# ]
# requires-python = ">=3.8"
# ///

"""
LM Studio API Swagger UI
A FastAPI application that provides a Swagger UI for LM Studio's OpenAI-compatible API

Usage:
    uv run lmstudio_swagger.py

Then open http://localhost:8000 in your browser.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import httpx
import os
from contextlib import asynccontextmanager

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1")

app = FastAPI(
    title="LM Studio API",
    description="OpenAI-compatible API for LM Studio with Swagger UI",
    version="1.0.0",
    docs_url="/",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Models
class ChatMessage(BaseModel):
    role: str = Field(
        ..., description="The role of the message author (system, user, assistant)"
    )
    content: str = Field(..., description="The content of the message")


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    messages: List[ChatMessage] = Field(
        ..., description="A list of messages comprising the conversation"
    )
    temperature: Optional[float] = Field(
        0.7, description="Sampling temperature between 0 and 2"
    )
    max_tokens: Optional[int] = Field(
        None, description="Maximum number of tokens to generate"
    )
    top_p: Optional[float] = Field(1.0, description="Nucleus sampling parameter")
    stream: Optional[bool] = Field(
        False, description="Whether to stream back partial progress"
    )
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    presence_penalty: Optional[float] = Field(
        0, description="Presence penalty between -2.0 and 2.0"
    )
    frequency_penalty: Optional[float] = Field(
        0, description="Frequency penalty between -2.0 and 2.0"
    )


class CompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    prompt: Union[str, List[str]] = Field(
        ..., description="The prompt(s) to generate completions for"
    )
    temperature: Optional[float] = Field(
        0.7, description="Sampling temperature between 0 and 2"
    )
    max_tokens: Optional[int] = Field(
        16, description="Maximum number of tokens to generate"
    )
    top_p: Optional[float] = Field(1.0, description="Nucleus sampling parameter")
    stream: Optional[bool] = Field(
        False, description="Whether to stream back partial progress"
    )
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    presence_penalty: Optional[float] = Field(
        0, description="Presence penalty between -2.0 and 2.0"
    )
    frequency_penalty: Optional[float] = Field(
        0, description="Frequency penalty between -2.0 and 2.0"
    )


class EmbeddingRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    input: Union[str, List[str]] = Field(..., description="Input text to embed")


async def make_lmstudio_request(endpoint: str, payload: dict):
    """Make a request to LM Studio server"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{LMSTUDIO_BASE_URL}/{endpoint}", json=payload, timeout=300.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot connect to LM Studio server at {LMSTUDIO_BASE_URL}. Make sure LM Studio is running with the server enabled.",
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))


@app.get("/v1/models", summary="List available models")
async def list_models():
    """
    Lists the currently available models.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{LMSTUDIO_BASE_URL}/models", timeout=30.0)
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot connect to LM Studio server at {LMSTUDIO_BASE_URL}. Make sure LM Studio is running with the server enabled.",
            )


@app.post("/v1/chat/completions", summary="Create chat completion")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Creates a completion for the chat message.
    """
    payload = request.model_dump(exclude_none=True)
    return await make_lmstudio_request("chat/completions", payload)


@app.post("/v1/completions", summary="Create completion")
async def create_completion(request: CompletionRequest):
    """
    Creates a completion for the provided prompt(s).
    """
    payload = request.model_dump(exclude_none=True)
    return await make_lmstudio_request("completions", payload)


@app.post("/v1/embeddings", summary="Create embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """
    Creates an embedding vector representing the input text.
    """
    payload = request.model_dump(exclude_none=True)
    return await make_lmstudio_request("embeddings", payload)


@app.get("/health", summary="Health check")
async def health_check():
    """
    Check if the proxy server is running and can connect to LM Studio.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{LMSTUDIO_BASE_URL}/models", timeout=5.0)
            if response.status_code == 200:
                return {"status": "healthy", "lmstudio_connected": True}
            else:
                return {
                    "status": "healthy",
                    "lmstudio_connected": False,
                    "lmstudio_status": response.status_code,
                }
        except:
            return {"status": "healthy", "lmstudio_connected": False}


def main():
    import uvicorn

    print(f"Starting LM Studio Swagger UI")
    print(f"LM Studio URL: {LMSTUDIO_BASE_URL}")
    print(f"Swagger UI: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
