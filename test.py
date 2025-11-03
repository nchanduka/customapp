"""
Unified RAG + Scraper Pipeline for Open WebUI
---------------------------------------------
"""

import os
import re
import time
import asyncio
import json
import logging
import requests

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional, Union, Generator, Iterator

from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from pydantic import BaseModel, Field

import tiktoken

import weaviate
# Fallback for v3
from weaviate import Client as V3Client
# Weaviate v4 Client importieren
from weaviate import WeaviateClient
from weaviate.classes import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PARENT_CHUNK_TOKENS = 350
CHILD_CHUNK_TOKENS = 50
OVERLAP_TOKENS = 20
MAX_API_TOKENS = 380

# Weaviate Client type
WeaviateClientType = Optional[Union['WeaviateClient', 'V3Client']]

# ================================================================
# Unified Pipeline
# ================================================================
class Pipeline:
    class Valves(BaseModel):
        EMBEDDING_MODEL_URL: str = Field(
            default="https://integrate.api.nvidia.com/v1",
            description="Embedding Model Endpoint")
        EMBEDDING_MODEL_NAME: str = Field(
            default="nvidia/nv-embedqa-e5-v5",
            description="Embedding Model Name")
        EMBEDDING_MODEL_API_KEY: str = Field(
            default="YOUR_EMBEDDING_MODEL_API_KEY",
            description="API Key for accessing the embedding model")
        WEAVIATE_URL: str = Field(
            default="http://weaviate.open-webui.svc.cluster.local:80",
            description="Weaviate database URL")
        COLLECTION_NAME: str = Field(
            default="HierarchicalManualChunks",
            description="Weaviate collection name")
        LLM_MODEL_URL: str = Field(
            default="https://integrate.api.nvidia.com/v1",
            description="LLM Model Endpoint")
        LLM_MODEL_NAME: str = Field(
            default="meta/llama-3.1-8b-instruct",
            description="LLM Model Name")
        LLM_MODEL_API_KEY: str = Field(
            default="YOUR_LLM_MODEL_API_KEY",
            description="API Key for accessing the LLM model")        
        TOP_K: int = Field(
            default=8,
            description="Number of chunks to retrieve")
        HPE_BASE_URL: str = Field(
            default="https://support.hpe.com",
            description="HPE Support Page URL")
        HPE_SUPPORT_PAGE: str = Field(
            default="https://support.hpe.com/connect/s/product?language=en_US&kmpmoid=1014847366&tab=manuals&cep=on",
            description="HPE Private Cloud AI Support Page")
        
        #OLLAMA_BASE_URL: str = Field(default="http://host.docker.internal:11434")
        #OLLAMA_MODEL: str = Field(default="llama3:8b")

    def __init__(self):
        self.type = "manifold"
        self.id = "rag_weaviate"
        self.name = "RAG "

        self.valves = self.Valves()
        self.weaviate_client: WeaviateClientType = None
        self.client_version = 0
        self.collection_exists = False

        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self._connect_weaviate()

    # ============================================================
    # Connect to Weaviate
    # ============================================================
    def _connect_weaviate(self) -> bool:
        if not weaviate not in globals():
            logger.error("Weaviate client not installed.")
            return False
        try:
            url = self.valves.WEAVIATE_URL
            parsed = urlparse(url)
            scheme = parsed.scheme or 'http'
            grpc_hostname = parsed.hostname.replace("weaviate", "weaviate-grpc", 1)
            self.weaviate_client = weaviate.connect_to_custom(
                http_host=parsed.hostname or "weaviate", 
                http_port=parsed.port or 8080,
                http_secure=False if scheme == 'http' else True,
                grpc_host=grpc_hostname,
                grpc_port=50051,
                grpc_secure=False if scheme == 'http' else True,
            )
            if self.weaviate_client.is_ready():
                logger.info("Connected to Weaviate (v{self.client_version}) at {url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            return False

    # ============================================================
    # Scraper + Ingestion
    # ============================================================
    async def scrape_and_ingest(self, max_manuals=1):
        if 'async_playwright' not in globals():
            raise RuntimeError("Playwright and BeautifulSoup required.")

    def generate_with_llm(self, prompt, stream=False):
        url = self.valves.LLM_MODEL_URL
        headers = {
            "Authorization": f"Bearer {self.valves.LLM_MODEL_API_KEY}",
            "Content-Type": "application/json",
        }        
        try:
            r = requests.post(
                f"{url}/chat/completions",
                json= {
                    "model": self.valves.LLM_MODEL_NAME, 
                    "messages": [
                        {"role": "system", "content": prompt},
                    ],
                    "temperature": 0.1
                },
                headers=headers,
                stream=True,         
                timeout=120,
            )
            r.raise_for_status()
            if stream:
                def generate():
                    for line in r.iter_lines():
                        if line:
                            import json
                            try:
                                chunk = json.loads(line)
                                if 'response' in chunk:
                                    yield chunk['response']
                            except Exception:
                                continue
                return generate()
            else:
                return r.json().get("response", "No response generated")
        except Exception as e:
            return f"Error: {e}"


    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        logger.info(f"pipe:{__name__}")
        logger.info(f"User message: {user_message}")

        # --- Command trigger for ingestion ---
        if "scrape" in user_message.lower() or "update database" in user_message.lower():
            import asyncio
            try:
                asyncio.run(self.scrape_and_ingest(max_manuals=1))
                return "Scraping and ingestion complete! You can now ask questions about the new documentation."
            except Exception as e:
                return f"Failed to run scraper: {e}"
       
        try:
            # Check if streaming is requested
            stream = body.get("stream", False)        
            chunks = self.search_chunks(user_message)
            if not chunks:
                return "I couldn't find any relevant information in the documentation database. Please try rephrasing your question or ensure the documentation has been indexed in Weaviate."
            context = self.create_context(chunks)
            enhanced_prompt = f"""You are a helpful technical documentation assistant. Answer the user's question based on the provided documentation context.

DOCUMENTATION CONTEXT:
{context}

INSTRUCTIONS:
- Provide detailed, step-by-step answers when applicable
- Use specific information from the documentation
- If the documentation doesn't fully answer the question, acknowledge this
- Be comprehensive and include all relevant details
- Do not mention "Source X" - integrate information naturally

USER QUESTION:
{user_message}

DETAILED ANSWER:"""
            # 4. Generate response with LLM
            logger.info(f"Generating response with context from {len(chunks)} chunks")
            response = self.generate_with_llm(enhanced_prompt, stream=stream)            
            return response            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return f"An error occurred while processing your request: {str(e)}"


