# rag_unified_pipeline.py
"""
Unified Open-WebUI pipeline combining scraper (Playwright/BeautifulSoup)
and RAG (Weaviate + embeddings + LLM). Drop this module into ./pipelines/.
Do NOT instantiate Pipeline at module level (Open-WebUI will load the class).
"""

import os
import re
import time
import json
import queue
import logging
import threading
import traceback
import requests
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Iterator, Union, Generator
from urllib.parse import urljoin, urlparse
from pathlib import Path
from pydantic import BaseModel, Field

# third-party libs expected in the environment
try:
    from playwright.async_api import async_playwright
    from bs4 import BeautifulSoup
    import tiktoken
    import weaviate
    from weaviate.classes import config as weaviate_config
    from weaviate.classes.query import Filter
except Exception:
    # The module will still import; runtime checks will raise clearer errors when functions are used.
    async_playwright = None
    BeautifulSoup = None
    tiktoken = None
    weaviate = None
    weaviate_config = None
    Filter = None

logger = logging.getLogger("rag_unified_pipeline")
logger.setLevel(os.getenv("RAG_PIPELINE_LOGLEVEL", "INFO"))
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)

# Constants used for chunking/token logic (can be tuned)
PARENT_CHUNK_TOKENS = 350
CHILD_CHUNK_TOKENS = 50
OVERLAP_TOKENS = 20
MAX_API_TOKENS = 380

# Basic directories (match original)
PDF_STORAGE_DIR = Path("/app/data_pdfs")
PDF_CLEAN_DIR = Path("/app/data_clean/manuals_pdf")
PDF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
PDF_CLEAN_DIR.mkdir(parents=True, exist_ok=True)


class Pipeline:
    """
    Single Pipeline class for Open-WebUI that:
      - triggers scraping & ingestion on user command
      - serves RAG retrieval + generation for normal queries
    """

    class Valves(BaseModel):
        # Minimal set; adapt environment variables / values as needed
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

    def __init__(self):
        self.id = "rag_unified"
        self.name = "RAG Unified Pipeline"
        self.valves = self.Valves()

        # Weaviate client will be set by ensure_weaviate_client()
        self.weaviate_client = None
        self.collection_exists = False
        self.tokenizer = None
        try:
            if tiktoken:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            logger.warning("tiktoken not available or encoding not present.")

        # internal queue/thread bookkeeping for streaming scrape
        self._scrape_threads = []

    # --------------------------
    # Weaviate connectivity
    # --------------------------
    def ensure_weaviate_client(self, max_retries: int = 8, delay: int = 5) -> None:
        """
        Ensure self.weaviate_client is set and collection existence is known.
        This is idempotent and will retry if Weaviate isn't ready yet.
        """
        if not weaviate:
            logger.error("weaviate library not imported in environment.")
            self.weaviate_client = None
            return

        url = self.valves.WEAVIATE_URL
        parsed = urlparse(url)
        host = parsed.hostname or "weaviate"
        port = parsed.port or 8080
        scheme = parsed.scheme or "http"

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Connecting to Weaviate at {host}:{port} (attempt {attempt}/{max_retries})")
                # use connect_to_custom to match provided codebases
                client = weaviate.connect_to_custom(
                    http_host=host,
                    http_port=port,
                    http_secure=False if scheme == 'http' else True,
                    grpc_host="localhost",
                    grpc_port=50051,
                    grpc_secure=False,
                    skip_init_checks=True,
                )
                              
                if client.is_ready():
                    logger.info("Weaviate is ready.")
                    self.weaviate_client = client
                    # check collection
                    try:
                        client.collections.get(self.valves.COLLECTION_NAME)
                        self.collection_exists = True
                        logger.info(f"Collection '{self.valves.COLLECTION_NAME}' exists.")
                    except Exception:
                        self.collection_exists = False
                        logger.info(f"Collection '{self.valves.COLLECTION_NAME}' does not exist yet.")
                    return
                else:
                    logger.warning("Weaviate client returned not-ready.")
            except Exception as e:
                logger.warning(f"Weaviate connection attempt failed: {e}")
            time.sleep(delay)
        logger.error("Failed to connect to Weaviate after retries.")
        self.weaviate_client = None

    # --------------------------
    # Token utilities & chunking (kept small and deterministic)
    # --------------------------
    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        # fallback naive approx
        return max(1, len(text.split()))

    def _smart_chunk_by_tokens(self, text: str, max_tokens: int, overlap_tokens: int = OVERLAP_TOKENS) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
        else:
            # naive fallback; split by sentences
            sentences = re.split(r'(?<=[.!?]) +', text)
            chunks = []
            cur = ""
            for s in sentences:
                if self._count_tokens(cur + " " + s) <= max_tokens:
                    cur = (cur + " " + s).strip()
                else:
                    if cur:
                        chunks.append(cur)
                    cur = s
            if cur:
                chunks.append(cur)
            return chunks

        safe_max = min(max_tokens, MAX_API_TOKENS)
        if len(tokens) <= safe_max:
            return [text]
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + safe_max, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens).strip()
            if chunk_text:
                chunks.append(chunk_text)
            if end >= len(tokens):
                break
            start = max(end - overlap_tokens, end)
        return chunks

    def _validate_and_split(self, text: str, max_tokens: Optional[int] = None) -> List[str]:
        if not max_tokens:
            max_tokens = MAX_API_TOKENS
        if not text:
            return []
        tokens = self._count_tokens(text)
        if tokens <= max_tokens:
            return [text.strip()]
        # fallback to the smart chunker
        return self._smart_chunk_by_tokens(text, max_tokens)

    def _create_hierarchical_chunks(self, text: str, title: str, url: str, document_id: str) -> List[Dict[str, Any]]:
        chunks_out = []
        parent_texts = self._smart_chunk_by_tokens(text, PARENT_CHUNK_TOKENS)
        parent_idx = 0
        for ptext in parent_texts:
            parents = self._validate_and_split(ptext)
            for parent in parents:
                parent_token_count = self._count_tokens(parent)
                parent_chunk_id = f"{document_id}_parent_{parent_idx}"
                parent_obj = {
                    "chunk_id": parent_chunk_id,
                    "chunk_type": "parent",
                    "title": title,
                    "url": url,
                    "document_id": document_id,
                    "text": parent,
                    "parent_id": None,
                    "child_ids": [],
                    "chunk_index": parent_idx,
                    "token_count": parent_token_count,
                }
                child_ids = []
                if parent_token_count > CHILD_CHUNK_TOKENS * 2:
                    child_texts = self._smart_chunk_by_tokens(parent, CHILD_CHUNK_TOKENS)
                    child_idx = 0
                    for ctext in child_texts:
                        validated_childs = self._validate_and_split(ctext)
                        for vct in validated_childs:
                            child_token_count = self._count_tokens(vct)
                            if child_token_count > MAX_API_TOKENS:
                                continue
                            child_id = f"{parent_chunk_id}_child_{child_idx}"
                            child_ids.append(child_id)
                            child_chunk = {
                                "chunk_id": child_id,
                                "chunk_type": "child",
                                "title": title,
                                "url": url,
                                "document_id": document_id,
                                "text": vct,
                                "parent_id": parent_chunk_id,
                                "child_ids": [],
                                "chunk_index": child_idx,
                                "token_count": child_token_count,
                            }
                            chunks_out.append(child_chunk)
                            child_idx += 1
                parent_obj["child_ids"] = child_ids
                chunks_out.append(parent_obj)
                parent_idx += 1
        return chunks_out

    # --------------------------
    # Scraper helpers (Playwright + BeautifulSoup)
    # --------------------------
    # Note: these are nested helpers used inside the async scrape_and_ingest function below.
    # We implement them as inner functions of scrape_and_ingest to keep self accessible via closure.

    async def scrape_and_ingest(self, max_manuals: int = 1, reporter: Optional[callable] = None):
        """
        Full async scraping & ingestion flow.
        reporter: optional callable(str) used to emit progress messages for streaming.
        """
        if async_playwright is None or BeautifulSoup is None:
            raise RuntimeError("Playwright and BeautifulSoup are required in the container.")

        def _report(msg: str):
            try:
                logger.info(msg)
                if reporter:
                    reporter(msg)
            except Exception:
                logger.debug("Reporter failed", exc_info=True)

        # small helpers (no `self` param because they capture self via closure)
        def _clean_filename(title: str) -> str:
            title = title or "untitled"
            clean = re.sub(r'[^\w\s\-\.]', '', str(title))
            clean = re.sub(r'[-\s]+', '_', clean).strip('_')
            return clean[:120] or "document"

        def _process_tables(soup: BeautifulSoup) -> str:
            table_texts = []
            for table in soup.find_all('table'):
                headers = [th.get_text(strip=True) for th in table.find_all('th')]
                rows_data = []
                for tr in table.find_all('tr'):
                    cells = [td.get_text(strip=True) for td in tr.find_all('td')]
                    if not cells:
                        continue
                    if headers and len(headers) == len(cells):
                        rows_data.append("\n".join(f"- {h}: {v}" for h, v in zip(headers, cells) if v))
                    else:
                        rows_data.append(" | ".join(cells))
                if rows_data:
                    table_texts.append("\n\n".join(rows_data))
                table.decompose()
            return "\n\n".join(table_texts)

        def _process_links(soup: BeautifulSoup, base_url: str):
            for a in soup.find_all('a', href=True):
                href = a.get('href') or ""
                text = a.get_text(strip=True) or href
                if href and not href.startswith(('#', 'javascript:')):
                    a.replace_with(f"{text} ({urljoin(base_url, href)})")
                else:
                    a.replace_with(text)

        def _clean_html_entry(entry: Dict[str, Any], base_url: str) -> Dict[str, Any]:
            html_content = entry.get('content') or entry.get('text') or ""
            soup = BeautifulSoup(html_content, "html.parser")
            table_text = _process_tables(soup)
            _process_links(soup, base_url)
            text_clean = soup.get_text(separator="\n", strip=True)
            text_clean = re.sub(r'\n\s*\n', '\n\n', text_clean).strip()
            if table_text:
                text_clean += "\n\n" + table_text
            return {"title": entry.get("title"), "url": entry.get("url") or base_url, "text_clean": text_clean}

        async def _scrape_hpe_manuals(page):
            start_url = self.valves.HPE_SUPPORT_PAGE
            await page.goto(start_url, wait_until="domcontentloaded", timeout=90000)
            await page.wait_for_timeout(1000)
            all_manuals = []
            current_page = 1
            consecutive_empty_pages = 0
            while True:
                try:
                    # Wait for the table element to appear (best-effort)
                    await page.wait_for_selector("table.slds-table", timeout=20000)
                except Exception:
                    _report(f"Page {current_page}: manuals table not found.")
                html = await page.content()
                soup = BeautifulSoup(html, "html.parser")
                table = soup.find("table", class_="slds-table")
                found_on_page = []
                if table:
                    rows = table.find_all("tr", role="row")
                    for r in rows[1:]:
                        link = r.find("a", href=True)
                        if not link:
                            continue
                        href = link.get("href")
                        title = link.get_text(strip=True)
                        if href and title:
                            if not href.startswith("http"):
                                href = urljoin(start_url, href)
                            # example metadata extraction: date columns
                            date_val = ""
                            date_cell = r.find("td", {"data-label": "Date"})
                            if date_cell:
                                date_val = date_cell.get_text(strip=True)
                            found_on_page.append({"title": title, "link": href, "date": date_val})
                if not found_on_page:
                    consecutive_empty_pages += 1
                else:
                    consecutive_empty_pages = 0
                    for m in found_on_page:
                        if m["link"] not in {x["link"] for x in all_manuals}:
                            all_manuals.append(m)
                if consecutive_empty_pages >= 2:
                    break
                # try to click next page
                next_page_num = current_page + 1
                try:
                    btn = await page.query_selector(f'a.pagination-item[data-number="{next_page_num}"]')
                    if btn:
                        await btn.click()
                        await page.wait_for_load_state("domcontentloaded", timeout=45000)
                        current_page += 1
                        await page.wait_for_timeout(1000)
                        continue
                except Exception:
                    pass
                break
            return all_manuals

        async def _toc_scraper(url, browser):
            page = await browser.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                await page.wait_for_timeout(1000)
                content = await page.content()
            except Exception as e:
                logger.debug("TOC load error", exc_info=True)
                return {"url": url, "toc": []}
            finally:
                if page and not page.is_closed():
                    await page.close()

            soup = BeautifulSoup(content, "html.parser")
            toc_container = soup.find(id="contentListArea")
            if not toc_container:
                # fallback: try top-level uls
                first_ul = soup.find("ul")
                if not first_ul:
                    return {"url": url, "toc": []}
                toc_container = first_ul

            def extract_node(node):
                title = ""
                href = None
                link = node.find("a", href=True)
                if link:
                    title = link.get_text(strip=True)
                    href = link.get("href")
                    if href and not href.startswith("http"):
                        href = urljoin(url, href)
                else:
                    span = node.find("span")
                    if span:
                        title = span.get_text(strip=True)
                return {"title": title, "href": href}

            def build(item):
                node_info = extract_node(item)
                if not node_info.get("title"):
                    return None
                res = {"title": node_info["title"], "url": node_info.get("href"), "children": []}
                child_ul = item.find("ul", recursive=False)
                if child_ul:
                    for li in child_ul.find_all("li", recursive=False):
                        if child := build(li):
                            res["children"].append(child)
                return res

            results = []
            for li in toc_container.find_all("li", recursive=False):
                if r := build(li):
                    results.append(r)
            return {"url": url, "toc": results}

        async def _scrape_document_contents(toc_data, browser):
            contents = []

            async def _scrape_node(node):
                node_url = node.get("url")
                if node_url and node_url.startswith("http"):
                    main_html = await _scrape_main_block(node_url, browser)
                    if main_html:
                        contents.append({"title": node.get("title"), "url": node_url, "content": main_html})
                    await asyncio.sleep(0.25)
                for child in node.get("children", []):
                    await _scrape_node(child)

            if isinstance(toc_data.get("toc"), list):
                for top in toc_data["toc"]:
                    await _scrape_node(top)
            return contents

        async def _scrape_main_block(url, browser):
            page = await browser.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                # give JS some time
                await page.wait_for_timeout(1000)
                # wait for main element if possible
                try:
                    await page.wait_for_selector("main", timeout=8000)
                except Exception:
                    pass
                content = await page.content()
                soup = BeautifulSoup(content, "html.parser")
                main_block = soup.find("main") or soup.find("div", id="content") or soup.body
                if not main_block:
                    return None
                return str(main_block)
            except Exception:
                logger.debug("main block scrape failed", exc_info=True)
                return None
            finally:
                try:
                    if page and not page.is_closed():
                        await page.close()
                except Exception:
                    pass

        async def _check_if_pdf(url, browser):
            page = await browser.new_page()
            found_pdf = False

            def on_response(response):
                nonlocal found_pdf
                try:
                    ct = response.headers.get("content-type", "").lower()
                    if "pdf" in ct:
                        found_pdf = True
                except Exception:
                    pass

            page.on("response", on_response)
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=45000)
                await page.wait_for_timeout(2000)
                return found_pdf
            except Exception:
                return False
            finally:
                try:
                    if page and not page.is_closed():
                        await page.close()
                except:
                    pass

        async def _download_pdf(url, browser):
            page = await browser.new_page()
            pdf_bytes = None

            def on_response(r):
                nonlocal pdf_bytes
                try:
                    ct = r.headers.get("content-type", "").lower()
                    if "pdf" in ct:
                        try:
                            pdf_bytes = r.body()
                        except Exception:
                            pass
                except Exception:
                    pass

            page.on("response", on_response)
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                await page.wait_for_timeout(2000)
                return pdf_bytes
            except Exception:
                return None
            finally:
                try:
                    if page and not page.is_closed():
                        await page.close()
                except:
                    pass

        async def _process_document(manual, browser):
            # returns dict with type and chunks
            url = manual.get("link")
            title = manual.get("title", "untitled")
            date = manual.get("date", "")
            _report(f"Processing document: {title}")
            try:
                is_pdf = await _check_if_pdf(url, browser)
                if is_pdf:
                    _report("Detected PDF; attempting download + Docling fallback (if available). PDF ingestion path not fully implemented here.")
                    pdf_bytes = await _download_pdf(url, browser)
                    # fallback - treat as no chunks if not implemented
                    if not pdf_bytes:
                        _report("PDF download failed; falling back to HTML.")
                # HTML path
                toc = await _toc_scraper(url, browser)
                if not toc.get("toc"):
                    _report("No TOC found; scraping main content.")
                    if main_html := await _scrape_main_block(url, browser):
                        cleaned = _clean_html_entry({"title": title, "url": url, "content": main_html}, url)
                        entries = [cleaned]
                    else:
                        entries = []
                else:
                    scraped = await _scrape_document_contents(toc, browser)
                    entries = [_clean_html_entry(e, url) for e in scraped if e.get("content")]
                if not entries:
                    _report("No content scraped for document.")
                    return None
                # chunk and return
                chunks = []
                doc_id_clean = _clean_filename(title)
                for idx, entry in enumerate(entries):
                    txt = entry.get("text_clean", "")
                    if not txt or len(txt) < 40:
                        continue
                    section_id = f"{doc_id_clean}_html_{idx}"
                    entry_chunks = self._create_hierarchical_chunks(txt, entry.get("title", title), entry.get("url", url), section_id)
                    for c in entry_chunks:
                        c.update({"date": date, "document_title": title, "document_url": url})
                    chunks.extend(entry_chunks)
                _report(f"Document chunked: {len(chunks)} chunks")
                return {"type": "html", "title": title, "url": url, "date": date, "chunks": chunks}
            except Exception as e:
                _report(f"Error processing document {title}: {e}")
                logger.debug(traceback.format_exc())
                return None

        # Main scraping & ingest flow
        reporter and reporter("Scraper: starting Playwright browser...")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
            page = await browser.new_page()
            manuals = await _scrape_hpe_manuals(page)
            await page.close()

            if not manuals:
                reporter and reporter("Scraper: no manuals found.")
                await browser.close()
                return

            reporter and reporter(f"Scraper: found {len(manuals)} manuals; processing up to {max_manuals}.")
            all_chunks = []
            processed_docs = 0
            for manual in manuals[:max_manuals]:
                processed_docs += 1
                doc_result = await _process_document(manual, browser)
                if doc_result and doc_result.get("chunks"):
                    all_chunks.extend(doc_result["chunks"])
                await asyncio.sleep(0.5)

            await browser.close()

        if not all_chunks:
            reporter and reporter("Scraper: no chunks produced.")
            return

        reporter and reporter(f"Scraper: generated total {len(all_chunks)} chunks. Uploading to Weaviate...")

        # ensure weaviate client
        self.ensure_weaviate_client()
        if not self.weaviate_client:
            reporter and reporter("Scraper: could not connect to Weaviate for upload.")
            return

        # create collection if doesn't exist (v4 client path)
        try:
            col = self.weaviate_client.collections.get(self.valves.COLLECTION_NAME)
            reporter and reporter(f"Scraper: collection exists: {self.valves.COLLECTION_NAME}")
        except Exception:
            reporter and reporter(f"Scraper: creating collection {self.valves.COLLECTION_NAME} ...")
            try:
                # build collection schema using v4 API helpers (weaviate_config)
                props = [
                    weaviate_config.Property(name="chunk_id", data_type=weaviate_config.DataType.TEXT),
                    weaviate_config.Property(name="chunk_type", data_type=weaviate_config.DataType.TEXT),
                    weaviate_config.Property(name="text", data_type=weaviate_config.DataType.TEXT),
                    weaviate_config.Property(name="title", data_type=weaviate_config.DataType.TEXT),
                    weaviate_config.Property(name="url", data_type=weaviate_config.DataType.TEXT),
                    weaviate_config.Property(name="document_id", data_type=weaviate_config.DataType.TEXT),
                    weaviate_config.Property(name="parent_id", data_type=weaviate_config.DataType.TEXT),
                    weaviate_config.Property(name="child_ids", data_type=weaviate_config.DataType.TEXT_ARRAY),
                    weaviate_config.Property(name="chunk_index", data_type=weaviate_config.DataType.INT),
                    weaviate_config.Property(name="document_title", data_type=weaviate_config.DataType.TEXT),
                    weaviate_config.Property(name="document_date", data_type=weaviate_config.DataType.TEXT),
                    weaviate_config.Property(name="document_url", data_type=weaviate_config.DataType.TEXT),
                ]
                manual_chunks = self.weaviate_client.collections.create(
                    name=self.valves.COLLECTION_NAME,
                    properties=props,
                    vectorizer_config=weaviate_config.Configure.Vectorizer.none(),
                )
                reporter and reporter("Scraper: created collection.")
                self.collection_exists = True
            except Exception as e:
                reporter and reporter(f"Scraper: failed to create collection: {e}")
                logger.debug(traceback.format_exc())
                return

        # perform batched insertion using the v4 client API pattern used earlier in your code
        try:
            collection_obj = self.weaviate_client.collections.get(self.valves.COLLECTION_NAME)
            batch = collection_obj.batch
            # v4: use manual_chunks.batch.fixed_size - but interface can differ; attempt a general approach
            with batch.fixed_size(batch_size=10) as b:
                for i, chunk in enumerate(all_chunks):
                    try:
                        # create embedding for chunk text
                        embedding = self._get_embedding_for_text(chunk.get("text", ""))
                        if embedding is None:
                            raise RuntimeError("Failed to create embedding")
                        properties = {
                            "chunk_id": chunk.get("chunk_id"),
                            "chunk_type": chunk.get("chunk_type"),
                            "text": chunk.get("text"),
                            "title": chunk.get("title"),
                            "url": chunk.get("url"),
                            "document_id": chunk.get("document_id"),
                            "parent_id": chunk.get("parent_id") or "",
                            "child_ids": chunk.get("child_ids", []),
                            "chunk_index": chunk.get("chunk_index", -1),
                            "document_title": chunk.get("document_title", ""),
                            "document_date": chunk.get("date", ""),
                            "document_url": chunk.get("document_url", ""),
                        }
                        b.add_object(collection=self.valves.COLLECTION_NAME, properties=properties, vector=embedding.tolist())
                        if (i + 1) % 50 == 0:
                            reporter and reporter(f"Uploaded {i+1}/{len(all_chunks)} chunks...")
                    except Exception as e:
                        logger.debug(f"Failed adding chunk to batch: {e}")
                        continue
            reporter and reporter(f"Scraper: upload finished. Uploaded approx {len(all_chunks)} chunks.")
        except Exception as e:
            reporter and reporter(f"Scraper: upload error: {e}")
            logger.debug(traceback.format_exc())

    # --------------------------
    # Embedding helpers
    # --------------------------
    def _get_embedding_for_text(self, text: str) -> Optional["np.ndarray"]:
        """
        Call embedding endpoint (NVIDIA) — adapt as needed. Returns numpy array or None.
        We keep it small here and rely on same pattern used in original files.
        """
        if not self.valves.EMBEDDING_MODEL_API_KEY:
            logger.error("No embedding API key set.")
            return None
        headers = {
            "Authorization": f"Bearer {self.valves.EMBEDDING_MODEL_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {"input": text, "model": self.valves.EMBEDDING_MODEL_NAME, "input_type": "passage"}
        try:
            r = requests.post(f"{self.valves.EMBEDDING_MODEL_URL}/embeddings", headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json().get("data", [])
            if data:
                return np.array(data[0]["embedding"], dtype=np.float32)
            logger.error("Embedding response missing data.")
            return None
        except Exception as e:
            logger.error(f"Embedding call failed: {e}")
            return None

    # --------------------------
    # Search + context creation
    # --------------------------
    def search_chunks(self, query: str, k: Optional[int] = None, target_version: Optional[str] = None) -> List[Dict[str, Any]]:
        if k is None:
            k = self.valves.TOP_K
        # ensure weaviate
        self.ensure_weaviate_client()
        if not self.weaviate_client:
            logger.error("Weaviate not available for search.")
            return []
        if not self.collection_exists:
            logger.warning("Collection missing. No search possible.")
            return []

        try:
            qembed = self._get_embedding_for_text(query)
            if qembed is None:
                logger.error("Failed to generate query embedding.")
                return []
            collection = self.weaviate_client.collections.get(self.valves.COLLECTION_NAME)
            where_filter = None
            if target_version:
                where_filter = (
                    Filter.by_property("document_id").like(f"*{target_version}*")
                    | Filter.by_property("document_title").like(f"*{target_version}*")
                )
            res = collection.query.near_vector(
                near_vector=qembed.tolist(),
                limit=k * 3,
                return_properties=[
                    "chunk_id",
                    "chunk_type",
                    "text",
                    "title",
                    "url",
                    "parent_id",
                    "chunk_index",
                    "document_id",
                    "document_title",
                    "document_date",
                ],
                return_metadata=None,
                filters=where_filter,
            )
            objects = [o.properties for o in res.objects] if getattr(res, "objects", None) else []
            chunks = []
            # prioritize child chunks, then parent fallback
            for obj in objects:
                if obj.get("chunk_type") == "child" and len(obj.get("text", "").strip()) > 1:
                    chunks.append(obj)
                    if len(chunks) >= k:
                        break
            if len(chunks) < k:
                existing = {c["chunk_id"] for c in chunks}
                for obj in objects:
                    if obj.get("chunk_type") == "parent" and obj.get("chunk_id") not in existing and len(obj.get("text", "").strip()) > 100:
                        obj["chunk_type"] = "parent_fallback"
                        chunks.append(obj)
                        if len(chunks) >= k:
                            break
            logger.info(f"Retrieved {len(chunks)} chunks.")
            return chunks[:k]
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []

    def create_context(self, chunks: List[Dict[str, Any]], query_version: Optional[str] = None) -> str:
        if not chunks:
            return ""
        parts = []
        for i, ch in enumerate(chunks, 1):
            title = ch.get("title", "Unknown")
            text = ch.get("text", "")
            url = ch.get("url", "N/A")
            parts.append(f"[Source {i}: {title} | URL: {url}]\n{text}")
        return "\n\n---\n\n".join(parts)

    # --------------------------
    # LLM call (with streaming safety)
    # --------------------------
    def generate_with_llm(self, prompt: str, stream: bool = False, idle_timeout: int = 30) -> Union[str, Iterator[str]]:
        """
        Calls an LLM endpoint (LLM_MODEL_URL) and returns either the string or a generator that yields strings.
        The streaming generator includes an idle timeout to avoid infinite loops.
        """
        headers = {"Authorization": f"Bearer {self.valves.LLM_MODEL_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": self.valves.LLM_MODEL_NAME, "messages": [{"role": "system", "content": prompt}], "temperature": 0.1}
        try:
            r = requests.post(f"{self.valves.LLM_MODEL_URL}/chat/completions", headers=headers, json=payload, stream=stream, timeout=120)
            r.raise_for_status()
            if stream:
                def gen():
                    last_activity = time.time()
                    max_idle = idle_timeout
                    for line in r.iter_lines():
                        if line:
                            last_activity = time.time()
                            try:
                                chunk = json.loads(line)
                                # adapt to provider format; try common keys
                                if isinstance(chunk, dict):
                                    if "response" in chunk:
                                        yield chunk["response"]
                                    elif "choices" in chunk:
                                        # standard OpenAI-like stream: extract delta content if available
                                        for c in chunk["choices"]:
                                            delta = c.get("delta", {})
                                            if isinstance(delta, dict) and delta.get("content"):
                                                yield delta.get("content")
                                            elif c.get("text"):
                                                yield c.get("text")
                                else:
                                    # fallback: yield raw text
                                    yield line.decode() if isinstance(line, bytes) else str(line)
                            except Exception:
                                # if parsing fails, yield raw line
                                try:
                                    yield line.decode() if isinstance(line, bytes) else str(line)
                                except:
                                    pass
                        # idle timeout check
                        if time.time() - last_activity > max_idle:
                            yield "[Stream aborted: idle timeout]"
                            break
                    try:
                        r.close()
                    except:
                        pass
                return gen()
            else:
                return r.json().get("response", r.text)
        except Exception as e:
            logger.error("LLM generation error", exc_info=True)
            return f"Error generating response: {e}"

    # --------------------------
    # Pipe: entry point Open-WebUI calls
    # --------------------------
    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator[str, None, None]]:
        """
        Called by Open-WebUI when a user message is routed to this pipeline.
        - Messages containing 'scrape' or 'update database' trigger scraping & ingest.
          When body['stream'] is true, returns a generator that streams progress updates.
        - Otherwise the pipeline performs RAG retrieval + generation and returns a string or generator.
        """
        logger.info(f"pipe called; message preview: {user_message[:120]}")

        stream_requested = bool(body.get("stream", False))

        lowered = (user_message or "").lower()
        if "scrape" in lowered or "update database" in lowered:
            # Start a background scraping thread and return a streaming generator (if requested)
            if stream_requested:
                q = queue.Queue()

                def reporter(msg: str):
                    try:
                        q.put_nowait(msg)
                    except Exception:
                        pass

                # background runner using an asyncio loop in thread
                def _bg_runner():
                    try:
                        # run the coroutine
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        coro = self.scrape_and_ingest(max_manuals=1, reporter=reporter)
                        new_loop.run_until_complete(coro)
                        reporter("DONE:Scraping completed.")
                        new_loop.close()
                    except Exception as e:
                        reporter(f"ERROR:Scraper failed: {e}")
                        logger.debug(traceback.format_exc())

                th = threading.Thread(target=_bg_runner, daemon=True)
                th.start()
                self._scrape_threads.append(th)

                # generator to stream messages from q to Open-WebUI
                def stream_gen():
                    while True:
                        try:
                            item = q.get(timeout=30)
                        except queue.Empty:
                            # heartbeat so UI doesn't time out
                            yield " \n"
                            continue
                        if isinstance(item, str) and item.startswith("DONE:"):
                            yield item.split("DONE:", 1)[1]
                            break
                        if isinstance(item, str) and item.startswith("ERROR:"):
                            yield item.split("ERROR:", 1)[1]
                            break
                        yield item

                return stream_gen()

            else:
                # non-streaming: start background thread and return immediate message
                def _bgfire_and_forget():
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(self.scrape_and_ingest(max_manuals=1))
                        loop.close()
                    except Exception:
                        logger.exception("Background scrape failed.")

                t = threading.Thread(target=_bgfire_and_forget, daemon=True)
                t.start()
                return "Scraping started in background. The database will be updated shortly."

        # Normal RAG retrieval flow
        try:
            self.ensure_weaviate_client()
            if not self.weaviate_client:
                return "Error: Could not connect to Weaviate. Please try again later."

            # Determine search criteria (a reduced version of your logic)
            # We'll just search with the user_message itself
            chunks = self.search_chunks(user_message, k=self.valves.TOP_K)
            if not chunks:
                return "I couldn't find relevant documentation for that question. Try rephrasing or run 'scrape' first."

            context = self.create_context(chunks)
            # Build a concise system + user prompt
            prompt = f"You are a technical doc assistant. Use the DOCUMENTATION CONTEXT below to answer the user's question.\n\nDOCUMENTATION CONTEXT:\n{context}\n\nUSER QUESTION:\n{user_message}\n\nAnswer concisely and reference sources if appropriate."

            result = self.generate_with_llm(prompt, stream=stream_requested, idle_timeout=25)
            return result

        except Exception as e:
            logger.error("Error in pipe()", exc_info=True)
            return f"Pipeline error: {e}"

    # --------------------------
    # Lifecycle hooks (optional)
    # --------------------------
    async def on_startup(self):
        logger.info("RAG pipeline startup: ensuring weaviate client")
        try:
            # spawn connection attempt but don't block heavily
            self.ensure_weaviate_client(max_retries=3, delay=2)
        except Exception:
            logger.debug("Startup connect attempt failed", exc_info=True)

    async def on_shutdown(self):
        logger.info("RAG pipeline shutdown: closing weaviate client if present")
        try:
            if self.weaviate_client:
                try:
                    self.weaviate_client.close()
                except Exception:
                    pass
                self.weaviate_client = None
        except Exception:
            logger.debug("Error closing weaviate client", exc_info=True)
