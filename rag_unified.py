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
from datetime import datetime
from typing import List, Dict, Any, Optional, Iterator, Union, Generator
from urllib.parse import urljoin, urlparse
from pathlib import Path
from pydantic import BaseModel, Field
try:
    from playwright.async_api import async_playwright
except ImportError:
    async_playwright = None
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
try:
    import tiktoken
except ImportError:
    tiktoken = None
try:
    import weaviate
    from weaviate.classes import config as weaviate_config
    from weaviate.classes.query import Filter
except ImportError:
    weaviate = None
    weaviate_config = None
    Filter = None
try:
    from docling.document_converter import DocumentConverter
except ImportError:
    DocumentConverter = None

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

# PDF-related constants
HPE_BASE_URL = "https://support.hpe.com"
MIN_PDF_SIZE_BYTES = 1024  # Minimum PDF size in bytes

# basic directories 
PDF_STORAGE_DIR = Path("/app/data_pdfs")
PDF_CLEAN_DIR = Path("/app/data_clean/manuals_pdf")

# create directories only if the parent directory exists and is writable
try:
    if PDF_STORAGE_DIR.parent.exists() and os.access(PDF_STORAGE_DIR.parent, os.W_OK):
        PDF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        PDF_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
except (OSError, PermissionError):
    # Will be created at runtime in container
    pass

def check_if_document_exists(client, document_url: str, collection_name="HierarchicalManualChunks") -> bool:
    """
    Check if a document with the given URL already exists in the Weaviate collection.
    Returns True if document exists, False otherwise.
    """
    if not client or not client.is_ready():
        logger.warning("Weaviate client is not ready.")
        return False
    
    try:
        if not client.collections.exists(collection_name):
            logger.info(f"Collection {collection_name} does not exist yet.")
            return False

        collection = client.collections.get(collection_name)
        
        response = collection.query.fetch_objects(
            filters=Filter.by_property("document_url").equal(document_url), 
            limit=1,
            return_properties=[]
        )
        
        exists = len(response.objects) > 0
        if exists:
            logger.info(f"Document with URL {document_url} already exists in Weaviate.")
        return exists
            
    except Exception as e:
        logger.error(f"Error checking if document exists (URL: {document_url}): {e}")
        return False


class PDFContentExtractor:
    """
    [__init__] --> [extract_content_from_pdf] -> extract text [ _extract_text_item ] -> extract tables [ _extract_table / _extract_tables_from_markdown ] -> combine texts [ _combine_text_fragments ] -> convert table to sentences [ convert_table_to_sentences ]
    parse row [ _parse_row ] -> parse markdown table [ _parse_markdown_table ] -> convert table to sentences [ convert_table_to_sentences ]
    """ 
    def __init__(self, output_dir: Path):
        # dir inside the container
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # initialize DocumentConverter
        if DocumentConverter is None:
            logger.warning("DocumentConverter (docling) is not available. PDF processing will be limited.")
            self.converter = None
        else:
            try: 
                self.converter = DocumentConverter()
            except Exception as e: 
                logger.error(f"Error initializing DocumentConverter: {e}")
                self.converter = None

    def extract_content_from_pdf(self, pdf_path: Path, manual_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured content (text and tables) from a PDF document using Docling.
        Args: Path to the PDF file, dictionary containing metadata (title, url, date)
        Returns: Dictionary containing extracted texts, tables, and metadata
        """
        if self.converter is None: 
            raise Exception("DocumentConverter not initialized.")
        
        try:
            # convert PDF to Docling document
            result = self.converter.convert(pdf_path)
            doc = result.document
            
            #metadata
            pdf_metadata = {
                'date': manual_metadata.get('date', ''), 
                'title': manual_metadata.get('title', pdf_path.stem),
                'url': manual_metadata.get('url', ''), 
                'scraping_source': 'hpe_support_website'
            }
            content_data = {
                'document_metadata': pdf_metadata, 
                'texts': [], 
                'tables': []
            }

            # extract text content
            if hasattr(doc, 'texts') and doc.texts:
                raw_texts = [self._extract_text_item(item, i+1, pdf_path.name) for i, item in enumerate(doc.texts)]
                content_data['texts'] = self._combine_text_fragments([t for t in raw_texts if t])

            # extract tables
            tables = []
            if hasattr(doc, 'tables') and doc.tables:
                tables = [tbl for i, table in enumerate(doc.tables) if (tbl := self._extract_table(table, i+1, pdf_path.name, 'docling'))]
            
            # fallback: try markdown extraction if no tables found
            if not tables and hasattr(doc, 'export_to_markdown'):
                try: 
                    markdown_content = doc.export_to_markdown()
                    if markdown_content and len(markdown_content) > 100:
                        tables = self._extract_tables_from_markdown(markdown_content, pdf_path.name)
                except Exception:
                    pass
            
            content_data['tables'] = tables

            # save extraction results
            output_path = self.output_dir / f"{pdf_path.stem}_content.json"
            try:
                with open(output_path, 'w', encoding='utf-8') as f: 
                    json.dump(content_data, f, indent=2, ensure_ascii=False)
            except Exception as e: 
                logger.warning(f"Could not save extraction results: {e}")
            
            # returns dict with text, tabels and metadata
            return content_data
        
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name} with Docling: {e}")
            return {
                'document_metadata': {
                    'date': manual_metadata.get('date', ''), 
                    'title': manual_metadata.get('title', pdf_path.stem),
                    'url': manual_metadata.get('url', ''), 
                    'scraping_source': 'hpe_support_website'
                }, 
                'texts': [], 
                'tables': []
            }

    def _extract_text_item(self, text_item, text_num: int, source_filename: str) -> Dict[str, Any] | None:
        """
        Extract and process a single text item from a Docling document.
        Args: Docling text object to process, sequential number for text ID generation, name of source PDF file
        Returns: Dictionary with text content, metadata and page info, or None if filtered out
        """
        try:
            # extract text content
            content = str(getattr(text_item, 'text', '') or getattr(text_item, 'orig', '')).strip()
            # extract label
            label = str(getattr(text_item, 'label', ''))
            # page number
            page_num = 1
            if hasattr(text_item, 'prov') and text_item.prov:
                try: 
                    page_num = next((p.page_no for p in text_item.prov if hasattr(p, 'page_no') and p.page_no), 1)
                except: 
                    pass
            # filter out very short or irrelevant texts
            if (len(content) < 3 or label in ['page_header', 'page_footer'] or content.startswith('Page ')
                or content in ['QuickSpecs', 'HPE Private Cloud AI', 'Overview']): 
                return None
            # rethurn structured text item
            return {
                'text_id': f"text_{text_num}", 
                'content': content, 
                'label': label,
                'page_number': page_num, 
                'source_document': source_filename
            }
        except: 
            return None

    def _extract_table(self, table, table_num: int, source_filename: str, method: str) -> Dict[str, Any] | None:
        """
        Extract and process table data from Docling document into structured format. Supports dual extraction: grid-based (primary) and text-based (fallback). 
        Handles header detection, row processing, and column normalization.
        Args: Docling table object to process, sequential number for table ID generation, name of source PDF file, extraction method used
        Returns: Dictionary with table data, metadata and page info, or None if table is empty or invalid
        """
        # page number
        try:
            page_num = 1
            if hasattr(table, 'prov') and table.prov:
                try: 
                    page_num = next((p.page_no for p in table.prov if hasattr(p, 'page_no') and p.page_no), 1)
                except: 
                    pass
            
            # handle Docling table objects with grid structure
            if hasattr(table, 'data') and hasattr(table.data, 'grid'):
                grid = table.data.grid
                if not grid:
                    return None
                
                # extract headers from first row
                headers = []
                if grid:
                    first_row = grid[0]
                    for cell in first_row:
                        cell_text = cell.text if hasattr(cell, 'text') else str(cell)
                        headers.append(cell_text.strip() if cell_text and cell_text.strip() else "")
                
                # extract data rows (skip header row)
                rows = []
                for row in grid[1:]:
                    row_data = []
                    for cell in row:
                        cell_text = cell.text if hasattr(cell, 'text') else str(cell)
                        row_data.append(cell_text.strip() if cell_text else "")
                    
                    # only add rows that have some non-empty content
                    if any(cell.strip() for cell in row_data):
                        rows.append(row_data)
                
                # if no proper headers found, create generic ones
                if not headers and rows:
                    max_cols = max(len(row) for row in rows) if rows else 0
                    headers = [f"Column {i+1}" for i in range(max_cols)]
                elif headers and not any(h.strip() for h in headers) and rows:
                    headers = [f"Column {i+1}" for i in range(len(headers))]
                
                # ensure all rows have the same number of columns as headers
                if headers and rows:
                    for i, row in enumerate(rows):
                        while len(row) < len(headers):
                            row.append("")
                        if len(row) > len(headers):
                            rows[i] = row[:len(headers)]
            
            # fallback to text-based extraction
            else:
                # text extraction
                text_content = getattr(table, 'text', lambda: str(table))() if callable(getattr(table, 'text', None)) else str(table)
                if not text_content or len(text_content.strip()) < 3: 
                    return None
                # parse text into lines, no empty space
                lines = [line for line in text_content.strip().split('\n') if line.strip()]
                if not lines: 
                    return None
                # find headers 
                headers = self._parse_row(lines[0])
                start_row_index = 1
                # check for separator line
                if len(lines) > 1 and all(c in '- |+' for c in lines[1].strip()): 
                    # skip separator line
                    start_row_index = 2
                # create generic headers if first row is not suitable
                elif not headers: 
                    headers = [f"Col_{i+1}" for i in range(len(self._parse_row(lines[0])))] if lines else []
                    start_row_index = 0
                # extract data rows
                rows = [row for line in lines[start_row_index:] if (row := self._parse_row(line))]
                # ensure all rows have same number of columns as headers
                if headers and rows:
                    # pad rows with fewer columns
                    max_row_len = max((len(r) for r in rows), default=0)
                    # no more columns than headers, so make less headers
                    if len(headers) > max_row_len > 0: 
                        headers = headers[:max_row_len]
            
            if not headers and not rows:
                return None
                
            return {
                'table_id': f"table_{table_num}", 
                'page_number': page_num, 
                'source_document': source_filename,
                'headers': headers, 
                'rows': rows, 
                'extraction_method': method
            }
        except Exception as e: 
            logger.warning(f"Error parsing table {table_num}: {e}")
            return None

    def _extract_tables_from_markdown(self, markdown_content: str, source_filename: str) -> List[Dict[str, Any]]:
        """
        Extract tables from markdown-formatted content. Supports dual extraction: grid-based (primary) and text-based (fallback). 
        Parses pipe-delimited markdown tables as final fallback when other extraction methods fail. Processes multiple tables from single document.
        Args: full markdown text with potential tables, name of source PDF file
        Returns: List of table dictionaries with headers, rows, and metadata or empty list if no valid tables found
        """
        tables = []
        lines = markdown_content.split('\n')
        current_table_lines = []
        table_count = 0
        in_table = False
        # iterate through lines to find markdown tables
        for line in lines:
            line = line.strip()
            # detect start of table
            if line.startswith('|') and line.endswith('|'): 
                current_table_lines.append(line)
                in_table = True
            elif in_table:
                # end of current table
                if len(current_table_lines) >= 2:
                    if table_data := self._parse_markdown_table(current_table_lines, table_count + 1, source_filename): 
                        tables.append(table_data)
                        table_count += 1
                # reset for next table
                current_table_lines = []
                in_table = False
        if in_table and len(current_table_lines) >= 2:
             if table_data := self._parse_markdown_table(current_table_lines, table_count + 1, source_filename): 
                 tables.append(table_data)
        return tables

    def _parse_markdown_table(self, lines: List[str], table_num: int, source_filename: str) -> Dict[str, Any] | None:
        """
        Parse individual markdown table from collected table lines into structured format.
        Processes pipe-delimited markdown table lines, extracts headers and data rows, and handles markdown separator lines. 
        Creates normalized table structure with consistent column count across all rows.
        Args: list of lines representing the markdown table, sequential number for table ID generation, name of source PDF file
        Returns: Dictionary with table data, metadata and page info, or None if table is empty or invalid
        """
        try:
            if not lines or len(lines) < 1: 
                return None
            headers = [h.strip() for h in lines[0].strip('|').split('|')]
            start_idx = 2 if len(lines) > 1 and all(c in '|-: ' for c in lines[1]) else 1
            rows = []
            for line in lines[start_idx:]:
                if row_cells := [cell.strip() for cell in line.strip('|').split('|')]: 
                    rows.append(row_cells + [''] * (len(headers) - len(row_cells)))
            if not headers and not rows: 
                return None
            return {
                'table_id': f"table_{table_num}", 
                'page_number': 1, 
                'source_document': source_filename,
                'headers': headers, 
                'rows': [r[:len(headers)] for r in rows], 
                'extraction_method': 'markdown_fallback'
            }
        except Exception as e: 
            logger.warning(f"Error parsing markdown table {table_num}: {e}")
            return None

    def _parse_row(self, text: str) -> List[str]:
        """
        Parse a single text line into table columns using intelligent format detection.
        Used by text-based table extraction when grid-based parsing fails.
        Automatically detects and handles multiple table formats: pipe-delimited, tab-separated, and multi-space separated text. Provides flexible parsing
        for various table layouts found in PDF extractions.
        Args: single line of text representing a table row
        Returns: List of strings representing individual cell values. Returns empty list if text is empty or whitespace-only. 
        Returns single-item list if no separators detected.
        """
        # no whitespace-only text
        text = text.strip()
        if not text: 
            return []
        # separate by pipe, tab, or multiple spaces
        if '|' in text and len(text.split('|')) > 1: 
            return [p.strip() for p in text.strip('|').split('|')]
        if '\t' in text: 
            return [p.strip() for p in text.split('\t') if p.strip()]
        if re.search(r'\s{2,}', text): 
            return [p.strip() for p in re.split(r'\s{2,}', text) if p.strip()]
        return [text]

    def _combine_text_fragments(self, texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Combine small text fragments from PDF extraction into larger, coherent text blocks.
        Args: List of text fragment dictionaries with content, labels, and page numbers.
        Returns: List of combined text fragment dictionaries.
        """
        if not texts: 
            return []
        combined = []
        current = None
        for item in texts:
            # clean content
            content = item['content'].strip()
            if not content or len(content) < 5: 
                continue
            # decide whether to start a new fragment
            start_new = (current is None or current['page_number'] != item['page_number'] or
                         len(current['content']) > 2500 or (item['label'] == 'section_header' and len(content.split()) <= 10))
            if start_new:
                if current and len(current['content']) > 30: 
                    combined.append(current)
                current = item.copy()
            else: 
                # append to current fragment
                current['content'] += ('\n' if content.startswith(tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-*>#')) else ' ') + content
        if current and len(current['content']) > 30: 
            combined.append(current)
        return combined

    def convert_table_to_sentences(self, table: Dict[str, Any], document_title: str = "") -> str:
        """
        Convert table data to natural language sentences for better LLM comprehension.
        Transforms structured table data (headers/rows) into readable sentences optimized for LLM processing. Includes specialized handling for HPE compatibility matrices 
        with intelligent component/version detection and natural language generation.
        Args: Table dictionary with 'headers' and 'rows', optional document title for context
        Returns: String containing generated sentences representing the table data
        """
        headers = table.get('headers', [])
        rows = table.get('rows', [])
        
        # clean headers, remove any object representations
        clean_headers = []
        for h in headers:
            if isinstance(h, str):
                clean_h = h.replace('\n', ' ').replace('|', '').strip()
                if clean_h and not clean_h.startswith("self_ref="):
                    clean_headers.append(clean_h)
                else:
                    clean_headers.append("")
            else:
                clean_headers.append("")
        
        # if we don't have clean headers and we have rows, create generic ones
        if (not clean_headers or not any(h.strip() for h in clean_headers)) and rows:
            num_cols = max(len(row) for row in rows) if rows else 0
            clean_headers = [f"Column {i+1}" for i in range(num_cols)]
        
        if not clean_headers or not rows:
            return ""
        
        sentences = []
        
        # identify table patterns for better sentence generation
        document_title_lower = document_title.lower()
        is_compatibility_matrix = ("compatibility matrix" in document_title_lower or 
                                 "firmware" in document_title_lower or
                                 "software compatibility" in document_title_lower or
                                 any("version" in h.lower() for h in clean_headers))
        is_hpe_matrix = ("hpe" in document_title_lower or "pcai" in document_title_lower)
        
        # special handling for HPE compatibility matrices
        if is_compatibility_matrix and is_hpe_matrix:
            # analyze table structure to identify component and version columns
            for row_idx, row in enumerate(rows):
                if not row:
                    continue
                    
                cleaned_row = [str(cell).replace('\n', ' ').replace('|', '').strip() for cell in row]
                
                # ensure we have the same number of columns
                while len(cleaned_row) < len(clean_headers):
                    cleaned_row.append("")
                
                # skip section headers: rows where first column has text but second column is empty,
                # and the first column text doesn't look like a component name with version info
                first_col = cleaned_row[0] if len(cleaned_row) > 0 else ""
                second_col = cleaned_row[1] if len(cleaned_row) > 1 else ""
                
                if (first_col and 
                    not second_col and 
                    not self._is_version_like(first_col) and 
                    any(word in first_col.lower() for word in ['software', 'firmware', 'drivers', 'pack'])):
                    # This is likely a section header, skip processing
                    continue
                
                # skip rows with no meaningful data
                non_empty_cells = [cell for cell in cleaned_row if cell and cell.lower() not in ['n/a', 'na', '-', '']]
                if len(non_empty_cells) == 0:
                    continue
                
                # component and version detection
                component = None
                version_info = []
                
                for i, (header, value) in enumerate(zip(clean_headers, cleaned_row)):
                    if not value or value.lower() in ['n/a', 'na', '-', '']:
                        continue
                    
                    header_lower = header.lower() if header else ""
                    
                    # for tables with empty first column header,
                    # treat the first column as component names unless the value looks like a version
                    if i == 0 and header_lower == "" and not self._is_version_like(value):
                        # first column with empty header and non-version value = component name
                        component = value
                        continue
                    
                    # detect if this is likely a version column
                    is_version_column = (
                        "version" in header_lower or 
                        "pcai" in header_lower or
                        "build" in header_lower or
                        (header_lower != "" and header_lower in ["value", "column 2"])  # Exclude empty headers
                    )
                    
                    # check if the value looks like a version
                    looks_like_version = self._is_version_like(value)
                    
                    if component is None and not (is_version_column or looks_like_version):
                        # this is then the component name
                        component = value
                    elif is_version_column or looks_like_version:
                        # this contains version information
                        version_type = self._classify_version_type(value)
                        if header and header.strip():
                            version_info.append((header.strip(), value, version_type))
                        else:
                            # generic version column
                            version_info.append(("version", value, version_type))
                
                # build sentence with intelligent version handling
                if component:
                    sentence_parts = [component]
                    
                    # process version info in original order
                    for header, value, version_type in version_info:
                        if version_type == "main_version":
                            sentence_parts.append(f"has version {value}")
                        elif version_type == "build":
                            sentence_parts.append(f"with build {value}")
                        elif version_type == "release":
                            sentence_parts.append(f"release {value}")
                        elif version_type == "custom":
                            sentence_parts.append(f"with custom version {value}")
                        elif version_type == "firmware":
                            sentence_parts.append(f"with firmware {value}")
                        elif "date" in header.lower():
                            sentence_parts.append(f"from {value}")
                        elif "support" in header.lower() or "compatible" in header.lower():
                            sentence_parts.append(f"supporting {value}")
                        elif "requirement" in header.lower():
                            sentence_parts.append(f"requiring {value}")
                        else:
                            sentence_parts.append(f"with {header} {value}")
                    
                    sentences.append(" ".join(sentence_parts))
        
        else:
            # standard table processing for non-compatibility matrices
            for row_idx, row in enumerate(rows):
                if not row or all(not str(cell).strip() for cell in row):
                    continue
                    
                cleaned_row = [str(cell).replace('\n', ' ').replace('|', '').strip() for cell in row]
                
                # ensure we have the same number of columns
                while len(cleaned_row) < len(clean_headers):
                    cleaned_row.append("")
                
                # generate natural language sentences
                if len(clean_headers) == 2:
                    # Two-column tables
                    if cleaned_row[0] and cleaned_row[1]:
                        sentences.append(f"{clean_headers[0]} {cleaned_row[0]} has {clean_headers[1]} {cleaned_row[1]}")
                
                elif len(clean_headers) >= 3:
                    # multi-column tables
                    primary_item = cleaned_row[0] if cleaned_row[0] else f"Item {row_idx + 1}"
                    sentence_parts = [primary_item]
                    
                    for i, (header, value) in enumerate(zip(clean_headers[1:], cleaned_row[1:]), 1):
                        if value and value.lower() not in ['n/a', 'na', '-', '']:
                            header_lower = header.lower()
                            if "version" in header_lower:
                                sentence_parts.append(f"has {header} {value}")
                            elif any(keyword in header_lower for keyword in ["specification", "spec", "parameter", "property", "feature"]):
                                sentence_parts.append(f"has {header} of {value}")
                            else:
                                sentence_parts.append(f"has {header} {value}")
                    
                    if len(sentence_parts) > 1:
                        sentences.append(" ".join(sentence_parts))
                
                # fallback for complex tables
                if not sentences or len(sentences) <= row_idx:
                    non_empty_pairs = [(h, v) for h, v in zip(clean_headers, cleaned_row) if v and v.lower() not in ['n/a', 'na', '-', '']]
                    if non_empty_pairs:
                        if len(non_empty_pairs) == 1:
                            sentences.append(f"{non_empty_pairs[0][0]}: {non_empty_pairs[0][1]}")
                        else:
                            pairs_text = ", ".join([f"{h} {v}" for h, v in non_empty_pairs])
                            sentences.append(f"Record with {pairs_text}")
        
        return ". ".join(sentences) + "." if sentences else ""

    def _is_version_like(self, text: str) -> bool:
        """
        Check if a text string looks like a version number or build identifier.
        Args: text string to evaluate
        Returns: True if text resembles a version/build, False otherwise
        """
        if not text:
            return False
        
        text = str(text).strip()
        
        # HPE patterns for version detection
        hpe_patterns = [
            re.compile(r'(?:version\s+|v\.?\s*|release\s+)?(\d+\.\d+(?:\.\d+)*(?:\.x)?)', re.IGNORECASE),
            re.compile(r'(?:hpe\s+)?(?:ai\s+essentials\s+software\s+)?(?:private\s+cloud\s+ai\s+)?(?:v\.?\s*)?(\d+\.\d+(?:\.\d+)*)', re.IGNORECASE),
            re.compile(r'essentials\s+software\s+(?:version\s+)?(\d+\.\d+(?:\.\d+)*)', re.IGNORECASE),
            re.compile(r'pcai\s+(?:v\.?\s*)?(\d+\.\d+(?:\.\d+)*)', re.IGNORECASE),
            re.compile(r'pcai\s+(\d+\.\d+(?:\.\d+)*)\s*\([^)]+\)', re.IGNORECASE),  # PCAI 1.6.1 (G2 N/L)
        ]
        
        general_patterns = [
            re.compile(r'(?:version\s+|v\.?\s*)(\d+\.\d+(?:\.\d+)*(?:\.x)?)', re.IGNORECASE),
            re.compile(r'(\d+\.\d+(?:\.\d+)*(?:\.x)?)(?:\s*lts)?', re.IGNORECASE),
            re.compile(r'(\d+\.\d+)\s*\(build\s*#?\d+\)', re.IGNORECASE),  # 24.04 (Build #775)
        ]
        
        # check against HPE patterns
        for pattern in hpe_patterns:
            if pattern.search(text):
                return True
        
        # check against general version patterns
        for pattern in general_patterns:
            if pattern.search(text):
                return True
        
        # additional patterns for build identifiers and custom versions
        build_patterns = [
            re.compile(r'build\s*#?\s*\d+', re.IGNORECASE),
            re.compile(r'\d+\.\d+\.\d+\.\d+(?:\.\d+)?', re.IGNORECASE),  # Multi-part versions like 96.00.87.00.02
            re.compile(r'custom\s+\d+(?:\.\d+)+', re.IGNORECASE),  # Custom 2025.07.00.00
            re.compile(r'[a-z]+_[a-z]+\d+\s+version\s+[\d.]+', re.IGNORECASE),  # Kickstart_June2025 version 0.6
            re.compile(r'PCAI\d+(?:\.\d+)+', re.IGNORECASE),  # PCAI202508.5
            re.compile(r'[A-Z]{2}\.\d+(?:\.\d+)+', re.IGNORECASE),  # FL.1.0.13.1130, GL.1.0.13.1060, FS.3.3.0
            re.compile(r'\d{2,4}\.\d{2,4}(?:\.\d{2,4}){2,}', re.IGNORECASE),  # Complex numeric versions
        ]
        
        for pattern in build_patterns:
            if pattern.search(text):
                return True
        
        return False
    
    def _classify_version_type(self, text: str) -> str:
        """
        Classify the type of version information.
        Args: text string containing version/build information
        Returns: String indicating version type
        """
        if not text:
            return "unknown"
        
        text_lower = text.lower()
        
        # build identifiers
        if re.search(r'build\s*#?\s*\d+', text_lower) or '#' in text:
            return "build"
        
        # custom versions
        if text_lower.startswith('custom') or re.search(r'custom\s+\d+', text_lower):
            return "custom"
        
        # HPE PCAI release codes
        if re.search(r'PCAI\d+(?:\.\d+)*', text, re.IGNORECASE):
            return "release"
        
        # HPE PCAI with parenthetical info 
        if re.search(r'pcai\s+\d+\.\d+(?:\.\d+)*\s*\([^)]+\)', text_lower):
            return "main_version"
        
        # date-based or named versions 
        if re.search(r'[a-z]+_[a-z]+\d+\s+version', text_lower):
            return "release"
        
        # firmware versions
        if re.search(r'^[A-Z]{2}\.\d+(?:\.\d+)+', text):
            return "firmware"
        
        # complex multi-part versions
        version_parts = re.findall(r'\d+', text)
        if len(version_parts) >= 5:
            return "build"
        elif len(version_parts) == 4:
            return "build"
        
        # standard version numbers
        if re.search(r'^\d+\.\d+(\.\d+)*$', text):
            return "main_version"
        
        # version with parenthetical build info 
        if re.search(r'\d+\.\d+\s*\(build\s*#?\d+\)', text_lower):
            return "main_version"
        
        return "other"
    



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
            description="LLM Model Endpoint (Ollama)")
        LLM_MODEL_NAME: str = Field(
            default="meta/llama-3.1-8b-instruct",
            description="LLM Model Name")
        LLM_MODEL_API_KEY: str = Field(
            default="YOUR_LLM_MODEL_API_KEY",
            description="API Key for accessing the LLM model (not needed for local Ollama)")        
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
        
        # override defaults with environment variables if present
        if os.getenv("WEAVIATE_URL"):
            self.valves.WEAVIATE_URL = os.getenv("WEAVIATE_URL")
        if os.getenv("LLM_MODEL_URL"):
            self.valves.LLM_MODEL_URL = os.getenv("LLM_MODEL_URL")
        if os.getenv("LLM_MODEL_NAME"):
            self.valves.LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
        if os.getenv("LLM_MODEL_API_KEY"):
            self.valves.LLM_MODEL_API_KEY = os.getenv("LLM_MODEL_API_KEY")
        if os.getenv("EMBEDDING_MODEL_API_KEY"):
            self.valves.EMBEDDING_MODEL_API_KEY = os.getenv("EMBEDDING_MODEL_API_KEY")

        # weaviate client will be set by ensure_weaviate_client()
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

        # version identification patterns and constants
        self._HPE_PATTERNS = [
            re.compile(r'(?:version\s+|v\.?\s*|release\s+)?(\d+\.\d+(?:\.\d+)*(?:\.x)?)', re.IGNORECASE),
            re.compile(r'(?:hpe\s+)?(?:ai\s+essentials\s+software\s+)?(?:private\s+cloud\s+ai\s+)?(?:v\.?\s*)?(\d+\.\d+(?:\.\d+)*)', re.IGNORECASE),
            re.compile(r'essentials\s+software\s+(?:version\s+)?(\d+\.\d+(?:\.\d+)*)', re.IGNORECASE),
            re.compile(r'pcai\s+(?:v\.?\s*)?(\d+\.\d+(?:\.\d+)*)', re.IGNORECASE),
        ]
        
        self._GENERAL_PATTERNS = [
            re.compile(r'(?:version\s+|v\.?\s*)(\d+\.\d+(?:\.\d+)*(?:\.x)?)', re.IGNORECASE),
            re.compile(r'(\d+\.\d+(?:\.\d+)*(?:\.x)?)(?:\s*lts)?', re.IGNORECASE),
        ]
        
        self._HPE_DOC_TYPES = [
            'installation guide', 'user guide', 'admin guide', 'release notes',
            'configuration guide', 'troubleshooting guide', 'api reference',
            'quickstart', 'getting started'
        ]
        
        self._GENERAL_DOC_KEYWORDS = [
            'manual', 'documentation', 'guide', 'handbook', 'reference',
            'tutorial', 'instructions', 'setup', 'configuration'
        ]
        
        self._PRODUCT_PATTERNS = [
            ('hpe_ai_essentials', [
                re.compile(r'hpe\s+ai\s+essentials', re.IGNORECASE),
                re.compile(r'ai\s+essentials\s+software', re.IGNORECASE),
                re.compile(r'essentials\s+software', re.IGNORECASE),
            ]),
            ('hpe_private_cloud_ai', [
                re.compile(r'hpe\s+private\s+cloud\s+ai', re.IGNORECASE),
                re.compile(r'private\s+cloud\s+ai', re.IGNORECASE),
                re.compile(r'\bpcai\b', re.IGNORECASE),
            ])
        ]
        
        self._DATE_FORMATS = [
            '%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y', '%m/%d/%Y',
            '%Y-%m-%d %H:%M:%S', '%d.%m.%Y %H:%M:%S',
            '%B %d, %Y', '%b %d, %Y', '%d %B %Y', '%d %b %Y'
        ]

    # --------------------------
    # Weaviate connectivity
    # --------------------------
    def ensure_weaviate_client(self, max_retries: int = 8, delay: int = 5) -> None:
        """
        Ensure self.weaviate_client is set and collection existence is known. This is idempotent and will retry if Weaviate isn't ready yet.
        Args: number of retries, delay between retries in seconds
        Returns: None
        """
        if not weaviate:
            logger.error("weaviate library not imported in environment.")
            self.weaviate_client = None
            return
        if self.weaviate_client:
            logger.info("Weaviate is already connected.")
            return

        # Parse Weaviate URL
        url = self.valves.WEAVIATE_URL
        parsed = urlparse(url)
        host = parsed.hostname or "weaviate"
        port = parsed.port or 8080
        scheme = parsed.scheme or "http"

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Connecting to Weaviate at {host}:{port}")
                client = weaviate.connect_to_custom(
                    http_host=host,
                    http_port=port,
                    http_secure=False if scheme == 'http' else True,
                    grpc_host=host,
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
    # Chunking
    # --------------------------
    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string for chunking purposes.
        Uses tiktoken tokenizer if available, otherwise falls back to word count approximation.
        Args: input text string to count tokens for
        Returns: number of tokens (int). Returns 0 for empty/None text, minimum 1 for non-empty text.
        """
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
        """
        Token-based text splitting with overlap between chunks.
        Args: text to be split, maximum token count per chunk, token overlap between consecutive chunks
        Returns: list of text chunks that comply with token limits
        """
        # validate input
        text = (text or "").strip()
        if not text:
            return []
        # tokenize text
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
        # create chunks with overlap
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
        """
        Validate input text and split into chunks if necessary. Acts as a safety wrapper around the smart chunking functionality to ensure no chunk 
        exceeds the maximum allowed token count for API calls.
        Args: text to validate and split, optional maximum token count per chunk
        Returns: list of validated text chunks within token limits
        """
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
        """ 
        Create hierarchical parent-child chunks from input text based on token limits.  
        Args: text to chunk, document title, document URL, unique document ID
        Returns: list of chunk dictionaries with metadata
        """
        chunks_out = []
        # create parent chunks
        parent_texts = self._smart_chunk_by_tokens(text, PARENT_CHUNK_TOKENS)
        parent_idx = 0
        for ptext in parent_texts:
            # validate and possibly split parent chunk further
            parents = self._validate_and_split(ptext)
            for parent in parents:
                parent_token_count = self._count_tokens(parent)
                parent_chunk_id = f"{document_id}_parent_{parent_idx}"
                # create parent chunk object
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
                # create child chunks if parent is too large
                if parent_token_count > CHILD_CHUNK_TOKENS * 2:
                    child_texts = self._smart_chunk_by_tokens(parent, CHILD_CHUNK_TOKENS)
                    child_idx = 0
                    for ctext in child_texts:
                        validated_childs = self._validate_and_split(ctext)
                        for vct in validated_childs:
                            child_token_count = self._count_tokens(vct)
                            # safety check
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
    # Table processing  
    # --------------------------
    def convert_table_to_markdown(self, table: Dict[str, Any]) -> str:
        """
        Convert table data to markdown format for HTML processing.
        Args: Table dictionary with headers and rows
        Returns: String containing markdown representation of the table
        """
        headers = table.get('headers', [])
        rows = table.get('rows', [])
        if not headers and rows: 
            num_cols = max(len(row) for row in rows) if rows else 0
            headers = [f"Column {i+1}" for i in range(num_cols)]
        if not headers: 
            return ""
        cleaned_headers = [str(h).replace('\n', ' ').replace('|', '').strip() for h in headers]
        md = "| " + " | ".join(cleaned_headers) + " |\n" + "| " + " | ".join("---" for _ in cleaned_headers) + " |\n"
        for row in rows:
            cleaned_row = [str(cell).replace('\n', ' ').replace('|', '').strip() for cell in row]
            md += "| " + " | ".join((cleaned_row + [''] * len(cleaned_headers))[:len(cleaned_headers)]) + " |\n"
        return md.strip()

    # --------------------------
    # Versions handling
    # --------------------------
    def _determine_search_criteria(self, user_message: str) -> tuple:
        """
        Determine target version and product line from user message, with fallbacks to latest versions.
        Args: user message string
        Returns: tuple of target version (str or None) and product line (str or None)
        """
        # extract version from query
        query_version = self._extract_version_from_query(user_message)
        product_line = self._detect_hpe_product_line(user_message)
        document_type = self._detect_document_type_from_query(user_message)

        # fallbacks to latest versions
        if not query_version and product_line:
            query_version = self._get_latest_version_for_product(product_line, document_type)
            if query_version:
                logger.info(f"Using latest version {query_version} for product line {product_line}")
        elif not query_version:
            query_version = self._get_latest_version_for_document(document_type)
            if query_version:
                logger.info(f"Using latest version {query_version} as general fallback")
        
        return query_version, product_line

    def _search_chunks_with_fallback(self, user_message: str, query_version: Optional[str]) -> tuple:
        """
        Search for chunks with version fallbacks.
        Args: user message string, target version (str or None)
        Returns: tuple of found chunks (list) and version details (dict)
        """
        logger.info(f"Searching for chunks with target version: {query_version}")
        chunks = self.search_chunks(user_message, target_version=query_version)
        version_details = {"related_version_used": None, "fallback_used": False}
        # Fallback 0: related versions
        if not chunks and query_version:
            # Fallback 1: version 
            related_versions = self._find_related_versions(query_version)
            for related_version in related_versions:
                if related_version != query_version:
                    logger.info(f"Trying related version: {related_version}")
                    chunks = self.search_chunks(user_message, target_version=related_version)
                    if chunks:
                        version_details["related_version_used"] = related_version
                        break
        
        if not chunks and query_version:
            # Fallback 2: no version
            logger.info(f"No results for {query_version}, trying without version filter")
            chunks = self.search_chunks(user_message, target_version=None)
            if chunks:
                version_details["fallback_used"] = True
        
        return chunks, version_details

    def _generate_no_results_response(self, query_version: Optional[str], product_line: Optional[str]) -> str:
        """
        Generate a user-friendly message when no relevant documentation is found.
        Args: target version (str or None), product line (str or None)
        Returns: formatted message string
        """
        version_msg = f" for version {query_version}" if query_version else ""
        product_msg = f" in {product_line}" if product_line else ""
        help_msg = ""
        # list available products and versions
        products_and_versions = self.get_available_products_and_versions()
        if products_and_versions:
            help_msg = "\n\nAvailable documentation:"
            # list products and up to 3 versions each
            for product, versions in products_and_versions.items():
                v_list = ', '.join(versions[:3])
                more_count = len(versions) - 3
                if more_count > 0: v_list += f" and {more_count} more versions"
                help_msg += f"\n• {product}: {v_list}"

            # suggest similar versions if query version was specified
            if query_version:
                all_versions = [v for versions in products_and_versions.values() for v in versions]
                similar = [v for v in all_versions if query_version.split('.')[0] in v][:3]
                if similar:
                    help_msg += f"\n\nSimilar versions found: {', '.join(similar)}"
        
        return (f"I couldn't find any relevant information{version_msg}{product_msg} "
                f"in the documentation database. Please try rephrasing your question.{help_msg}")

    def _build_enhanced_prompt(self, context: str, user_message: str, query_version: Optional[str], version_details: Dict) -> str:
        """
        Build an enhanced prompt for the language model with version context.
        Args: documentation context string, user message string, target version (str or None), version details (dict)
        Returns: formatted prompt string
        """
        version_instruction = ""
        fallback_notice = ""
        
        related_version = version_details.get("related_version_used")
        fallback_used = version_details.get("fallback_used")

        if query_version:
            if related_version:
                version_instruction = f"- The user asked for {query_version}, but you are providing info from {related_version}\n"
                fallback_notice = (f"\nIMPORTANT: You found info for {related_version}, not {query_version}. "
                                   f"Clearly state this difference in your response.\n")
            elif fallback_used:
                version_instruction = f"- User asked for {query_version}, but no specific doc was found\n"
                fallback_notice = (f"\nIMPORTANT: No specific doc for {query_version} was found. "
                                   f"You are using other versions. Mention which versions you are using.\n")
            else:
                version_instruction = f"- Pay special attention to version-specific info for {query_version}\n"
        
        return f"""You are a helpful technical documentation assistant specializing in version-specific documentation. Answer the user's question based on the provided documentation context.

DOCUMENTATION CONTEXT:
{context}{fallback_notice}

INSTRUCTIONS:
- Provide detailed, step-by-step answers when applicable
- Use specific information from the documentation
{version_instruction}- If multiple versions are referenced, clearly distinguish between them
- If the documentation doesn't fully answer the question, acknowledge this
- Be comprehensive and include all relevant details
- When referencing information, mention the document version if available
- Do not mention "Source X" - integrate information naturally

USER QUESTION:
{user_message}

DETAILED ANSWER:"""

    def _extract_version_from_text(self, text: str, log_detection: bool = False) -> Optional[str]:
        """
        Extract version string from text, with optional logging for query detection.
        Args: text string to search, flag to enable logging
        Returns: extracted version string or None
        """
        if not text: 
            return None
            
        text_lower = text.lower()
        
        # check HPE-specific patterns first
        for pattern in self._HPE_PATTERNS:
            match = pattern.search(text_lower)
            if match:
                version = match.group(1)
                if log_detection:
                    logger.info(f"HPE-specific version detected in query: {version}")
                return version
        
        # check general patterns
        for pattern in self._GENERAL_PATTERNS:
            match = pattern.search(text_lower)
            if match:
                version = match.group(1)
                if log_detection:
                    logger.info(f"General version detected in query: {version}")
                return version
        
        if log_detection:
            logger.info("No specific version detected in query")
        return None

    def _extract_version_from_query(self, query: str) -> Optional[str]:
        """
        Extract version from user query with logging.
        Args: user query string
        Returns: extracted version string or None
        """
        return self._extract_version_from_text(query, log_detection=True)

    def _version_to_tuple(self, version: str) -> tuple:
        """
        Convert version string to a tuple for comparison.
        Args: version string
        Returns: tuple of integers representing version
        """
        try:
            parts = version.split('.')
            result = []
            for part in parts:
                if part.lower() == 'x': result.append(999)
                else: result.append(int(part))
            while len(result) < 3: result.append(0)
            return tuple(result)
        except:
            return (0, 0, 0)

    def _compare_versions(self, version1: str, version2: str) -> int:
        """
        Compare two version strings.
        Args: two version strings
        Returns: 1 if version1 > version2, -1 if version1 < version
        """
        v1_tuple = self._version_to_tuple(version1)
        v2_tuple = self._version_to_tuple(version2)
        if v1_tuple > v2_tuple: return 1
        if v1_tuple < v2_tuple: return -1
        return 0

    def _is_newer_date(self, date1: str, date2: str) -> bool:
        """
        Compare two date strings to determine if date1 is newer than date2.
        Args: two date strings
        Returns: True if date1 > date2, False otherwise
        """
        if not date1 or not date2: return bool(date1)
        parsed_date1, parsed_date2 = None, None
        for fmt in self._DATE_FORMATS:
            try:
                if not parsed_date1: parsed_date1 = datetime.strptime(date1.strip(), fmt)
            except: continue
            try:
                if not parsed_date2: parsed_date2 = datetime.strptime(date2.strip(), fmt)
            except: continue
            if parsed_date1 and parsed_date2: break
        
        if parsed_date1 and parsed_date2: return parsed_date1 > parsed_date2
        return date1 > date2  # Fallback

    def _detect_document_type_from_query(self, query: str) -> Optional[str]:
        """
        Detect document type from user query.
        Args: user query string
        Returns: detected document type string or None
        """
        query_lower = query.lower()
        for doc_type in self._HPE_DOC_TYPES:
            if doc_type in query_lower:
                logger.info(f"HPE document type detected: {doc_type}")
                return doc_type
        for keyword in self._GENERAL_DOC_KEYWORDS:
            if keyword in query_lower:
                logger.info(f"General document type detected: {keyword}")
                return keyword
        return None

    def _detect_hpe_product_line(self, query: str) -> Optional[str]:
        """
        Detect HPE product line from user query.
        Args: user query string
        Returns: detected product line key or None
        """
        query_lower = query.lower()
        for product_key, patterns in self._PRODUCT_PATTERNS:
            for pattern in patterns:
                if pattern.search(query_lower):
                    logger.info(f"HPE product line detected: {product_key}")
                    return product_key
        return None

    def _fetch_all_documents(self, product_line: Optional[str] = None) -> List[Dict]:
        """
        Fetch all documents from Weaviate collection.
        Args: product line (str or None)
        Returns: list of document metadata dictionaries
        """
        if not self.weaviate_client or not self.collection_exists:
            return []
        # Fetch documents
        try:
            collection = self.weaviate_client.collections.get(self.valves.COLLECTION_NAME)
            response = collection.query.fetch_objects(
                limit=1000,
                return_properties=["document_id", "document_title", "document_date"]
            )
            
            documents = []
            for obj in response.objects:
                # extract properties
                props = obj.properties
                doc_text = (props.get("document_id", "") + " " + props.get("document_title", "")).lower()
                
                # filter by product line if specified
                if product_line:
                    if product_line == 'hpe_ai_essentials' and not any(k in doc_text for k in ['ai essentials', 'essentials software']):
                        continue
                    if product_line == 'hpe_private_cloud_ai' and not any(k in doc_text for k in ['private cloud ai', 'pcai']):
                        continue
                
                documents.append({
                    "doc_id": props.get("document_id", ""),
                    "doc_title": props.get("document_title", ""),
                    "doc_date": props.get("document_date", ""),
                    "doc_text": doc_text
                })
            return documents
        except Exception as e:
            logger.error(f"Error fetching documents from Weaviate: {e}")
            return []

    def _get_latest_version_for_product(self, product_line: str = None, document_pattern: str = None) -> Optional[str]:
        """
        Get the latest version for a specific product line and optional document pattern.
        Args: product line (str or None), document pattern (str or None)
        Returns: latest version string or None
        """
        
        all_docs = self._fetch_all_documents(product_line=product_line)
        documents = {}
        document_dates = {}

        for doc in all_docs:
            # extract version
            version = self._extract_version_from_text(doc["doc_text"])
            if version:
                # filter by document pattern if specified
                if not document_pattern or document_pattern.lower() in doc["doc_text"]:
                    key = doc["doc_id"] or doc["doc_title"]
                    if key:
                        if key not in documents:
                            documents[key] = version
                            document_dates[key] = doc["doc_date"]
                        else:
                            # compare versions
                            version_cmp = self._compare_versions(version, documents[key])
                            if version_cmp > 0 or (version_cmp == 0 and self._is_newer_date(doc["doc_date"], document_dates[key])):
                                documents[key] = version
                                document_dates[key] = doc["doc_date"]
        
        if documents:
            # determine latest version
            latest_version = max(documents.values(), key=lambda v: self._version_to_tuple(v))
            logger.info(f"Latest version found for product {product_line or 'all'}: {latest_version}")
            return latest_version
        return None

    def _get_latest_version_for_document(self, document_pattern: str = None) -> Optional[str]:
        """
        Get the latest version for documents matching a specific pattern.
        Args: document pattern (str or None)
        Returns: latest version string or None
        """
        return self._get_latest_version_for_product(product_line=None, document_pattern=document_pattern)

    def get_available_versions(self, document_pattern: str = None) -> List[str]:
        """
        Get all available versions for documents matching a specific pattern.
        Args: document pattern (str or None)
        Returns: list of available version strings
        """
        all_docs = self._fetch_all_documents()
        versions = set()
        for doc in all_docs:
            # extract version
            version = self._extract_version_from_text(doc["doc_text"])
            if version:
                # filter by document pattern if specified
                if not document_pattern or document_pattern.lower() in doc["doc_text"]:
                    versions.add(version)
        # sort versions
        sorted_versions = sorted(list(versions), key=self._version_to_tuple, reverse=True)
        logger.info(f"Available versions: {sorted_versions}")
        return sorted_versions

    def get_available_products_and_versions(self) -> Dict[str, List[str]]:
        """
        Get available products and their versions.
        Returns: dictionary mapping product names to lists of version strings
        """
        all_docs = self._fetch_all_documents()
        products = {
            'HPE AI Essentials Software': set(),
            'HPE Private Cloud AI': set(),
            'Other/Unknown': set()
        }
        for doc in all_docs:
            # extract version
            version = self._extract_version_from_text(doc["doc_text"])
            if version:
                if 'ai essentials' in doc["doc_text"] or 'essentials software' in doc["doc_text"]:
                    products['HPE AI Essentials Software'].add(version)
                elif 'private cloud ai' in doc["doc_text"] or 'pcai' in doc["doc_text"]:
                    products['HPE Private Cloud AI'].add(version)
                else:
                    products['Other/Unknown'].add(version)
        
        result = {}
        for product, versions in products.items():
            if versions:
                # sort versions
                sorted_versions = sorted(list(versions), key=self._version_to_tuple, reverse=True)
                result[product] = sorted_versions
                logger.info(f"{product}: {sorted_versions}")
        return result

    def _find_related_versions(self, target_version: str) -> List[str]:
        """
        Find related versions based on the target version.
        Args: target version string
        Returns: list of related version strings
        """
        if not target_version: return []
        # get all available versions
        available_versions = self.get_available_versions()
        related_versions = []
        # build matching criteria
        target_parts = target_version.replace('.x', '').split('.')
        
        for version in available_versions:
            # check if version matches target version pattern
            version_clean = version.replace('.x', '')
            version_parts = version_clean.split('.')
            
            if len(version_parts) >= len(target_parts):
                # compare versions
                match = all(version_parts[i] == target_part for i, target_part in enumerate(target_parts))
                if match:
                    related_versions.append(version)
        
        related_versions.sort(key=self._version_to_tuple, reverse=True)
        if related_versions:
            logger.info(f"Related versions found for {target_version}: {related_versions}")
        return related_versions

    # --------------------------
    # Scraper 
    # --------------------------
    def _extract_row_metadata(self, row) -> Dict[str, str]:
        """
        Extract metadata such as date from a table row.
        Args: BeautifulSoup row element
        Returns: dictionary with metadata fields
        """
        metadata = {'date': ''}
        cells = row.find_all('td', role='gridcell')
        for cell in cells:
            cell_text = cell.get_text(strip=True)
            if cell_text:
                if cell.get('data-label') == 'Date' or 'date' in cell.get('data-col-key-value', '').lower(): 
                    metadata['date'] = cell_text
                    break
                if re.fullmatch(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}', cell_text, re.I): 
                    metadata['date'] = cell_text
                    break
                if re.fullmatch(r'\d{4}-\d{2}-\d{2}', cell_text): 
                    metadata['date'] = cell_text
                    break
        return metadata

    async def _scrape_hpe_manuals_improved(self, page):
        """
        Scrape HPE manuals from the support page with improved pagination handling.
        Args: Playwright page object
        Returns: list of manual metadata dictionaries
        """

        start_url = self.valves.HPE_SUPPORT_PAGE
        await page.goto(start_url, wait_until='networkidle', timeout=90000)
        all_manuals_data = []
        current_page = 1
        consecutive_empty_pages = 0
        
        while True:
            logger.info(f"Processing page {current_page}")
            try:
                await page.wait_for_selector('table.slds-table a[href*="docId"]', timeout=30000)
                await page.wait_for_timeout(1000)
            except Exception as e: 
                logger.warning(f"Table not found/empty on page {current_page}. Stopping. Error: {e}")
                break

            html_content = await page.content()
            soup = BeautifulSoup(html_content, 'html.parser')
            table = soup.find('table', class_='slds-table')
            manuals_on_page = []

            if table:
                rows = table.find_all('tr', role='row')
                if not rows or len(rows) <= 1: 
                    logger.info(f"No data rows on page {current_page}.")
                    consecutive_empty_pages += 1
                else:
                    found_new_on_page = False
                    for row in rows[1:]:  # Skip header
                        # find link
                        link_element = row.find('a', href=re.compile(r'docDisplay.*docId='))
                        if link_element and (href := link_element.get('href')) and (title := link_element.get_text(strip=True)):
                            absolute_href = urljoin(self.valves.HPE_BASE_URL, href)
                            if absolute_href not in {m['link'] for m in all_manuals_data + manuals_on_page}:
                                metadata = self._extract_row_metadata(row)
                                manuals_on_page.append({'title': title, 'link': absolute_href, 'date': metadata.get('date', '')})
                                print(f" Found: {title} ({metadata.get('date', 'No date')})")
                                found_new_on_page = True
                    if not found_new_on_page: 
                        logger.info(f"No new documents on page {current_page}.")
                        consecutive_empty_pages += 1
                    else: 
                        consecutive_empty_pages = 0
                        all_manuals_data.extend(manuals_on_page)
            else: 
                logger.warning(f"Table not found on page {current_page}.")
                consecutive_empty_pages += 1

            # check for consecutive empty pages
            if consecutive_empty_pages >= 2: 
                logger.info("2 consecutive empty pages. Stopping.")
                break

            next_page_number = current_page + 1
            try:
                # click next page button
                page_button = await page.query_selector(f'a.pagination-item[data-number="{next_page_number}"]')
                if page_button:
                    print(f"Going to page {next_page_number}")
                    await page_button.click()
                    await page.wait_for_load_state('networkidle', timeout=45000)
                    current_page += 1
                else:
                    print(f"Page {next_page_number} link not found. End pagination.")
                    break
            except Exception as e:
                print(f"Error clicking page {next_page_number} or page load failed: {e}")
                break
               
            await asyncio.sleep(1)
            
        unique_manuals = list({m['link']: m for m in all_manuals_data}.values())
        logger.info(f"Scraping finished. Found {len(unique_manuals)} unique manuals.")
        return unique_manuals

    async def _toc_scraper_improved(self, url, browser):
        """
        Scrape the table of contents (TOC) from a HPE manual page.
        Args: url (str), browser (Playwright browser object)
        Returns: dictionary with TOC structure
        """
        page = await browser.new_page()
        content = None
        try:
            logger.info(f"Loading TOC page: {url}")
            await page.goto(url, wait_until='networkidle', timeout=60000)
            await page.wait_for_selector('#contentListArea', timeout=15000)
            await page.wait_for_timeout(1000)
            content = await page.content()
        except Exception as e:
            logger.error(f"Error loading or finding TOC container for {url}: {e}")
            return {"url": url, "toc": []}
        finally:
            if page and not page.is_closed(): 
                await page.close()
        if not content: 
            return {"url": url, "toc": []}

        soup = BeautifulSoup(content, 'html.parser')
        # Find the TOC container
        toc_container = soup.find(id="contentListArea")
        if not toc_container:
            logger.warning(f"TOC container #contentListArea not found in HTML for {url}")
            return {"url": url, "toc": []}

        processed_ids = set()

        def extract_node_info(node, base_url):
            """
            Extract metadata from a TOC node.
            Args: node (BeautifulSoup element), base_url (str)
            Returns: dictionary with metadata fields
            """
            node_id = str(node)
            if node_id in processed_ids: 
                return None
            processed_ids.add(node_id)
            title = ''
            href = None
            # try to find link first
            link = node.find('a', href=True, recursive=False)
            if not link: 
                link = node.find('a', href=True)

            if link:
                # extract title and href
                title = link.get_text(strip=True)
                href = link.get('href')
            else:
                # fallback to span or div
                span = node.find('span', {'data-for': True})
                if span: 
                    title = span.get_text(strip=True)
                    href = span.get('href')
                else:
                     div = node.find('div')
                     if div: 
                         title = div.get_text(strip=True)
                         
            if href and not href.startswith(('http', '#', 'javascript:')):
                href = urljoin(base_url, href)

            return {'title': title, 'href': href} if title else None

        def build_hierarchy(start_node, level=0, base_url=url):
            """
            Recursively build TOC hierarchy from a starting node.
            Args: start_node (BeautifulSoup element), level (int), base_url (str)
            Returns: dictionary representing TOC node with children
            """

            node_info = extract_node_info(start_node, base_url)
            if not node_info: 
                return None
            result = {'title': node_info['title'], 'url': node_info['href'], 'level': level, 'children': []}
            # process child nodes in class 'content-menu-nested'
            child_ul = start_node.find('ul', class_='content-menu-nested', recursive=False)
            if child_ul:
                child_items = child_ul.find_all('li', class_='content-menu-topicref', recursive=False)
                if not child_items:
                    child_items = child_ul.find_all('li', recursive=False)
                for child in child_items:
                    if child_result := build_hierarchy(child, level + 1, base_url):
                        result['children'].append(child_result)
            return result
        
        
        # Start building hierarchy from top-level chapters
        chapters = toc_container.find_all('li', class_='content-menu-topicref', recursive=False)
        if not chapters:
            first_ul = toc_container.find('ul', recursive=False)
            if first_ul: 
                chapters = first_ul.find_all('li', class_='content-menu-topicref', recursive=False)
        if not chapters:
             chapters = toc_container.find_all('li', class_='content-menu-topicref')

        results = [ch for chapter in chapters if (ch := build_hierarchy(chapter, 0, url)) and ch.get('title')]
        
        # Remove duplicates 
        final_results = []
        seen_keys = set()
        for res in results:
            key = (res['title'], res['url'])
            if key not in seen_keys:
                final_results.append(res)
                seen_keys.add(key)

        logger.info(f"Found {len(final_results)} top-level TOC entries for {url}")
        return {"url": url, "toc": final_results}

    async def _scrape_document_contents_improved(self, toc_data, browser):
        """
        Scrape document contents based on TOC structure.
        Args: toc_data (dict), browser (Playwright browser object)
        Returns: list of content section dictionaries
        """
        contents = []
        base_url = toc_data.get("url", "")

        async def scrape_node(node):
            # scrape content for a TOC node
            node_url = node.get('url')
            if node_url and not node_url.startswith(('http:', 'https:', '#', 'javascript:')):
                node_url = urljoin(base_url, node_url)
            # only process valid http URLs
            if node_url and node_url.startswith('http'):
                clean_node_url = urlparse(node_url)._replace(query='', fragment='').geturl()
                clean_base_url = urlparse(base_url)._replace(query='', fragment='').geturl()
                # avoid re-scraping base URL
                if clean_node_url == clean_base_url and any(d['url'] == node_url for d in contents):
                     print(f"  Skipping TOC link (already scraped): {node.get('title', 'No Title')}")
                else:
                    print(f"  Scraping TOC link: {node.get('title', 'No Title')} -> {node_url}")
                    main_html = await self._scrape_manual_main_block_improved(node_url, browser)
                    # store content if found
                    if main_html:
                        contents.append({'title': node.get('title', ''), 'url': node_url, 'content': main_html})
                    await asyncio.sleep(0.5)

            for child in node.get('children', []):
                await scrape_node(child)
        # start scraping from top-level nodes
        if isinstance(toc_data.get('toc'), list):
            for top_node in toc_data['toc']:
                await scrape_node(top_node)

        logger.info(f"Scraped {len(contents)} sections from TOC for {base_url}")
        return contents

    async def _scrape_manual_main_block_improved(self, url, browser):
        """
        Scrape the main content block from a manual page.
        Args: url (str), browser (Playwright browser object)
        Returns: HTML string of main content block or None
        """
        page = await browser.new_page()
        try:
            print(f"    Navigating to content page: {url}")
            await page.goto(url, wait_until='networkidle', timeout=60000)
            # Try original selectors first
            main_selector = 'main.ditasrc, main[role="main"]'
            try:
                await page.wait_for_selector(main_selector, timeout=15000)
            except Exception as e:
                 print(f"    Warning: Original selectors ('{main_selector}') not found for {url}. Error: {e}")
                 try:
                     await page.wait_for_selector('main', timeout=5000)
                     main_selector = 'main'
                 except:
                     print(f"    No main content block found at all for {url}.")
                     return None

            await page.wait_for_timeout(500)
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            # Try to find the main content block
            main_block = soup.select_one(main_selector)

            if not main_block:
                print(f"    No main block found (despite waiting) for {url}")
                return None
            else:
                 print(f"    Found main content block: <{main_block.name} ...>")
                 return str(main_block)
        except Exception as e: 
            print(f"    Error scraping main block {url}: {e}")
            return None
        finally: 
            await page.close()

    def _clean_filename_improved(self, title):
        """
        Clean and sanitize a title string to create a safe filename.
        Args: title string
        Returns: cleaned filename string
        """
        if not title: 
            title = "untitled"
        clean = re.sub(r'[^\w\s\-\.]', '', str(title))
        clean = re.sub(r'[-\s]+', '_', clean).strip('_')
        base, ext = os.path.splitext(clean)
        clean = base[:100].strip('_') + ext
        return clean if clean else "document"

    def _clean_html_entry_improved(self, entry, base_url):
        """
        Clean HTML content from a scraped entry.
        Args: entry dictionary, base_url string
        Returns: dictionary with cleaned text and metadata
        """

        html_content = entry.get('content', '')
        entry_url = entry.get('url') or base_url
        if not html_content: 
            return {"title": entry.get("title"), "url": entry_url, "text_clean": ""}
        
        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove unwanted elements
        selectors_to_remove = ['nav', 'footer', 'header', 'script', 'style', 'aside', '.sidebar', '#sidebar', '.toc', '#toc',
                               '.breadcrumb', 'form', 'button', 'input', 'select', 'textarea', 'img', 'figure', 'figcaption',
                               '.metadata', '.related-links', '.noprint', '[aria-hidden="true"]']
        for selector in selectors_to_remove:
            for element in soup.select(selector): 
                element.decompose()
        
        # process tables, convert to markdown format for HTML processing
        table_texts = []
        for table in soup.find_all('table'):
            # extract headers and rows in a structured way
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            rows_data = []
            
            for tr in table.find_all('tr'):
                cells = [td.get_text(strip=True) for td in tr.find_all('td')]
                if cells:
                    rows_data.append(cells)
            
            if rows_data:
                # create table dict structure for markdown conversion
                table_dict = {
                    'headers': headers if headers else [f"Column {i+1}" for i in range(len(rows_data[0])) if rows_data],
                    'rows': rows_data
                }
                
                # convert to markdown format for HTML tables
                table_markdown = self.convert_table_to_markdown(table_dict)
                if table_markdown:
                    table_texts.append(table_markdown)
                    
            table.decompose()
        table_text = '\n\n'.join(table_texts)
        
        # process links
        for a in soup.find_all('a', href=True):
            href = a.get('href', '')
            text = a.get_text(strip=True) or href
            if href and not href.startswith(('#', 'javascript:')):
                absolute_href = urljoin(base_url, href)
                a.replace_with(f"{text} ({absolute_href})")
            else: 
                a.replace_with(text if text else '')
        
        text_clean = soup.get_text(separator='\n', strip=True)
        text_clean = re.sub(r'\n\s*\n+', '\n\n', text_clean)
        text_clean = re.sub(r'[ \t]+', ' ', text_clean)
        text_clean = re.sub(r'^\s+', '', text_clean, flags=re.MULTILINE)
        if table_text: 
            text_clean += '\n\n--- Tables ---\n' + table_text
        return {"title": entry.get("title", ""), "url": entry_url, "text_clean": text_clean.strip()}
    
    # These are nested helpers used inside the async scrape_and_ingest function below.
    # We implement them as inner functions of scrape_and_ingest to keep self accessible via closure.
    async def scrape_and_ingest(self, max_manuals: int = 15, reporter: Optional[callable] = None):
        """
        Scrape HPE manuals and ingest them into Weaviate.
        Args: max_manuals (int): maximum number of manuals to process
              reporter (callable): optional function for reporting progress
        Returns: None
        """
        logger.info(f"=== SCRAPE_AND_INGEST STARTED === max_manuals={max_manuals}")
        logger.info(f"Starting scrape and ingest process with max_manuals={max_manuals}")
        
        # check for missing dependencies
        missing_deps = []
        if async_playwright is None:
            missing_deps.append("playwright")
        if BeautifulSoup is None:
            missing_deps.append("beautifulsoup4")
            
        if missing_deps:
            error_msg = f"Playwright and BeautifulSoup are required in the container. Missing: {', '.join(missing_deps)}. Please rebuild the container."
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            if reporter:
                reporter(f"Scraper failed: {error_msg}\n")
            raise RuntimeError(error_msg)
        
        # warn about optional dependencies
        if DocumentConverter is None:
            warning_msg = "Warning: DocumentConverter (docling) is not available. PDF processing will be limited."
            logger.warning(warning_msg)
            if reporter:
                reporter(warning_msg + "\n")

        def _report(msg: str):
            """
            Report progress messages.
            """
            try:
                logger.info(msg)
                if reporter:
                    reporter(msg + "\n")
            except Exception:
                logger.debug("Reporter failed", exc_info=True)


        async def _check_if_pdf(url, browser):
            """
            Check if the given URL points to a PDF document.
            Args: url (str), browser (Playwright browser object)
            Returns: True if PDF, False otherwise
            """
            page = await browser.new_page()
            found_pdf = False
            
            # check URL patterns
            pdf_indicators = ['.pdf', 'docId=', 'compatibility matrix', 'matrix', 'firmware', 'quickspecs']
            if any(indicator in url.lower() for indicator in pdf_indicators):
                _report(f"PDF detected by URL pattern: {url}")
                found_pdf = True

            # check response headers if not already detected
            if not found_pdf:
                def on_response(response):
                    nonlocal found_pdf
                    try:
                        ct = response.headers.get("content-type", "").lower()
                        if "pdf" in ct or "application/pdf" in ct:
                            found_pdf = True
                            _report(f"PDF detected by content-type: {ct}")
                    except Exception:
                        pass

                page.on("response", on_response)
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    await page.wait_for_timeout(1000)
                except Exception as e:
                    _report(f"Error in PDF detection: {e}")
                        
            try:
                if page and not page.is_closed():
                    await page.close()
            except:
                pass
                
            return found_pdf

        async def _process_document(manual, browser):
            """
            Process a single manual document.
            Args: manual (dict), browser (Playwright browser object)
            Returns: dict with type and chunks
            """
            url = manual.get("link")
            title = manual.get("title", "untitled")
            date = manual.get("date", "")
            _report(f"Processing document: {title}")
            
            logger.info(f"Processing document: {title} (URL: {url}, Date: {date})")
            
            # check if document already exists in Weaviate
            self.ensure_weaviate_client()
            if self.weaviate_client and check_if_document_exists(self.weaviate_client, url, self.valves.COLLECTION_NAME):
                _report(f"Document already exists, skipping: {title}")
                logger.info(f"Skipping existing document: {title}")
                return {"type": "skipped", "title": title, "url": url}
                
                
            try:
                # force PDF detection for specific document types
                force_pdf_types = ['compatibility matrix', 'firmware', 'software compatibility', 'quickspecs']
                is_force_pdf = any(keyword in title.lower() for keyword in force_pdf_types)
                
                print(f"Force PDF keywords check: {force_pdf_types}")
                print(f"Title matches force PDF: {is_force_pdf}")
                print(f"Checking if URL points to PDF...")
                
                is_pdf = await _check_if_pdf(url, browser)
                print(f"PDF detection result: {is_pdf}")
                
                # override PDF detection for known PDF document types
                if not is_pdf and is_force_pdf:
                    is_pdf = True
                    _report(f"Forcing PDF processing for document type: {title}")
                    logger.info(f"Overriding PDF detection to True due to document type")
                
                logger.debug(f"PDF check result: {is_pdf} (forced: {is_force_pdf})")
                logger.info(f"Final PDF processing decision: {is_pdf}")
                
                if is_pdf:
                    logger.debug("Detected PDF; attempting download + Docling processing.")
                    # extract doc_id from URL for PDF download
                    doc_id = None
                    if 'docId=' in url:
                        try:
                            doc_id = url.split('docId=')[1].split('&')[0]
                            logger.debug(f"Extracted doc_id: {doc_id}")
                        except:
                            logger.debug("Failed to extract doc_id from URL")
                            pass
                    
                    if doc_id:
                        try:
                            logger.debug(f"Attempting to download PDF with doc_id: {doc_id}")
                            pdf_bytes, download_method = await download_pdf_direct_api(url, doc_id, title)
                            logger.debug(f"PDF download result: {len(pdf_bytes) if pdf_bytes else 0} bytes, method: {download_method}")
                            if pdf_bytes:
                                # save PDF
                                pdf_path_str = save_pdf_to_disk(pdf_bytes, title)
                                logger.debug(f"PDF saved to: {pdf_path_str}")
                                if pdf_path_str and DocumentConverter:
                                    _report(f"Processing PDF with Docling: {title}")
                                    try:
                                        # initialize PDF extractor and process
                                        print(f"\n=== PIPELINE PDF PROCESSING START ===")
                                        print(f"Starting PDF processing for: {title}")
                                        print(f"PDF file path: {pdf_path_str}")
                                        pdf_extractor = PDFContentExtractor(PDF_CLEAN_DIR)
                                        manual_metadata = {"title": title, "url": url, "date": date}
                                        
                                        print(f"Calling extract_content_from_pdf...")
                                        pdf_content = pdf_extractor.extract_content_from_pdf(
                                            Path(pdf_path_str), manual_metadata
                                        )
                                        print(f"PDF extraction completed. Found {len(pdf_content.get('texts', []))} text items and {len(pdf_content.get('tables', []))} tables")
                                        print(f"=== PIPELINE PDF PROCESSING START END ===\n")
                                        
                                        # convert PDF content to chunks
                                        chunks = []
                                        doc_id_clean = self._clean_filename_improved(title)
                                        
                                        # process text content
                                        for idx, text_item in enumerate(pdf_content.get('texts', [])):
                                            txt = text_item.get('content', '')
                                            if not txt or len(txt) < 40:
                                                continue
                                            section_id = f"{doc_id_clean}_pdf_text_{idx}"
                                            entry_chunks = self._create_hierarchical_chunks(
                                                txt, title, url, section_id
                                            )
                                            for c in entry_chunks:
                                                c.update({
                                                    "date": date, 
                                                    "document_title": title, 
                                                    "document_url": url,
                                                    "page_number": text_item.get('page_number', 1),
                                                    "content_type": "text"
                                                })
                                            chunks.extend(entry_chunks)
                                        
                                        # process table content
                                        print(f"\n=== PIPELINE TABLE PROCESSING START ===")
                                        tables_found = pdf_content.get('tables', [])
                                        print(f"Tables to process: {len(tables_found)}")
                                        if tables_found:
                                            print(f"Table IDs: {[t.get('table_id', 'unknown') for t in tables_found]}")
                                        else:
                                            print("No tables found in PDF content - table processing will be skipped")
                                        print(f"=== PIPELINE TABLE PROCESSING START END ===\n")
                                        
                                        for idx, table_item in enumerate(pdf_content.get('tables', [])):
                                            table_sentences = pdf_extractor.convert_table_to_sentences(table_item, title)
                                            
                                            # debug: Show table processing in pipeline
                                            print(f"Processing table {idx + 1}/{len(tables_found)}: {table_item.get('table_id', 'unknown')}")

                                            if not table_sentences or len(table_sentences) < 20:
                                                continue
                                            section_id = f"{doc_id_clean}_pdf_table_{idx}"
                                            entry_chunks = self._create_hierarchical_chunks(
                                                f"Table data: {table_sentences}", title, url, section_id
                                            )
                                            for c in entry_chunks:
                                                c.update({
                                                    "date": date, 
                                                    "document_title": title, 
                                                    "document_url": url,
                                                    "page_number": table_item.get('page_number', 1),
                                                    "content_type": "table"
                                                })
                                            chunks.extend(entry_chunks)
                                        
                                        _report(f"PDF processed successfully: {len(chunks)} chunks")
                                        return {"type": "pdf", "title": title, "url": url, "date": date, "chunks": chunks}
                                        
                                    except Exception as e:
                                        _report(f"Error processing PDF with Docling: {e}")
                                        logger.debug(traceback.format_exc())
                                        _report("PDF processing failed; falling back to HTML processing.")
                                else:
                                    _report("Docling not available for PDF processing; falling back to HTML processing.")
                            else:
                                _report("PDF download failed; falling back to HTML.")
                        except Exception as pdf_error:
                            _report(f"PDF download error: {pdf_error}; falling back to HTML.")
                    else:
                        _report("Could not extract doc_id from URL; falling back to HTML.")
                else:
                    _report("Document was not detected as PDF, proceeding with HTML processing.")
                
                # HTML path (fallback or for non-PDF documents)
                toc = await self._toc_scraper_improved(url, browser)
                if not toc.get("toc"):
                    _report("No TOC found; scraping main content.")
                    if main_html := await self._scrape_manual_main_block_improved(url, browser):
                        cleaned = self._clean_html_entry_improved({"title": title, "url": url, "content": main_html}, url)
                        entries = [cleaned]
                    else:
                        entries = []
                else:
                    scraped = await self._scrape_document_contents_improved(toc, browser)
                    entries = [self._clean_html_entry_improved(e, url) for e in scraped if e.get("content")]
                if not entries:
                    _report("No content scraped for document.")
                    return None
                # chunk and return
                chunks = []
                doc_id_clean = self._clean_filename_improved(title)
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

        
        
        # main scraping & ingest flow
        reporter and reporter("Scraper: starting Playwright browser...\n")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
            page = await browser.new_page()
            manuals = await self._scrape_hpe_manuals_improved(page)
            await page.close()

            if not manuals:
                reporter and reporter("Scraper: no manuals found.\n")
                await browser.close()
                return

                reporter and reporter(f"Scraper: found {len(manuals)} manuals; processing first {min(max_manuals, len(manuals))} documents.\n")
            all_chunks = []
            processed_docs = 0
            skipped_docs = 0
            # process first max_manuals documents
            selected_manuals = manuals[:max_manuals]
            for manual in selected_manuals:
                doc_result = await _process_document(manual, browser)
                if doc_result and doc_result.get("chunks"):
                    all_chunks.extend(doc_result["chunks"])
                    processed_docs += 1
                elif doc_result is None:  # document was skipped (already exists)
                    skipped_docs += 1
                await asyncio.sleep(0.5)

            await browser.close()
            
            if skipped_docs > 0:
                reporter and reporter(f"Scraper: skipped {skipped_docs} documents (already exist in database).\n")
            reporter and reporter(f"Scraper: processed {processed_docs} new documents.\n")

        if not all_chunks:
            reporter and reporter("Scraper: no chunks produced.\n")
            return

        reporter and reporter(f"Scraper: generated total {len(all_chunks)} chunks. Uploading to Weaviate...\n")

        # ensure weaviate client
        self.ensure_weaviate_client()
        if not self.weaviate_client:
            reporter and reporter("Scraper: could not connect to Weaviate for upload.\n")
            return

        # create collection if doesn't exist (v4 client path)
        try:
            col = self.weaviate_client.collections.get(self.valves.COLLECTION_NAME)
            reporter and reporter(f"Scraper: collection exists: {self.valves.COLLECTION_NAME}\n")
        except Exception:
            reporter and reporter(f"Scraper: creating collection {self.valves.COLLECTION_NAME} ...\n")
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
                reporter and reporter("Scraper: created collection.\n")
                self.collection_exists = True
            except Exception as e:
                reporter and reporter(f"Scraper: failed to create collection: {e}\n")
                logger.debug(traceback.format_exc())
                return

        # use individual insertions to avoid gRPC issues
        try:
            collection_obj = self.weaviate_client.collections.get(self.valves.COLLECTION_NAME)
            reporter and reporter(f"Scraper: starting upload of {len(all_chunks)} chunks...\n")

            # use individual insertions instead of batch to avoid gRPC
            for i, chunk in enumerate(all_chunks):
                try:
                    # create embedding for chunk text
                    chunk_text = chunk.get("text", "")
                    if not chunk_text:
                        continue
                        
                    embedding = self._get_embedding_for_text(chunk_text)
                    if embedding is None:
                        continue
                        
                    properties = {
                        "chunk_id": chunk.get("chunk_id"),
                        "chunk_type": chunk.get("chunk_type"),
                        "text": chunk_text,
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
                    
                    # use individual insert instead of batch to avoid gRPC
                    collection_obj.data.insert(properties=properties, vector=embedding.tolist())
                    
                    if (i + 1) % 5 == 0:
                        reporter and reporter(f"Uploaded {i+1}/{len(all_chunks)} chunks...\n")
                except Exception as e:
                    continue
                    
            reporter and reporter(f"Scraper: uploaded {len(all_chunks)} chunks successfully\n")
        except Exception as e:
            reporter and reporter(f"Scraper: upload error: {e}\n")
            logger.debug(traceback.format_exc())

    # --------------------------
    # Embedding helpers
    # --------------------------
    def _get_embedding_for_text(self, text: str) -> Optional["np.ndarray"]:
        """
        Get embedding for a given text using the configured embedding model.
        Args: text (str): input text
        Returns: numpy array of embedding or None on failure
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
    # Search chunks
    # --------------------------
    def search_chunks(self, query: str, k: Optional[int] = None, target_version: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks based on the query.
        Args: query (str): user query
              k (int): number of chunks to retrieve
              target_version (str): optional version filter
        Returns: list of chunk dictionaries
        """

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
            collection = self.valves.COLLECTION_NAME
            query_vector = qembed.tolist()
            where_filter = ""
            if target_version:
                where_filter = f"""
                ,where: {{
                  operator: Or
                  operands: [
                    {{
                      path: ["document_id"]
                      operator: Like
                      valueText: "*{target_version}*"
                    }},
                    {{
                      path: ["document_title"]
                      operator: Like
                      valueText: "*{target_version}*"
                    }}
                  ]
                }}
                """
            # build GraphQL query
            graphql_query = {
                "query": f"""
                {{
                  Get {{
                    {collection}(
                      nearVector: {{ vector: {json.dumps(query_vector)} }},
                      limit: {k * 3}
                      {where_filter}
                    ) {{
                      chunk_id
                      chunk_type
                      text
                      title
                      url
                      parent_id
                      chunk_index
                      document_id
                      document_title
                      document_date
                    }}
                  }}
                }}
                """
            }
            # perform REST GraphQL query
            url = f"{self.valves.WEAVIATE_URL.rstrip('/')}/v1/graphql"
            res = requests.post(url, json=graphql_query, timeout=60)
    
            if res.status_code != 200:
                logger.error(f"GraphQL query failed: {res.status_code} {res.text}")
                return []
    
            data = res.json()
            objects = (
                data.get("data", {}).get("Get", {}).get(collection, [])
                if data and "data" in data
                else []
            )            
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

    
    # --------------------------
    # create context
    # --------------------------
    def create_context(self, chunks: List[Dict[str, Any]], query_version: Optional[str] = None) -> str:
        """
        Create a context string from the retrieved chunks.
        Args: chunks (list): list of chunk dictionaries, query_version (str): optional version filter
        Returns: formatted context string
        """
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
    # LLM call
    # --------------------------
    def generate_with_llm(self, prompt: str, stream: bool = False, idle_timeout: int = 30) -> Union[str, Iterator[str]]:
        """
        Generate a response from the LLM given a prompt.
        Args: prompt (str): input prompt, stream (bool): whether to stream the response, idle_timeout (int): timeout for streaming inactivity
        Returns: response string or generator of strings
        """
        # check if it's Ollama (no API key needed)
        is_ollama = "ollama" in self.valves.LLM_MODEL_URL.lower() or ":11434" in self.valves.LLM_MODEL_URL
        
        headers = {"Content-Type": "application/json"}
        if not is_ollama and self.valves.LLM_MODEL_API_KEY:
            headers["Authorization"] = f"Bearer {self.valves.LLM_MODEL_API_KEY}"
        
        if is_ollama:
            # Ollama format
            payload = {
                "model": self.valves.LLM_MODEL_NAME,
                               "prompt": prompt,
                "stream": stream,
                "options": {"temperature": 0.1}
            }
            endpoint = f"{self.valves.LLM_MODEL_URL.rstrip('/')}/api/generate"
        else:
            # OpenAI-compatible format
            payload = {
                "model": self.valves.LLM_MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "stream": stream
            }
            endpoint = f"{self.valves.LLM_MODEL_URL.rstrip('/')}/chat/completions"
        
        try:
            r = requests.post(endpoint, headers=headers, json=payload, stream=stream, timeout=120)
            r.raise_for_status()
            
            if stream:
                def gen():
                    last_activity = time.time()
                    max_idle = idle_timeout
                    for line in r.iter_lines():
                        if line:
                            last_activity = time.time()
                            try:
                                line_str = line.decode('utf-8')
                                if line_str.startswith('data: '):
                                    line_str = line_str[6:]
                                if line_str.strip() == '[DONE]':
                                    break
                                    
                                chunk = json.loads(line_str)
                                
                                if is_ollama:
                                    # Ollama format
                                    if chunk.get("response"):
                                        yield chunk["response"]
                                    if chunk.get("done", False):
                                        break
                                else:
                                    # OpenAI format
                                    if "choices" in chunk:
                                        for c in chunk["choices"]:
                                            delta = c.get("delta", {})
                                            if isinstance(delta, dict) and delta.get("content"):
                                                yield delta.get("content")
                                            elif c.get("text"):
                                                yield c.get("text")
                            except json.JSONDecodeError:
                                # if parsing fails, try to yield raw line
                                try:
                                    if line_str and not line_str.startswith('data: '):
                                        yield line_str
                                except:
                                    pass
                            except Exception:
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
                response_data = r.json()
                if is_ollama:
                    return response_data.get("response", r.text)
                else:
                    # OpenAI format
                    if "choices" in response_data and response_data["choices"]:
                        return response_data["choices"][0].get("message", {}).get("content", r.text)
                    return response_data.get("response", r.text)
        except Exception as e:
            logger.error(f"LLM generation error: {e}", exc_info=True)
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
        Args: user_message (str): user input message
              model_id (str): model identifier
              messages (list): message history (not used here)
              body (dict): additional request body parameters
        Returns: response string or generator of strings
        """
        try:
            logger.info(f"=== PIPE CALLED ===")
            logger.info(f"Message: '{user_message}'")
            logger.info(f"Model ID: '{model_id}'") 
            logger.info(f"Body: {body}")
            logger.info(f"Stream requested: {body.get('stream', False)}")
            
            # debug print to help troubleshooting
            logger.debug(f"Pipeline called with message: {user_message[:100]}...")

            stream_requested = bool(body.get("stream", False))

            lowered = (user_message or "").lower()
            logger.info(f"Checking if message contains scraping keywords: '{lowered}'")
            
            # only trigger scraping if message explicitly contains one of these keywords
            scrape_keywords = ["scrape", "update database", "refresh data"]
            if any(keyword in lowered for keyword in scrape_keywords):
                logger.info("=== SCRAPE COMMAND DETECTED ===")

                # check dependencies before proceeding
                missing_deps = []
                if async_playwright is None:
                    missing_deps.append("playwright")
                if BeautifulSoup is None:
                    missing_deps.append("beautifulsoup4")
                    
                if missing_deps:
                    error_msg = f"Missing required dependencies: {', '.join(missing_deps)}. Please rebuild the container."
                    logger.error(error_msg)
                    return error_msg
                
                # start a background scraping thread and return a streaming generator (if requested)
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
                            coro = self.scrape_and_ingest(reporter=reporter)
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
                            loop.run_until_complete(self.scrape_and_ingest())
                            loop.close()
                        except Exception:
                            logger.exception("Background scrape failed.")

                    t = threading.Thread(target=_bgfire_and_forget, daemon=True)
                    t.start()
                    return "Scraping started in background. The database will be updated shortly."

            # normal RAG retrieval flow with version identification
            self.ensure_weaviate_client()
            if not self.weaviate_client:
                return "Error: Could not connect to Weaviate. Please try again later."

            # determine search criteria using version identification
            query_version, product_line = self._determine_search_criteria(user_message)
            logger.info(f"Identified version: {query_version}, product: {product_line}")
            
            # search with version-aware fallback strategy
            chunks, version_details = self._search_chunks_with_fallback(user_message, query_version)

            if not chunks:
                return self._generate_no_results_response(query_version, product_line)

            context = self.create_context(chunks, query_version)
            # build enhanced prompt with version context
            prompt = self._build_enhanced_prompt(context, user_message, query_version, version_details)

            result = self.generate_with_llm(prompt, stream=stream_requested, idle_timeout=25)
            return result

        except Exception as e:
            logger.error("Error in pipe()", exc_info=True)
            return f"Pipeline error: {e}"

    # --------------------------
    # Lifecycle hooks (optional)
    # --------------------------
    async def on_startup(self):
        """
        Called when the pipeline is started by Open-WebUI.
        Ensures the Weaviate client is initialized.
        """
        logger.info("RAG pipeline startup: ensuring weaviate client")
        try:
            # spawn connection attempt but don't block heavily
            self.ensure_weaviate_client(max_retries=3, delay=10)
        except Exception:
            logger.debug("Startup connect attempt failed", exc_info=True)

    async def on_shutdown(self):
        """
        Called when the pipeline is stopped by Open-WebUI.
        Cleans up resources such as the Weaviate client.
        """ 
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



# ============================================================
# PDF DOWNLOAD LOGIC
# ============================================================

async def download_pdf_strategy_1_direct_api(doc_id, title, session=None):
    """
    Download PDF directly via HPE public API.
    Args: doc_id (str): document ID
          title (str): document title (for logging)
          session (requests.Session): optional session for connection reuse
    Returns: bytes of PDF or None on failure
    """
    try:
        # urls
        api_url = f"{HPE_BASE_URL}/hpesc/public/api/document/{doc_id}"
        print(f"[PDF Download] Attempting Direct API: {api_url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36', 
            'Accept': 'application/pdf,*/*', 
            'Referer': HPE_BASE_URL
        }
        _session = session or requests.Session()
        response = _session.get(api_url, headers=headers, timeout=60, stream=True)
        if not session: 
            _session.close()
        # evaluate response
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '').lower()
            content_length = int(response.headers.get('content-length', 0))
            is_pdf_type = 'pdf' in content_type
            is_large_enough = content_length == 0 or content_length >= MIN_PDF_SIZE_BYTES
            # detailed logging
            if is_pdf_type and is_large_enough:
                pdf_bytes = response.content
                if len(pdf_bytes) >= MIN_PDF_SIZE_BYTES:
                    print(f"[PDF Download] SUCCESS: Downloaded {len(pdf_bytes)} bytes (Type: {content_type})")
                    return pdf_bytes
                else: 
                    print(f"[PDF Download] Failed: Type PDF, but actual size {len(pdf_bytes)} < {MIN_PDF_SIZE_BYTES} bytes.")
            elif not is_pdf_type: 
                print(f"[PDF Download] Failed: Content-Type '{content_type}' is not PDF.")
            elif not is_large_enough: 
                print(f"[PDF Download] Failed: Content-Length {content_length} < {MIN_PDF_SIZE_BYTES} bytes.")
        else: 
            print(f"[PDF Download] Failed: Status code {response.status_code}")
    except requests.exceptions.RequestException as e: 
        print(f"[PDF Download] Network Exception: {e}")
    except Exception as e: 
        print(f"[PDF Download] General Exception: {e}")
        traceback.print_exc()
    return None

async def download_pdf_direct_api(url, doc_id, title):
    """
    Download PDF directly via HPE public API.
    Args: doc_id (str): document ID
          title (str): document title (for logging)
          session (requests.Session): optional session for connection reuse
    Returns: bytes of PDF or None on failure
    """
    print(f"\n{'='*70}\nAttempting PDF Download: {title}\nDoc ID: {doc_id}\nURL: {url}\n{'='*70}\n")
    session = requests.Session()
    pdf_bytes = None
    try: 
        pdf_bytes = await download_pdf_strategy_1_direct_api(doc_id, title, session)
    except Exception as e: 
        print(f"Error during PDF download call: {e}")
    finally: 
        session.close()

    if pdf_bytes:
        print(f"\n{'='*70}\nPDF DOWNLOAD SUCCESS\nSize: {len(pdf_bytes):,} bytes\n{'='*70}\n")
        return pdf_bytes, "Direct API"
    else:
        print(f"\n{'='*70}\nPDF DOWNLOAD FAILED\n{'='*70}\n")
        return None, None

def save_pdf_to_disk(pdf_bytes, title):
    """
    Save PDF bytes to disk with a safe filename.
    Args: pdf_bytes (bytes): PDF content
          title (str): document title for filename
    Returns: path to saved PDF or None on failure
    """
    try:
        # clean filename for safe disk storage
        if not title: 
            title = "untitled"
        clean = re.sub(r'[^\w\s\-\.]', '', str(title))
        clean = re.sub(r'[-\s]+', '_', clean).strip('_')
        base, ext = os.path.splitext(clean)
        clean = base[:100].strip('_') + ext
        safe_filename = (clean if clean else "document") + ".pdf"
        
        pdf_path = PDF_STORAGE_DIR / Path(safe_filename)
        pdf_path.write_bytes(pdf_bytes)
        logger.debug(f"PDF saved to: {pdf_path}")
        return str(pdf_path)
    except Exception as e: 
        logger.error(f"Error saving PDF '{title}': {e}")
        return None

