import os
import re
import time
import numpy as np
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, PayloadSchemaType
from langchain.embeddings import OpenAIEmbeddings
from typing import List, Dict, Any, Optional, Union, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Load environment variables
load_dotenv()


class SECFilingPipeline:
    """Pipeline for fetching SEC filings and storing them in Qdrant."""
    
    def __init__(
        self,
        fmp_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        qdrant_url: str = "http://74.208.122.216:6333",
        collection_name: str = "filings",
        email: str = "x.tan@traderverse.io",
        name: str = "Traderware"
    ):
        """Initialize the SEC filing pipeline.
        
        Args:
            fmp_api_key: Financial Modeling Prep API key
            openai_api_key: OpenAI API key for embeddings
            qdrant_url: URL of the Qdrant server
            collection_name: Name of the Qdrant collection
            email: Email for SEC downloader
            name: Name for SEC downloader
        """
        # Set API keys
        self.fmp_api_key = fmp_api_key or os.getenv("FMP_API_KEY")
        if not self.fmp_api_key:
            raise ValueError("FMP API key is required")
        
        openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        self.qdrant_client = QdrantClient(url=qdrant_url,timeout=30)
        self.collection_name = collection_name
        
        # Import helpers late to avoid circular imports
        from helper import (
            processing_html2txt,
            combine_sentences,
            calculate_cosine_distances,
            find_appropriate_threshold,
        )
        from sec_downloader import Downloader
        from pdf_to_gcp import HtmlToPdfGcpUploader
        
        self.processing_html2txt = processing_html2txt
        self.combine_sentences = combine_sentences
        self.calculate_cosine_distances = calculate_cosine_distances
        self.find_appropriate_threshold = find_appropriate_threshold
        self.downloader = Downloader(name, email)
        self.html_pdf_uploader = HtmlToPdfGcpUploader()
        
        # Ensure collection exists
        self._ensure_collection()
    
    def _ensure_collection(self) -> None:
        """Ensure the Qdrant collection exists with proper configuration."""
        # Check if collection exists, if not create it
        if not self.qdrant_client.collection_exists(self.collection_name):
            # We don't know vector size yet, but we'll recreate after first embedding if needed
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1536,  # Default OpenAI embedding size
                    distance=Distance.COSINE
                )
            )
        
        # Ensure datetime index on "date"
        try:
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="date",
                field_schema=PayloadSchemaType.DATETIME,
                wait=True
            )
        except Exception:
            # Ignore if already exists
            pass
    
    def fetch_filings(
        self,
        symbol: str,
        date_from: str,
        date_to: str,
        form_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Fetch SEC filings for a symbol within a date range.
        
        Args:
            symbol: Stock ticker symbol
            date_from: Start date in YYYY-MM-DD format
            date_to: End date in YYYY-MM-DD format
            form_types: List of form types to filter by (default: ["8-K", "10-K", "10-Q"])
            
        Returns:
            List of filing data dictionaries
        """
        if form_types is None:
            form_types = ["8-K", "10-K", "10-Q"]
        
        # Convert date strings to datetime objects for comparison
        from_date = datetime.strptime(date_from, "%Y-%m-%d")
        to_date = datetime.strptime(date_to, "%Y-%m-%d")
        
        # Split date range into yearly chunks due to API limitation (typically only returns ~1 year of data)
        all_filings = []
        current_from = from_date
        
        while current_from < to_date:
            # Calculate the end date for this chunk (1 year from start or the end date, whichever is sooner)
            chunk_to = min(datetime(current_from.year + 1, current_from.month, current_from.day), to_date)
            
            # Format dates for API request
            chunk_from_str = current_from.strftime("%Y-%m-%d")
            chunk_to_str = chunk_to.strftime("%Y-%m-%d")
            
            logging.info(f"Fetching filings for {symbol} from {chunk_from_str} to {chunk_to_str}")
            
            url = f"https://financialmodelingprep.com/stable/sec-filings-search/symbol?symbol={symbol}&from={chunk_from_str}&to={chunk_to_str}&apikey={self.fmp_api_key}"
            
            try:
                response = requests.get(url)
                response.raise_for_status()
                chunk_data = response.json()
                
                if isinstance(chunk_data, list):
                    # Filter by form type if specified
                    if form_types:
                        chunk_data = [filing for filing in chunk_data if filing.get("formType") in form_types]
                    
                    # Add filings to the result list
                    all_filings.extend(chunk_data)
                    logging.info(f"Found {len(chunk_data)} filings for {symbol} from {chunk_from_str} to {chunk_to_str}")
                else:
                    logging.warning(f"Unexpected response format for {symbol} from {chunk_from_str} to {chunk_to_str}")
                
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching filings for {symbol} from {chunk_from_str} to {chunk_to_str}: {e}")
            
            # Move to the next chunk
            current_from = chunk_to + timedelta(days=1)
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
        
        # Remove duplicates (in case of overlapping date ranges)
        unique_filings = {}
        for filing in all_filings:
            # Create a unique key for each filing
            key = f"{filing.get('symbol')}_{filing.get('formType')}_{filing.get('filingDate')}"
            unique_filings[key] = filing
        
        # Sort filings by date (newest first)
        sorted_filings = sorted(
            unique_filings.values(), 
            key=lambda x: x.get('filingDate', ''), 
            reverse=True
        )
        
        logging.info(f"Total filings found for {symbol} from {date_from} to {date_to}: {len(sorted_filings)}")
        return sorted_filings
    
    def process_filing(self, filing: Dict[str, Any], timeout: int = 120, max_retries: int = 3) -> bool:
        """Process a single filing and insert into Qdrant.
        
        Args:
            filing: Filing data dictionary
            timeout: Timeout in seconds for download operations
            max_retries: Maximum number of retry attempts for timeouts
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Processing {filing['symbol']} {filing['formType']} from {filing['filingDate']}")
            
            # 1️⃣ Fetch & clean HTML with retry logic
            html = None
            retries = 0
            
            while html is None and retries <= max_retries:
                try:
                    # Try primary downloader with timeout
                    html = self.downloader.download_filing(
                        url=filing["finalLink"], 
                    ).decode()
                except requests.exceptions.Timeout:
                    retries += 1
                    if retries <= max_retries:
                        wait_time = 2 ** retries  # Exponential backoff
                        print(f"Download timed out. Retry {retries}/{max_retries} after {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"Download timed out after {max_retries} retries. Trying alternative method...")
                except Exception as e:
                    print(f"Failed to download with SEC downloader: {e}")
                    break  # Try alternative method
            
            # If primary method failed, try alternative
            if html is None:
                try:
                    html = self.html_pdf_uploader.download_using_request(
                        filing["finalLink"], 
                        timeout=timeout
                    )
                except requests.exceptions.Timeout:
                    print(f"Alternative download timed out. Skipping filing.")
                    return False
                except Exception as e:
                    print(f"Failed to download with alternative method: {e}")
                    return False
            
            raw_text = self.processing_html2txt(html)
            
            # 2️⃣ Sentence splitting & combine
            sentence_texts = re.split(r"(?<=[.#:])\s+", raw_text)
            sentences = [{"sentence": s, "index": i} for i, s in enumerate(sentence_texts)]
            sentences = self.combine_sentences(sentences)
            
            # 3️⃣ Embed sentences for semantic chunking
            sent_embeds = self.embeddings.embed_documents([s["combined_sentence"] for s in sentences])
            for i, emb in enumerate(sent_embeds):
                sentences[i]["combined_sentence_embedding"] = emb
            
            # 4️⃣ Semantic chunking
            distances, sentences = self.calculate_cosine_distances(sentences)
            threshold, _, _ = self.find_appropriate_threshold(sentences, distances, 95, 1000)
            break_idx = np.percentile(distances, threshold)
            boundaries = [i for i, d in enumerate(distances) if d > break_idx]
            
            chunk_texts = []
            start = 0
            for b in boundaries:
                chunk_texts.append(" ".join(s["sentence"] for s in sentences[start : b + 1]))
                start = b + 1
            if start < len(sentences):
                chunk_texts.append(" ".join(s["sentence"] for s in sentences[start:]))
            
            # 5️⃣ Embed chunks
            try:
                chunk_embeddings = self.embeddings.embed_documents(chunk_texts)
            except Exception as e:
                print(f"Chunks Embedding: {e}")
            # Check if collection needs to be recreated with correct vector size
            if not self.qdrant_client.collection_exists(self.collection_name):
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=len(chunk_embeddings[0]),
                        distance=Distance.COSINE
                    )
                )
            print('ew')
            # 8️⃣ Parse & normalize filing date
            raw_date = filing["filingDate"].split(" ")[0]  # e.g. "2025-01-30"
            try:
                dt = datetime.strptime(raw_date, "%Y-%m-%d")
            except ValueError:
                dt = datetime.fromisoformat(raw_date)
            safe_date_iso = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # 9️⃣ Build points
            try:
                base_id = int(datetime.now().timestamp() * 1000)
                points = []
                for idx, (chunk, vector) in enumerate(zip(chunk_texts, chunk_embeddings)):
                    points.append(
                        PointStruct(
                            id=base_id + idx,
                            vector=vector,
                            payload={
                                "content": chunk,
                                "file_name": f"{filing['symbol']}_{filing['formType']}_{raw_date}",
                                "ticker": filing["symbol"],
                                "filing_type": filing["formType"],
                                "date": safe_date_iso,
                            }
                        )
                    )
            except Exception as e:
                print(f"Failed to build points: {e}")
            # 🔟 Upsert into Qdrant
            try:
                self.qdrant_client.upsert(collection_name=self.collection_name, points=points)
                print(f"✅ Inserted {len(points)} chunks into Qdrant for: {filing['symbol']} {filing['formType']}")
                return True
            except Exception as e:
                print(f"Failed to build points: {e}")
            
        except Exception as e:
            print(f"Error processing filing {filing.get('symbol', 'unknown')} {filing.get('formType', 'unknown')}: {e}")
            return False
    
    def process_filings(
        self,
        filings: List[Dict[str, Any]],
        batch_size: int = 1,
        delay_seconds: float = 0.0,
        timeout: int = 120,
        max_retries: int = 3
    ) -> Tuple[int, int]:
        """Process multiple filings in batches.
        
        Args:
            filings: List of filing data dictionaries
            batch_size: Number of filings to process in parallel
            delay_seconds: Delay between processing filings in seconds
            timeout: Timeout in seconds for download operations
            max_retries: Maximum number of retry attempts for timeouts
            
        Returns:
            Tuple of (successful_count, total_count)
        """
        successful = 0
        total = len(filings)
        failed_filings = []
        
        for i in range(0, total, batch_size):
            batch = filings[i:i + batch_size]
            
            for filing in batch:
                if self.process_filing(filing, timeout=timeout, max_retries=max_retries):
                    successful += 1
                else:
                    failed_filings.append(filing)
                
                if delay_seconds > 0 and i + batch_size < total:
                    time.sleep(delay_seconds)
        
        # Report failed filings
        if failed_filings:
            print(f"\n❌ Failed to process {len(failed_filings)} filings:")
            for filing in failed_filings:
                print(f"  - {filing['symbol']} {filing['formType']} from {filing['filingDate']}")
        
        return successful, total
    
    def run(
        self,
        symbol: str,
        date_from: str,
        date_to: str,
        form_types: Optional[List[str]] = None,
        batch_size: int = 1,
        delay_seconds: float = 0.0,
        timeout: int = 120,
        max_retries: int = 3
    ) -> Tuple[int, int]:
        """Run the full pipeline for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            date_from: Start date in YYYY-MM-DD format
            date_to: End date in YYYY-MM-DD format
            form_types: List of form types to filter by (default: ["8-K", "10-K", "10-Q"])
            batch_size: Number of filings to process in parallel
            delay_seconds: Delay between processing filings in seconds
            timeout: Timeout in seconds for download operations
            max_retries: Maximum number of retry attempts for timeouts
            
        Returns:
            Tuple of (successful_count, total_count)
        """
        # Fetch filings
        filings = self.fetch_filings(symbol, date_from, date_to, form_types)
        
        if not filings:
            print(f"No filings found for {symbol} from {date_from} to {date_to}")
            return 0, 0
        
        print(f"Found {len(filings)} filings for {symbol}")
        
        # Process filings with timeout and retry parameters
        successful = 0
        total = len(filings)
        
        for i in range(0, total, batch_size):
            batch = filings[i:i + batch_size]
            
            for filing in batch:
                if self.process_filing(filing, timeout=timeout, max_retries=max_retries):
                    successful += 1
                
                if delay_seconds > 0 and i + batch_size < total:
                    time.sleep(delay_seconds)
        
        return successful, total


def main():
    """Example usage of the SEC filing pipeline."""
    load_dotenv()
    
    # Get API keys from environment
    fmp_api_key = os.getenv("FMP_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Create pipeline
    pipeline = SECFilingPipeline(
        fmp_api_key=fmp_api_key,
        openai_api_key=openai_api_key
    )
    
    # Run pipeline for a multi-year period
    symbol = "AAPL"
    date_from = "2021-01-01"  # Going back several years
    date_to = "2025-04-16"
    form_types = ["8-K", "10-K", "10-Q"]
    
    logging.info(f"Starting pipeline for {symbol} from {date_from} to {date_to}")
    
    successful, total = pipeline.run(
        symbol=symbol,
        date_from=date_from,
        date_to=date_to,
        form_types=form_types,
        batch_size=1,          # Process one filing at a time
        delay_seconds=2.0,     # 2 second delay between filings
        timeout=180,           # 3 minute timeout for downloads
        max_retries=3          # 3 retries for failed downloads
    )
    
    print(f"Processing complete: {successful}/{total} filings successfully processed")
    
    if successful < total:
        print(f"⚠️ {total - successful} filings failed to process. Check logs for details.")


if __name__ == "__main__":
    main()