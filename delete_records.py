#!/usr/bin/env python3
"""
Script to delete all 8-K filings from a Qdrant vector database collection.
This script connects to a Qdrant instance, finds all points with 
filing_type "8-K", and deletes them.

You can edit the configuration parameters below instead of using command line arguments.
"""

from qdrant_client import QdrantClient
import logging
import sys

# ============================================================================
# CONFIGURATION - Edit these parameters as needed
# ============================================================================
# Required parameters
COLLECTION_NAME = "filings"  # Name of your Qdrant collection

# Connection parameters
QDRANT_HOST = "74.208.122.216"    # Qdrant server host
QDRANT_PORT = 6333           # Qdrant server port
QDRANT_HTTPS = False         # Use HTTPS connection
QDRANT_API_KEY = None        # API key (if required, otherwise None)
QDRANT_PREFIX = None         # URL prefix (if required, otherwise None)

# Processing parameters
BATCH_SIZE = 100             # Number of records to process in a batch
DRY_RUN = False               # Set to False to actually delete records
# ============================================================================

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def connect_to_qdrant():
    """Establish connection to Qdrant server."""
    try:
        if QDRANT_HTTPS:
            # HTTPS connection
            url = f"https://{QDRANT_HOST}"
            if QDRANT_PORT != 443:
                url += f":{QDRANT_PORT}"
            if QDRANT_PREFIX:
                url += f"/{QDRANT_PREFIX}"
            
            client = QdrantClient(
                url=url,
                api_key=QDRANT_API_KEY
            )
        else:
            # HTTP connection
            client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                prefix=QDRANT_PREFIX,
                api_key=QDRANT_API_KEY
            )
        
        logger.info(f"Successfully connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {str(e)}")
        raise

def get_8k_filings(client):
    """Get all points with filing_type '8-K'."""
    try:
        # Some versions of Qdrant client don't support filtering in scroll operation
        # So we'll get all points and filter manually
        logger.info("Retrieving all points from collection...")
        
        # Get all points first (without vectors to save bandwidth)
        all_points = []
        offset = 0
        
        while True:
            try:
                # Try older style API call first
                batch = client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=None,  # Get all points, not vector search
                    limit=BATCH_SIZE,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
            except Exception:
                # Try newer style API call
                try:
                    batch = client.search(
                        collection_name=COLLECTION_NAME,
                        query=None,  # Get all points, not vector search
                        limit=BATCH_SIZE,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False
                    )
                except Exception as e:
                    # Last resort, try the scroll API
                    try:
                        scroll_result = client.scroll(
                            collection_name=COLLECTION_NAME,
                            limit=BATCH_SIZE,
                            offset=offset,
                            with_payload=True,
                            with_vectors=False
                        )
                        batch, next_offset = scroll_result
                        if next_offset is None:
                            next_batch_available = False
                        else:
                            offset = next_offset
                    except Exception as e:
                        logger.error(f"All retrieval methods failed: {str(e)}")
                        raise
            
            if not batch:
                break
                
            logger.info(f"Retrieved batch of {len(batch)} points")
            all_points.extend(batch)
            
            # If we got fewer points than the batch size, we're done
            if len(batch) < BATCH_SIZE:
                break
                
            offset += BATCH_SIZE
        
        logger.info(f"Retrieved a total of {len(all_points)} points")
        
        # Filter for 8-K filings
        filing_8k_points = []
        
        for point in all_points:
            # Handle different response structures
            if hasattr(point, 'payload') and isinstance(point.payload, dict):
                payload = point.payload
                point_id = point.id
            elif isinstance(point, dict) and 'payload' in point:
                payload = point['payload']
                point_id = point['id']
            else:
                logger.warning(f"Unexpected point structure, skipping: {point}")
                continue
            
            # Check if this point has filing_type = 8-K
            if 'filing_type' in payload and payload['filing_type'] == '8-K':
                filing_8k_points.append(point_id)
        
        logger.info(f"Found {len(filing_8k_points)} points with filing_type '8-K'")
        return filing_8k_points
    
    except Exception as e:
        logger.error(f"Error retrieving 8-K filings: {str(e)}")
        raise

def delete_points(client, point_ids):
    """Delete points by their IDs in batches."""
    if not point_ids:
        logger.info("No 8-K filings found to delete")
        return
    
    total_points = len(point_ids)
    
    if DRY_RUN:
        logger.info(f"DRY RUN: Would delete {total_points} 8-K filings")
        # Print first few IDs as sample
        if point_ids:
            sample = point_ids[:5]
            sample_str = ", ".join(str(id) for id in sample)
            logger.info(f"Sample IDs that would be deleted: {sample_str}...")
        return
    
    logger.info(f"Deleting {total_points} 8-K filings...")
    
    # Process in batches to avoid overwhelming the server
    successful_deletions = 0
    for i in range(0, total_points, BATCH_SIZE):
        batch = point_ids[i:i + BATCH_SIZE]
        
        # Try each ID individually as a last resort
        for point_id in batch:
            try:
                # Try the most basic delete operation - delete a single point by ID
                client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=[point_id]  # Just a single ID in a list
                )
                successful_deletions += 1
                
                # Log progress periodically
                if successful_deletions % 10 == 0:
                    logger.info(f"Deleted {successful_deletions}/{total_points} points")
                
            except Exception as e:
                logger.error(f"Failed to delete point {point_id}: {str(e)}")
    
    logger.info(f"Deletion complete. Successfully deleted {successful_deletions}/{total_points} points.")

def main():
    """Main function to execute the script."""
    logger.info("Starting 8-K filing deletion process")
    logger.info(f"Collection: {COLLECTION_NAME}")
    logger.info(f"Qdrant connection: {QDRANT_HOST}:{QDRANT_PORT}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Dry run: {'Yes' if DRY_RUN else 'No'}")
    
    try:
        # Connect to Qdrant
        client = connect_to_qdrant()
        
        # Check if collection exists
        try:
            collections = client.get_collections().collections
            collection_names = [collection.name for collection in collections]
        except Exception:
            # Try an alternative method to check if collection exists
            try:
                collection_info = client.get_collection(collection_name=COLLECTION_NAME)
                collection_names = [COLLECTION_NAME]  # If this works, the collection exists
            except Exception as e:
                logger.error(f"Error checking collections: {str(e)}")
                collection_names = []
        
        if COLLECTION_NAME not in collection_names:
            logger.error(f"Collection '{COLLECTION_NAME}' does not exist")
            return 1
        
        # Get point IDs for 8-K filings
        filing_ids = get_8k_filings(client)
        
        if not filing_ids:
            logger.info(f"No 8-K filings found in collection '{COLLECTION_NAME}'")
            return 0
        
        # Delete the 8-K filings
        delete_points(client, filing_ids)
        
        if not DRY_RUN:
            logger.info(f"Successfully deleted {len(filing_ids)} 8-K filings from collection '{COLLECTION_NAME}'")
        else:
            logger.info(f"DRY RUN COMPLETE - {len(filing_ids)} 8-K filings would be deleted")
        
        return 0
    
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())