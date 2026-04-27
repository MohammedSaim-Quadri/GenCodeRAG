from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    CollectionStatus,
    CollectionConfig,
    PayloadSchemaType,
    PointStruct
)
from dotenv import load_dotenv
import os
from tqdm import tqdm
import time
from logger import setup_logger
from settings import settings
logger = setup_logger(__name__)

# === Load model ===
print("🧠 Loading embedding model...")
logger.info("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Load code chunks ===
CHUNK_FILE = Path("data/chunks/github_code_chunks.jsonl")
print("📦 Reading code chunks...")
logger.info(f"Reading code chunks from {CHUNK_FILE}...")
with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]
    if not chunks:
        raise ValueError("No chunks found in chunk file.")
texts = [chunk["code"] for chunk in chunks]

# === Compute embeddings ===
print(f"🔢 Computing embeddings for {len(texts)} chunks...")
logger.info(f"Computing embeddings for {len(texts)} chunks...")
embeddings = model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=True)

client = QdrantClient(
    url=settings.QDRANT_HOST, 
    api_key=settings.QDRANT_API_KEY,
    timeout=120
)

# Create collection if it doesn't exist
if not client.collection_exists(collection_name=settings.COLLECTION_NAME):
    client.create_collection(
    collection_name=settings.COLLECTION_NAME,
    vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
    )

    # Create payload indexes for filtering (after collection creation)
    print("🔧 Creating payload indexes...")
    logger.info("Creating payload indexes for filtering...")
    try:
        client.create_payload_index(
            collection_name=settings.COLLECTION_NAME,
            field_name="language",
            field_type="keyword"
        )
        print("✅ Created index for 'language' field")
        logger.info("Created index for 'language' field")
        
        # Optional: Create other indexes if needed
        client.create_payload_index(
            collection_name=settings.COLLECTION_NAME,
            field_name="repo",
            field_type="text"
        )
        print("✅ Created index for 'repo' field")
        logger.info("Created index for 'repo' field")
        
        client.create_payload_index(
            collection_name=settings.COLLECTION_NAME,
            field_name="path",
            field_type="text"
        )
        print("✅ Created index for 'path' field")
        logger.info("Created index for 'path' field")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not create indexes: {e}")
        print("   You may need to create them manually later")
        logger.warning(f"Could not create indexes: {e}")
        logger.warning("Indexes may need to be created manually later")

# === Upload to Qdrant ===
print(f"📤 Uploading {len(embeddings)} vectors to Qdrant...")
logger.info(f"Uploading {len(embeddings)} vectors to Qdrant...")

points = [
    PointStruct(
        id=chunks[i]["chunk_id"],
        vector=embeddings[i],
        payload=chunks[i]
    )
    for i in range(len(embeddings))
]

def upload_batch_with_retry(client, collection_name, batch, max_retries=3):
    """Upload a batch with retry logic"""
    for attempt in range(max_retries):
        try:
            client.upsert(collection_name=collection_name, points=batch)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"⚠️  Retry {attempt + 1}/{max_retries} after {wait_time}s due to: {str(e)[:100]}...")
                logger.warning(f"Retry {attempt + 1}/{max_retries} for batch upload due to: {str(e)[:100]}...")
                time.sleep(wait_time)
            else:
                print(f"❌ Failed after {max_retries} attempts: {e}")
                logger.error(f"Failed to upload batch after {max_retries} attempts: {e}")
                return False
    return False

batch_size = 100
successful_uploads = 0

print(f"Starting batch upload with batch size: {batch_size}")
logger.info(f"Starting batch upload to Qdrant with batch size: {batch_size}")

for i in tqdm(
    range(0, len(points), batch_size),
    desc="📤 Uploading to Qdrant"
    ):
    batch = points[i:i+batch_size]
    if upload_batch_with_retry(client, settings.COLLECTION_NAME, batch):
        successful_uploads += len(batch)
        print(f"✅ Uploaded batch {i // batch_size + 1}")
        logger.info(f"Uploaded batch {i // batch_size + 1} successfully")
    else:
        print(f"❌ Failed to upload batch {i//batch_size + 1}")
        logger.error(f"Failed to upload batch {i//batch_size + 1}")

print(f"\n✅ Successfully uploaded {successful_uploads}/{len(points)} vectors")
logger.info(f"Finished uploading vectors. Successfully uploaded {successful_uploads}/{len(points)} vectors")

if __name__ == "__main__":
    print("🚀 Starting Qdrant embedding upload...")
    logger.info("Starting Qdrant embedding upload...")

    print(f"\n✅ Successfully uploaded {successful_uploads}/{len(points)} vectors")
    logger.info(f"Finished uploading vectors. Successfully uploaded {successful_uploads}/{len(points)} vectors")

    try:
        info = client.get_collection(collection_name=settings.COLLECTION_NAME)
        print(f"📊 Total vectors in collection: {info.points_count}")
        logger.info(f"Total vectors in collection after upload: {info.points_count}")
    except Exception as e:
        print(f"❌ Could not confirm upload: {e}")
        logger.error(f"Could not confirm upload: {e}")
