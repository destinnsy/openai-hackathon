from dotenv import load_dotenv
load_dotenv()  # This will load the environment variables from the .env file

import chromadb
import os
from openai import OpenAI
from loguru import logger
from pydantic import BaseModel, Field
from chromadb import  EmbeddingFunction, Embeddings

chroma_client = chromadb.EphemeralClient()

openai_client = OpenAI(
            api_key=os.environ.get("OPENAI_KEY"),
        )


class CustomOpenAIEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: list[str]) -> Embeddings:
        formatted_input = [text.replace("\n", " ") for text in input]
        emb_resp = openai_client.embeddings.create(input=formatted_input, model='text-embedding-3-small').data
        return [emb.embedding for emb in emb_resp]

collection = chroma_client.get_or_create_collection(name="user_history", embedding_function=CustomOpenAIEmbeddingFunction())

# deleting
def delete_from_index(collection, ids):
    collection.delete(
        ids=ids
    )

import uuid
# inserting
def insert_to_index(collection, documents, metadatas=None):
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=[str(uuid.uuid4()) for i in range(len(documents))],
    )
    logger.info(f"Successfully inserted {len(documents)} documents")

class QueryResult(BaseModel):
    ids: list[list[str]]
    documents: list[list[str]]
    distances: list[list[float]]

# querying https://docs.trychroma.com/guides#filtering-by-metadata
def query_index(query_texts, n_results=1, where=None, where_document=None):
    query_result = collection.query(
        query_texts=query_texts,
        n_results=n_results,
        where=where,
        where_document=where_document,
    )
    return QueryResult(**query_result)

def update_index(ids, documents, metadatas=None):
    collection.update(
        ids=ids,
        metadatas=metadatas,
        documents=documents,
    )
    logger.info(f"Successfully updated {len(ids)} documents")