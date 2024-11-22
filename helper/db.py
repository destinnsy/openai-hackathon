from dotenv import load_dotenv
load_dotenv()  # This will load the environment variables from the .env file

import chromadb
import os
from openai import OpenAI

chroma_client = chromadb.EphemeralClient()

openai_client = OpenAI(
            api_key=os.environ.get("OPENAI_KEY"),
        )