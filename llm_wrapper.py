"""
llm_wrapper.py
--------------
Wrapper classes for AWS Bedrock:
- Titan embeddings for vector DB (FAISS).
- Llama3 for text generation.
- Stable Diffusion XL for image generation.
"""

import boto3
import json
import base64
import os
import random
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
import numpy as np
import faiss

# Define the Titan embeddings class
class TitanEmbeddings(Embeddings):
    """Wrapper for AWS Titan Embedding model."""

    def __init__(self, region_name="us-east-1", model_id="amazon.titan-embed-text-v1"):
        self.client = boto3.client("bedrock-runtime", region_name=region_name)
        self.model_id = model_id

    def embed_documents(self, texts):
        """Embed a list of documents."""
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text):
        """Embed a singlequery document."""
        return self._embed_text(text)

    def _embed_text(self, text):
        """Internal helper for calling Titan embedding model."""
        body = json.dumps({"inputText": text})
        # Call AWS Bedrock to invoke the model
        response = self.client.invoke_model(
            body=body,                       # JSON request payload
            modelId=self.model_id,          # Which model to use (Titan here)
            accept="application/json",       # We expect JSON back
            contentType="application/json"   # We are sending JSON
        )
        # Parse the JSON response from Bedrock
        result = json.loads(response["body"].read())
        return result["embedding"]

# Define the Bedrock client for Llama3 and Stable Diffusion
class BedrockClient:
    """Wrapper for AWS Bedrock LLMs."""
    def __init__(self, region_name="us-east-1"):
        self.client = boto3.client("bedrock-runtime", region_name=region_name)

    # -------- Llama3 (Text Generation) --------
    def generate_text(self, prompt: str):
        """Generate text using the Llama3 model."""
        model_id = "meta.llama3-70b-instruct-v1:0"
        # Prepare the request body as JSON
        body = json.dumps({ 
            "prompt": prompt,   # User prompt
            "max_gen_len": 512, # Maximum length of generated text
            "top_p": 0.9,       # Nucleus sampling parameter
            "temperature": 0.5   # Sampling temperature
        })
        # Call AWS Bedrock to invoke the model
        response = self.client.invoke_model(
            body=body,                       # JSON request payload
            modelId=model_id,                # Which model to use (Llama3 here)
            contentType="application/json",  # We are sending JSON
            accept="application/json"        # We expect JSON back
        )
        response_body = json.loads(response['body'].read())
        return response_body["generation"]

    # -------- Stable Diffusion (Image Generation) --------
    def generate_image(self, prompt: str, out_dir="output_images"):
        """Generate an image using the Stable Diffusion XL model."""
        model_id = "stability.stable-diffusion-xl-v1"
        seed = random.randint(0, 4294967295) # Random seed for image generation

        request = {
            "text_prompts": [{"text": prompt}], # user prompt
            "style_preset": "photographic",     # Style (photo-realistic)
            "seed": seed,                       # Random seed for variation
            "cfg_scale": 10,                    # How strongly the model follows the prompt (higher = more literal)
            "steps": 50,                        # Number of diffusion steps (higher = better quality, but slower)
            "width": 1024,                      # Width of the generated image
            "height": 1024                      # Height of the generated image
        }

        # Convert the Python dict into a JSON string
        # Bedrock requires the body as a JSON-formatted string
        body = json.dumps(request)
         # Call AWS Bedrock to invoke the model
        response = self.client.invoke_model(
            body=body,                       # JSON request payload
            modelId=model_id,                # Which model to use (Stable Diffusion XL here)
            contentType="application/json",  # We are sending JSON
            accept="application/json"        # We expect JSON back
        )

        response_body = json.loads(response["body"].read()) # Parse the JSON response from Bedrock
        # The Stable Diffusion response contains a list of "artifacts" (images, metadata, etc.)
        artifact = response_body.get("artifacts")[0]
        # Extract the Base64-encoded image string from the artifact
        # `.get("base64")` returns the image encoded in Base64 format
        # `.encode("utf-8")` converts that string into a byte string (required for decoding step)
        image_encoded = artifact.get("base64").encode("utf-8")
        # Decode the Base64-encoded image into raw binary (PNG/JPEG) bytes
        image_bytes = base64.b64decode(image_encoded)

        # Save the image to a file
        os.makedirs(out_dir, exist_ok=True)
        image_path = f"{out_dir}/generated_{seed}.png"
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        return image_path


# Helper: Build FAISS VectorStore
def create_faiss_store(docs, region="us-east-1"):
    """Create a FAISS vector store from the given documents."""
    embeddings = TitanEmbeddings(region_name=region)
    index = faiss.IndexFlatL2(len(embeddings.embed_query("test")))
    vectorstore = FAISS(
        embedding_function=embeddings, # Embedding function for text documents
        index=index, # FAISS index for similarity search
        docstore=InMemoryDocstore({}), # In-memory document store
        index_to_docstore_id={} # Mapping from index to document store ID
    )
    # Add documents to the vector store
    vectorstore.add_documents([Document(page_content=d) for d in docs])
    return vectorstore
