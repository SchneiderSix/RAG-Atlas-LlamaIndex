from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor, SentenceTransformerRerank)
from llama_index.core import Document
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings
from llama_index.core import VectorStoreIndex
import os

# Load env vars
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
mongo_uri = os.getenv("MONGO_URI")

# Setup MongoDB connection
client = MongoClient(mongo_uri)
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Connected to MongoDB Atlas! 🍃")
except Exception as e:
    print(e)

# Set MongoDB Atlas db, colecction and vi
DB_NAME = "sample_mflix"
db = client[DB_NAME]
COLLECTION_NAME = "movies"
collection = db[COLLECTION_NAME]
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

# Load the data
# Set a number of documents to find
data = collection.find({}, {"_id": 0, "title": 1, "fullplot": 1}).limit(100)
parsed_data_list = [" "+i.get("title", "")+": " +
                    i.get("fullplot", "") for i in data]

# print("---\n"+"".join(parsed_data_list))

# Specify llm
model = ChatOpenAI(
    temperature=0.5,
    model="gpt-3.5-turbo"
)

Settings.llm = model
# The temperature is used to control the randomness of the output.
# When you set it higher, you'll get more random outputs.
# When you set it lower, towards 0, the values are more deterministic.

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small", dimensions=256)

# Create the sentence window node parser w/ default settings,
# this approach expands the context by including a couple of surrounding sentences alongside the retrieved sentence
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,  # The number of sentences on each side of a sentence to capture
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

# Create documents for Node Parser
documents = [Document(text=i) for i in parsed_data_list]
sentence_nodes = node_parser.get_nodes_from_documents(documents)

# Embedding nodes from documents
for node in sentence_nodes:
    node.embedding = Settings.embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )

# Create embeddings in atlas Vector Store
vector_search = MongoDBAtlasVectorSearch(
    client,
    db_name=DB_NAME,
    collection_name=COLLECTION_NAME,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
)

# Add Sentence Nodes to Atlas Vector
vector_search.add(sentence_nodes)


# Index from Vector Store in Atlas
index_sentences = VectorStoreIndex.from_vector_store(vector_search)

# In advanced RAG, the MetadataReplacementPostProcessor is used to replace the sentence in each node
# with it's surrounding context as part of the sentence-window-retrieval method.
# The target key defaults to 'window' to match the node_parser's default
postproc = MetadataReplacementPostProcessor(target_metadata_key="window")

# For advanced RAG, add a re-ranker to re-ranks the retrieved context for its relevance to the query.
# Note : Retrieve a larger number of similarity_top_k, which will be reduced to top_n.
# BAAI/bge-reranker-base
# link: https://huggingface.co/BAAI/bge-reranker-base
rerank = SentenceTransformerRerank(top_n=2, model="BAAI/bge-reranker-base")

# Question to be answered
question = "What is about The Perils of Pauline?"

# Build query from our index with specific parameters
response = index_sentences.as_query_engine(
    similarity_top_k=6,
    vector_store_query_mode="hybrid",
    alpha=0.5,
    node_postprocessors=[postproc, rerank],
).query(
    question
)

# Celebrate 🍃
print(response)
