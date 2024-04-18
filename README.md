# RAG-Atlas-LlamaIndex
Example of usage of RAG, MongoDB Atlas as document collection and vector store. LlamaIndex to process the data.
This example uses documents from a collection that contains movie synopsis, this collection used was a dummy collection from MongoDB University.
Sentence Window Retrieval was the approach to use as a retriever, this was handled by LlamaIndex taking the text into nodes and then save the embeddings into our Atlas Vector Search.
LlamaIndex's query was used to call OpenAI's API, I tried to use LangChain to handle this call but currently it doesn't support the embeddings generated in the Atlas Vector Search.
🍃
