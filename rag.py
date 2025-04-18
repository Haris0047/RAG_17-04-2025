import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# ✅ Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# ✅ MongoDB Atlas setup
client = MongoClient(MONGO_URI)
db = client["qualitative"]
fillings = db["fillings"]
earnings = db["earnings"]

fillings_index = "fillings"
earnings_index = "earnings"

# ✅ Embedding model
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# ✅ Vector Store (no upload, just connect)
vector_store_fillings = MongoDBAtlasVectorSearch(
    embedding=embedding_model,
    collection=fillings,
    index_name=fillings_index,
    text_key="content"  # <--- ✅ very important to match the key storing text!
)

vector_store_earnings = MongoDBAtlasVectorSearch(
    embedding=embedding_model,
    collection=earnings,
    index_name=earnings_index,
    text_key="content"  # <--- ✅ very important to match the key storing text!
)


# ✅ Retriever
retriever_fillings = vector_store_fillings.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

retriever_earnings = vector_store_earnings.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# ✅ Prompt template
prompt_template = """
You are a professional financial analyst assistant. Your task is to provide a detailed, factual, and well-structured answer based strictly on the information contained in the <context> section below.

If the context does not contain enough information to answer the question, clearly state that the answer is not available within the provided context. Do not assume, hallucinate, or fabricate data.

<fillings_context>
{fillings_context}
</fillings_context>

<earnings_context>
{earnings_context}
</earnings_context>

Question:
{question}

Instructions:
- Base your answer **only** on the above context.
- If the answer includes figures, financial terms, or events, explain them briefly and clearly.
- Structure the response into logical paragraphs.
- When applicable, cite sections or filing types (e.g., "as reported in the 10-K filing").
- If helpful, include bullet points for clarity.

Answer:
"""

prompt = PromptTemplate.from_template(prompt_template)

# ✅ LLM (OpenAI)
llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

# ✅ Output parser
output_parser = StrOutputParser()

# ✅ Formatting
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ✅ RAG pipeline
# retrieval_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | output_parser
# )

# # === ✅ Query execution ===
# query = "what's the revenue of apple?"
# response = retrieval_chain.invoke(query)
# print("Answer:\n", response)


# === Run query + print context ===
query = "what are the risks apple is currently facing?"



# Step 1: Retrieve documents
docs_fillings = retriever_fillings.get_relevant_documents(query)
docs_earnings = retriever_earnings.get_relevant_documents(query)

# 🔍 Print context
print("\n--- Retrieved Context of fillings ---\n")
for i, doc in enumerate(docs_fillings, 1):
    print(f"[Chunk {i}]")
    print(doc.page_content.strip())
    print("-" * 40)
    
print("\n--- Retrieved Context of earnings ---\n")
for i, doc in enumerate(docs_earnings, 1):
    print(f"[Chunk {i}]")
    print(doc.page_content.strip())
    print("-" * 40)

# Step 3: Format and pass to LLM
formatted_fillings_context = format_docs(docs_fillings)
formatted_earnings_context = format_docs(docs_earnings)
prompt_input = prompt.format(fillings_context=formatted_fillings_context,earnings_context=formatted_earnings_context, question=query)
response = llm.invoke(prompt_input)

# Step 4: Output
print("\n--- LLM Answer ---\n")
print(response)