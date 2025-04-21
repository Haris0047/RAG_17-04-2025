import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.pydantic_v1 import BaseModel, Field
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.schema.output_parser import StrOutputParser
from datetime import datetime, timedelta

# ========== Load environment ==========
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# ========== MongoDB setup ==========
client = MongoClient(MONGO_URI)
db = client["qualitative"]
fillings_collection = db["fillings"]
earnings_collection = db["earnings"]

# ========== Embeddings & Vector Store ==========
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

vector_store_fillings = MongoDBAtlasVectorSearch(
    embedding=embedding_model,
    collection=fillings_collection,
    index_name="fillings",
    text_key="content"
)

vector_store_earnings = MongoDBAtlasVectorSearch(
    embedding=embedding_model,
    collection=earnings_collection,
    index_name="earnings",
    text_key="content"
)

# ========== LLM ==========
llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)

# ========== Formatting Function ==========
def format_docs(docs_with_scores):
    return "\n\n".join(doc.page_content for doc in docs_with_scores)

# ========== Tool Input Schema ==========
class FilingQueryInput(BaseModel):
    ticker: Optional[str] = Field(description="The stock ticker, e.g., AAPL")
    filling_type: Optional[str] = Field(description="Filing type like 10-K, 10-Q, 8-K")
    date_from: Optional[str] = Field(description="Start of date range (YYYY-MM-DD)")
    date_to: Optional[str] = Field(description="End of date range (YYYY-MM-DD)")
    last_n_years: Optional[int] = Field(description="Number of years to look back from today")
    desc: Optional[str] = Field(description="Main search description or question")
    
class EarningsQueryInput(BaseModel):
    ticker: Optional[str] = Field(description="The stock ticker symbol, e.g., AAPL")
    quarter: Optional[str] = Field(description="The earnings quarter, e.g., Q1, Q2, Q3, Q4")
    date_from: Optional[str] = Field(description="Start date (YYYY-MM-DD)")
    date_to: Optional[str] = Field(description="End date (YYYY-MM-DD)")
    last_n_years: Optional[int] = Field(description="How many years back to search")
    desc: Optional[str] = Field(description="Search description or keyword")



# ========== Tool Function ==========
@tool(args_schema=FilingQueryInput)
def get_filing_documents(
    ticker: Optional[str] = None,
    filling_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    last_n_years: Optional[int] = None,
    desc: Optional[str] = None
) -> str:
    """
    Retrieves SEC filings based on metadata filters: ticker, filing type, and flexible date ranges.
    Can filter between two dates or use 'last N years' from today.
    """
    filter = {}

    if ticker:
        filter["ticker"] = {"$eq": ticker.upper()}
    if filling_type:
        filter["filling_type"] = {"$eq": filling_type.upper()}

    # Flexible date filtering
    if last_n_years:
        now = datetime.now()
        start = datetime(now.year - last_n_years, 1, 1)
        end = now
        filter["date"] = {"$gte": start, "$lte": end}
    elif date_from and date_to:
        filter["date"] = {
            "$gte": datetime.fromisoformat(date_from),
            "$lte": datetime.fromisoformat(date_to)
        }
    elif date_from:
        filter["date"] = {"$gte": datetime.fromisoformat(date_from)}
    elif date_to:
        filter["date"] = {"$lte": datetime.fromisoformat(date_to)}

    results = vector_store_fillings.similarity_search_with_score(
        query=desc or "",
        k=5,
        pre_filter=filter
    )
    
    print(filter)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(results)

    return format_docs([doc for doc, _ in results])

from langchain_core.tools import tool
from datetime import datetime

@tool(args_schema=EarningsQueryInput)
def get_earnings_transcripts(
    ticker: Optional[str] = None,
    quarter: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    last_n_years: Optional[int] = None,
    desc: Optional[str] = None
) -> str:
    """
    Retrieves earnings call transcripts based on ticker, quarter, and date filters.
    Allows filtering by last N years or a specific date range.
    """
    filter = {}

    if ticker:
        filter["ticker"] = {"$eq": ticker.upper()}
    if quarter:
        filter["quarter"] = {"$eq": quarter.upper()}

    if last_n_years:
        now = datetime.now()
        start = datetime(now.year - last_n_years, 1, 1)
        filter["date"] = {"$gte": start, "$lte": now}
    elif date_from and date_to:
        filter["date"] = {
            "$gte": datetime.fromisoformat(date_from),
            "$lte": datetime.fromisoformat(date_to)
        }
    elif date_from:
        filter["date"] = {"$gte": datetime.fromisoformat(date_from)}
    elif date_to:
        filter["date"] = {"$lte": datetime.fromisoformat(date_to)}

    results = vector_store_earnings.similarity_search_with_score(
        query=desc or "",
        k=5,
        pre_filter=filter
    )
    
    print(filter)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(results)


    return format_docs([doc for doc, _ in results])


# ========== Prompt + Agent ==========
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a financial assistant that retrieves SEC filing and earning data using metadata filters."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, [get_filing_documents,get_earnings_transcripts], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[get_filing_documents,get_earnings_transcripts])

# ========== Run an example ==========
query = {
    "input": "how apple is performing in the last 2 years. Explain with numbers like eps.",
    "chat_history": [],
    "agent_scratchpad": ""
}

result = agent_executor.invoke(query)
print("🔍 Answer:\n", result["output"])
