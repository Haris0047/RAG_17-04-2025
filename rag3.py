from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.schema.output_parser import StrOutputParser
from datetime import datetime,timedelta
from langchain.pydantic_v1 import BaseModel as LCBaseModel, Field
import uvicorn

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# MongoDB setup
client = MongoClient(MONGO_URI)
db = client["qualitative"]
fillings_collection = db["fillings"]
earnings_collection = db["earnings"]
news_collection = db["news"]

# Embedding & Vector Store
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

vector_store_news = MongoDBAtlasVectorSearch(
    embedding=embedding_model,
    collection=news_collection,
    index_name="news",
    text_key="content"
)

# Chat LLM
llm = ChatOpenAI(model_name="gpt-4.1", temperature=0, openai_api_key=OPENAI_API_KEY)

def format_docs(docs_with_scores):
    return "\n\n".join(doc.page_content for doc, _ in docs_with_scores)

class FilingQueryInput(LCBaseModel):
    ticker: Optional[str] = Field(description="The stock ticker, e.g., AAPL")
    filling_type: Optional[List[str]] = Field(description="List of filing types like ['10-K', '10-Q', '8-K']")
    date_from: Optional[str] = Field(description="Start of date range (YYYY-MM-DD)")
    date_to: Optional[str] = Field(description="End of date range (YYYY-MM-DD)")
    last_n_years: Optional[int] = Field(description="Number of years to look back from today")
    desc: Optional[str] = Field(description="Main search description or question")

class EarningsQueryInput(LCBaseModel):
    ticker: Optional[str] = Field(description="The stock ticker symbol, e.g., AAPL")
    quarter: Optional[str] = Field(description="The earnings quarter, e.g., Q1, Q2, Q3, Q4")
    date_from: Optional[str] = Field(description="Start date (YYYY-MM-DD)")
    date_to: Optional[str] = Field(description="End date (YYYY-MM-DD)")
    last_n_years: Optional[int] = Field(description="How many years back to search")
    desc: Optional[str] = Field(description="Search description or keyword")
    
class NewsQueryInput(LCBaseModel):
    ticker: Optional[str] = Field(description="The stock ticker symbol, e.g., AAPL")
    publisher: Optional[str] = Field(description="The publisher of the news, e.g., CNBC, Parth Sanghvi")
    date_from: Optional[str] = Field(description="Start date (YYYY-MM-DD)")
    date_to: Optional[str] = Field(description="End date (YYYY-MM-DD)")
    desc: Optional[str] = Field(description="Search description or keyword")


@tool(args_schema=FilingQueryInput)
def get_filing_documents(**kwargs) -> str:
    """
    Retrieves SEC filings based on metadata filters: ticker, filing type, and flexible date ranges.
    Can filter between two dates or use 'last N years' from today.
    """
    filter = {}
    ticker = kwargs.get("ticker")
    filling_type = kwargs.get("filling_type")
    date_from = kwargs.get("date_from")
    date_to = kwargs.get("date_to")
    last_n_years = kwargs.get("last_n_years")
    desc = kwargs.get("desc")

    if ticker:
        filter["ticker"] = {"$eq": ticker.upper()}
    if filling_type:
        filter["filling_type"] = {"$in": [ft.upper() for ft in filling_type]}

    if last_n_years:
        now = datetime.now()
        start = datetime(now.year - last_n_years, 1, 1)
        end = now
        filter["date"] = {"$gte": start, "$lte": end}
    elif date_from and date_to:
        filter["date"] = {"$gte": datetime.fromisoformat(date_from), "$lte": datetime.fromisoformat(date_to)}
    elif date_from:
        filter["date"] = {"$gte": datetime.fromisoformat(date_from)}
    elif date_to:
        filter["date"] = {"$lte": datetime.fromisoformat(date_to)}

    results = vector_store_fillings.similarity_search_with_score(query=desc or "", k=5, pre_filter=filter)
    
    print(filter)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ filings")
    print(results)
    
    return format_docs(results)

@tool(args_schema=EarningsQueryInput)
def get_earnings_transcripts(**kwargs) -> str:
    """
    Retrieves earnings call transcripts based on ticker, quarter, and date filters.
    Allows filtering by last N years or a specific date range.
    """
    filter = {}
    ticker = kwargs.get("ticker")
    quarter = kwargs.get("quarter")
    date_from = kwargs.get("date_from")
    date_to = kwargs.get("date_to")
    last_n_years = kwargs.get("last_n_years")
    desc = kwargs.get("desc")

    if ticker:
        filter["ticker"] = {"$eq": ticker.upper()}
    if quarter:
        filter["quarter"] = {"$eq": quarter.upper()}

    if last_n_years:
        now = datetime.now()
        start = datetime(now.year - last_n_years, 1, 1)
        filter["date"] = {"$gte": start, "$lte": now}
    elif date_from and date_to:
        filter["date"] = {"$gte": datetime.fromisoformat(date_from), "$lte": datetime.fromisoformat(date_to)}
    elif date_from:
        filter["date"] = {"$gte": datetime.fromisoformat(date_from)}
    elif date_to:
        filter["date"] = {"$lte": datetime.fromisoformat(date_to)}

    results = vector_store_earnings.similarity_search_with_score(query=desc or "", k=5, pre_filter=filter)
    
    print(filter)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ earnings")
    print(results)
    
    return format_docs(results)

@tool(args_schema=NewsQueryInput)
def get_news_articles(**kwargs) -> str:
    """
    Retrieves news articles based on ticker, publisher, and date filters.
    Defaults to the last 7 days if no date is specified.
    """
    filter = {}
    ticker = kwargs.get("ticker")
    publisher = kwargs.get("publisher")
    date_from = kwargs.get("date_from")
    date_to = kwargs.get("date_to")
    desc = kwargs.get("desc")

    if ticker:
        filter["ticker"] = {"$eq": ticker.upper()}
    if publisher:
        filter["publisher"] = {"$regex": publisher, "$options": "i"}

    # Handle date filtering
    now = datetime.now()
    if date_from and date_to:
        filter["date"] = {
            "$gte": datetime.fromisoformat(date_from),
            "$lte": datetime.fromisoformat(date_to)
        }
    elif date_from:
        filter["date"] = {"$gte": datetime.fromisoformat(date_from)}
    elif date_to:
        filter["date"] = {"$lte": datetime.fromisoformat(date_to)}
    else:
        # ⏱️ Default to last 7 days
        week_ago = now - timedelta(days=7)
        filter["date"] = {"$gte": week_ago, "$lte": now}

    results = vector_store_news.similarity_search_with_score(
        query=desc or "", k=5, pre_filter=filter
    )

    print(filter)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ news")
    print(results)

    return format_docs(results)


# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a financial assistant that retrieves multiple SEC filings (10-K,10-Q and 8-K), earning data and latest news of stocks using metadata filters. Answer user query citing to the given filings, earnings or news."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, [get_filing_documents, get_earnings_transcripts,get_news_articles], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[get_filing_documents, get_earnings_transcripts,get_news_articles])

# FastAPI Setup
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_handler(req: QueryRequest):
    result = agent_executor.invoke({
        "input": req.query,
        "chat_history": [],
        "agent_scratchpad": ""
    })
    return {"response": result["output"]}

if __name__ == "__main__":
    uvicorn.run("rag3:app", host="0.0.0.0", port=8000, reload=True) 