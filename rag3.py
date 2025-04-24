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

    
class NewsQueryInput(LCBaseModel):
    ticker: Optional[str] = Field(description="The stock ticker symbol, e.g., AAPL")
    publisher: Optional[str] = Field(description="The publisher of the news, e.g., CNBC, Parth Sanghvi")
    date_from: Optional[str] = Field(description="Start date (YYYY-MM-DD)")
    date_to: Optional[str] = Field(description="End date (YYYY-MM-DD)")
    desc: Optional[str] = Field(description="Search description or keyword")

class CompanyDisclosureQueryInput(LCBaseModel):
    ticker: Optional[str] = Field(description="The stock ticker symbol, e.g., AAPL")
    filling_type: Optional[List[str]] = Field(description="List of filing types like ['10-K', '10-Q']")
    quarter: Optional[str] = Field(description="The earnings quarter, e.g., Q1, Q2")
    date_from: Optional[str] = Field(description="Start date (YYYY-MM-DD)")
    date_to: Optional[str] = Field(description="End date (YYYY-MM-DD)")
    last_n_years: Optional[int] = Field(description="How many years back to search")
    desc: Optional[str] = Field(description="Search description or keyword")


@tool(args_schema=CompanyDisclosureQueryInput)
def get_company_disclosures(**kwargs) -> str:
    """
    Retrieves both SEC filings (10-K, 10-Q, etc.) and earnings call transcripts using metadata filters and semantic relevance.
    """
    filter_common = {}
    ticker = kwargs.get("ticker")
    filling_type = kwargs.get("filling_type")
    quarter = kwargs.get("quarter")
    date_from = kwargs.get("date_from")
    date_to = kwargs.get("date_to")
    last_n_years = kwargs.get("last_n_years")
    desc = kwargs.get("desc") or ""

    # Build date filter
    now = datetime.now()
    if last_n_years:
        start = datetime(now.year - last_n_years, 1, 1)
        filter_common["date"] = {"$gte": start, "$lte": now}
    elif date_from and date_to:
        filter_common["date"] = {"$gte": datetime.fromisoformat(date_from), "$lte": datetime.fromisoformat(date_to)}
    elif date_from:
        filter_common["date"] = {"$gte": datetime.fromisoformat(date_from)}
    elif date_to:
        filter_common["date"] = {"$lte": datetime.fromisoformat(date_to)}

    if ticker:
        filter_common["ticker"] = {"$eq": ticker.upper()}

    # Filters specific to filings
    filter_filings = dict(filter_common)
    if filling_type:
        filter_filings["filling_type"] = {"$in": [ft.upper() for ft in filling_type]}

    # Filters specific to earnings
    filter_earnings = dict(filter_common)
    if quarter:
        filter_earnings["quarter"] = {"$eq": quarter.upper()}

    # Vector search on both
    filings_results = vector_store_fillings.similarity_search_with_score(desc, k=5, pre_filter=filter_filings)
    earnings_results = vector_store_earnings.similarity_search_with_score(desc, k=5, pre_filter=filter_earnings)

    # Merge and sort by similarity score
    combined = filings_results + earnings_results
    combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)[:5]

    print("Filter (Common):", filter_common)
    print("Combined Disclosure Results >>>>", combined_sorted)

    return format_docs(combined_sorted)

    

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
    ("system", 
     "You are a financial assistant that retrieves:\n"
     "- SEC filings (10-K, 10-Q, 8-K),\n"
     "- Earnings transcripts,\n"
     "- News articles from verified sources,\n"
     "using metadata filters and semantic search.\n\n"
     "When answering the user query:\n"
     "• Always mention the **exact SEC filing type** (e.g., '10-Q')\n"
     "• Always **name the news source** (e.g., 'CNBC', 'Reuters') in the response\n"
     "• If possible, include **filing dates or quarters** (e.g., 'Q4 2023')\n"
     "• Clearly cite whether the insight came from a **filing**, **earnings call**, or **news article**\n\n"
     "Format your response in a professional tone, clearly distinguishing each source of information.\n"
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


agent = create_tool_calling_agent(llm, [get_company_disclosures, get_news_articles], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[get_company_disclosures, get_news_articles])


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