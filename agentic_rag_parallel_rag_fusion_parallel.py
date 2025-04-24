# Imports
from datetime import datetime
import random
from typing import Annotated
from typing_extensions import TypedDict
from fastapi import FastAPI, Request
import uvicorn
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.messages import ToolMessage,AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
from langchain.pydantic_v1 import BaseModel as LCBaseModel, Field
from typing import List, Optional
from datetime import datetime,timedelta
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from dotenv import load_dotenv
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


import os
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

# Define State
class ParallelState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tools_output: dict
    tools_to_run: list
    waiting_for_tools: bool

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: ParallelState, config: RunnableConfig):
        configuration = config.get("configurable", {})
        user_id = configuration.get("user_id", None)
        state = {**state, "user_info": user_id}

        response = self.runnable.invoke(state)
        new_state = {
            "messages": state["messages"] + [response],
            "tools_output": {},
            "tools_to_run": [],
            "waiting_for_tools": False
        }

        if hasattr(response, 'tool_calls'):
            tool_calls = response.tool_calls
            tool_specs = []
            for call in tool_calls:
                tool_specs.append({
                "id": call["id"],
                "tool_name": call["name"],
                "args": call["args"]
            })

            new_state["tools_to_run"] = tool_specs
            new_state["waiting_for_tools"] = True


        return new_state

    
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
     "You are a financial assistant that retrieves:\n"
     "- SEC filings (10-K, 10-Q, 8-K),\n"
     "- Earnings transcripts,\n"
     "- News articles from verified sources,\n"
     "using metadata filters and semantic search.\n\n"
     "When answering the user query:\n"
     "• Only mention **actual dates** or **time periods (e.g., Q2 2025)** if they are explicitly present in the document metadata or content.\n"
     "• Never fabricate or guess a date — if a document has no clear date, describe it as 'undated' or 'no date provided'.\n"
     "• Always include the **exact SEC filing type** (e.g., '10-Q'),\n"
     "• Always **name the news source** (e.g., 'CNBC', 'Reuters') in the response\n"
     "• Clearly cite whether the insight came from a **filing**, **earnings call**, or **news article**\n"
     "• If available, include the **retrieved publication date** from metadata (e.g., 'Published on April 22, 2025').\n\n"
     "Format your response in a professional tone, clearly distinguishing each source of information."
    ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())


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

def generate_query_variants(query: str, llm) -> List[str]:
    prompt = f"""
    You are helping to improve search relevance in a financial assistant system.

    Given the user query: "{query}", generate 3 rephrased versions that:

    1. Ask the same question in different ways
    2. Vary the phrasing enough to trigger different semantic matches
    3. Are optimized for retrieving financial documents

    List each version on a new line, without numbering or extra formatting.
    """
    response = llm.invoke(prompt)
    raw_output = response.content.strip()
    return [line.strip("-• ").strip() for line in raw_output.split("\n") if line.strip()]


# def rag_fusion_search(vector_store, queries, k=5, filter=None):
#     results = []
#     for q in queries:
#         results.extend(vector_store.similarity_search_with_score(q, k=k, pre_filter=filter or {}))
#     # Deduplicate and sort
#     seen = set()
#     fused = []
#     print("&&&&&&&&&&&&&&&&&&&&&&&&&&&")
#     print(vector_store,len(results))
#     print("&&&&&&&&&&&&&&&&&&&&&&&&&&&")
#     for doc, score in sorted(results, key=lambda x: x[1], reverse=True):
#         if doc.page_content not in seen:
#             fused.append((doc, score))
#             seen.add(doc.page_content)
#         if len(fused) >= k:
#             break
#     return fused

def rag_fusion_search_parallel(vector_store, queries, k=5, filter=None):
    results = []

    def search_one_query(q):
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f"[{now}] [Thread {threading.get_ident()}] Running query: {q}")
        return vector_store.similarity_search_with_score(q, k=k, pre_filter=filter or {})

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(search_one_query, q) for q in queries]
        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except Exception as e:
                print(f"Error during query execution: {e}")

    # Deduplicate and sort
    seen = set()
    fused = []
    for doc, score in sorted(results, key=lambda x: x[1], reverse=True):
        if doc.page_content not in seen:
            fused.append((doc, score))
            seen.add(doc.page_content)
        if len(fused) >= k:
            break

    return fused

def fetch_all_sources(vector_store_tasks: dict, queries: list, k: int = 5) -> dict:
    """
    Executes RAG Fusion search across multiple vector stores in parallel.

    Args:
        vector_store_tasks (dict): A dictionary with structure:
            {
                "filings": (vector_store_instance, filter_dict),
                "earnings": (vector_store_instance, filter_dict),
                ...
            }
        queries (list): A list of reformulated queries for RAG Fusion.
        k (int): Number of unique top-k results to return per store.

    Returns:
        dict: A dictionary mapping each store name to its fused search results.
    """
    with ThreadPoolExecutor() as executor:
        futures = {
            name: executor.submit(rag_fusion_search_parallel, store, queries, k, _filter)
            for name, (store, _filter) in vector_store_tasks.items()
        }
        return {name: future.result() for name, future in futures.items()}

# Define the beginner-friendly tools
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
    query_variants = generate_query_variants(desc, llm)
    print(query_variants)
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
    vector_store_tasks = {
    "filings": (vector_store_fillings, filter_filings),
    "earnings": (vector_store_earnings, filter_earnings),
}
    results = fetch_all_sources(vector_store_tasks, queries=query_variants)
    filings_results = results["filings"]
    earnings_results = results["earnings"]
    
    # filings_results = rag_fusion_search_parallel(vector_store_fillings, query_variants, k=5, filter=filter_filings)
    # earnings_results = rag_fusion_search_parallel(vector_store_earnings, query_variants, k=5, filter=filter_earnings)

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
    query_variants = generate_query_variants(desc, llm)
    print(query_variants)
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

    results = rag_fusion_search_parallel(vector_store_news, query_variants, k=5, filter=filter)

    print(filter)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ news")
    print(results)

    return format_docs(results)



def execute_tools_in_parallel(state: ParallelState) -> ParallelState:
    new_state = state.copy()
    tools_to_run = state.get("tools_to_run", [])
    results = {}
    tool_messages = []

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                lambda t=tool: next(tool_obj for tool_obj in tools_to_use if tool_obj.name == t["tool_name"]).invoke(t["args"])
            ): tool
            for tool in tools_to_run
        }

        for future in as_completed(futures):
            tool = futures[future]
            tool_name = tool["tool_name"]
            tool_id = tool["id"]

            try:
                result = future.result()
                results[tool_name] = result
                tool_messages.append(ToolMessage(tool_call_id=tool_id, content=result))
            except Exception as e:
                error = f"Error: {str(e)}"
                results[tool_name] = error
                tool_messages.append(ToolMessage(tool_call_id=tool_id, content=error))

    new_state["tools_output"] = results
    new_state["tools_to_run"] = []
    new_state["waiting_for_tools"] = False
    new_state["messages"].extend(tool_messages)
    return new_state



tools_to_use = [get_company_disclosures, get_news_articles]


# Runnables
assistant_runnable = primary_assistant_prompt | llm.bind_tools(tools_to_use)

# Helper functions
# def handle_tool_error(error):
#     return f"An error occurred while using the tool: {str(error)}"

# def create_tool_node_with_fallback(tools: list) -> dict:
#     return ToolNode(tools).with_fallbacks(
#         [RunnableLambda(lambda x: tool.invoke(x))], exception_key="error"
#     )

def tool_routing(state: ParallelState):
    print("===========================")
    print("Routing: waiting_for_tools =", state["waiting_for_tools"])
    print("===========================")

    if state["waiting_for_tools"]:
        return "tools"
    elif state["tools_output"]:  # tools are done → go back to assistant
        return "assistant"
    else:
        return "end"  # or END




def build_parallel_graph():
    builder = StateGraph(ParallelState)

    builder.add_node("assistant", Assistant(assistant_runnable))
    builder.add_node("tools", execute_tools_in_parallel)

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tool_routing, {"tools": "tools", "assistant": END})
    builder.add_conditional_edges("tools", tool_routing, {"tools": "tools", "assistant": "assistant", "end": END})

    return builder



def _print_event(event: dict, _printed: set, max_length=None):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in:", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if max_length and len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

            
    
def main(user_query: str) -> str:
    builder = build_parallel_graph()
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    import uuid
    thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            "user_id": "cplog",
            "thread_id": thread_id,
        }
    }

    _printed = set()
    final_content = "No response generated."

    events = graph.stream(
        {"messages": [("user", user_query)]}, config, stream_mode="values"
    )

    for event in events:
        _print_event(event, _printed)
        if "messages" in event:
            last_msg = event["messages"][-1]
            if hasattr(last_msg, "content"):
                final_content = last_msg.content

    return final_content

    
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_handler(req: QueryRequest):
    response = main(req.query)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run("agentic_rag_parallel_rag_fusion_parallel:app", host="0.0.0.0", port=8000, reload=True) 