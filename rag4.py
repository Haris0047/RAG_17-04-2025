# Imports
from datetime import datetime
import random
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.messages import ToolMessage,AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain.pydantic_v1 import BaseModel as LCBaseModel, Field
from typing import List, Optional
from datetime import datetime,timedelta
from pymongo import MongoClient
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_mongodb import MongoDBAtlasVectorSearch
import json
import concurrent.futures
from dotenv import load_dotenv
import os
import uvicorn

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
    # filings_results = vector_store_fillings.similarity_search_with_score(desc, k=5, pre_filter=filter_filings)
    # earnings_results = vector_store_earnings.similarity_search_with_score(desc, k=5, pre_filter=filter_earnings)
    
    filings_results = rag_fusion_search(vector_store_fillings, query_variants, k=5, filter=filter_filings)
    earnings_results = rag_fusion_search(vector_store_earnings, query_variants, k=5, filter=filter_earnings)

    # Merge and sort by similarity score
    combined = filings_results + earnings_results
    combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)[:5]

    print("#################################################")
    print(query_variants)
    print("#################################################")

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
    desc = kwargs.get("desc") or ""
    query_variants = generate_query_variants(desc, llm)

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

    # results = vector_store_news.similarity_search_with_score(
    #     query=desc or "", k=5, pre_filter=filter
    # )
    results = rag_fusion_search(vector_store_news, query_variants, k=5, filter=filter)

    print("#################################################")
    print(query_variants)
    print("#################################################")
    
    print(filter)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ news")
    print(results)
    print('wqwqwqwqwq')

    return format_docs(results)

def generate_query_variants(query: str, llm) -> List[str]:
    prompt = f"Rephrase the following query in 3 different ways to retrieve relevant documents:\n\nQuery: {query}"
    variants = llm.invoke(prompt).content.strip().split("\n")
    return [v.strip("-• ") for v in variants if v]

def rag_fusion_search(vector_store, queries, k=5, filter=None):
    results = []
    for q in queries:
        results.extend(vector_store.similarity_search_with_score(q, k=k, pre_filter=filter or {}))
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

def execute_tools_in_parallel(state: ParallelState) -> ParallelState:
    new_state = state.copy()
    tools_to_run = state.get("tools_to_run", [])
    results = {}
    tool_messages = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                lambda t=tool: next(tool_obj for tool_obj in tools_to_use if tool_obj.name == t["tool_name"]).invoke(t["args"])
            ): tool
            for tool in tools_to_run
        }

        for future in concurrent.futures.as_completed(futures):
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

# Prompts
primary_assistant_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a financial assistant that retrieves multiple SEC filings (10-K,10-Q and 8-K), earning data and latest news of stocks using metadata filters. Answer user query citing to the given filings, earnings or news with proper reference."
              "\n\nCurrent user:\n\n{user_info}\n"
              "\nCurrent time: {time}."),
    ("placeholder", "{messages}"),
]).partial(time=datetime.now())

# Runnables
assistant_runnable = primary_assistant_prompt | llm.bind_tools(tools_to_use)


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

            
    
def main():
    builder = build_parallel_graph()
    # The checkpointer lets the graph persist its state
    # this is a complete memory for the entire graph.
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    # print(graph.get_graph().draw_ascii())
    # Add any additional execution code here
    import uuid
    thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            # fetch the user's id
            "user_id": "cplog",
            # Checkpoints are accessed by thread_id
            "thread_id": thread_id,
        }
    }
    _printed = set()
    events = graph.stream(
        {"messages": ("user", 'what are the risks apple has been facing recently?')}, config, stream_mode="values"
    )
    # what are the risks apple has been facing recently according to 10-Q?
    for event in events:
        _print_event(event, _printed)
        
    return event['messages'][-1].content
    
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_handler(req: QueryRequest):
    result = main()
    return {"response": result}

if __name__ == "__main__":
    uvicorn.run("rag3:app", host="0.0.0.0", port=8000, reload=True) 