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
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range,MatchText
from qdrant_client.http.models import DatetimeRange
from qdrant_client.http.models import SearchRequest
from dotenv import load_dotenv
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import re
from langchain.schema import Document

import os
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# MongoDB setup
qdrant_client = QdrantClient(url="http://localhost:6333")

# Embedding & Vector Store
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

vector_store_fillings = QdrantVectorStore(
    client=qdrant_client,
    collection_name="fillings",
    embeddings=embedding_model,
    content_payload_key="content",    # for content lookup
)

vector_store_earnings = QdrantVectorStore(
    client=qdrant_client,
    collection_name="earnings",
    embeddings=embedding_model,
    content_payload_key="content",
)

vector_store_news = QdrantVectorStore(
    client=qdrant_client,
    collection_name="news",
    embeddings=embedding_model,
    content_payload_key="content",
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
     "You are a precise financial research assistant that retrieves and analyzes:\n"
     "- SEC filings (10-K annual reports, 10-Q quarterly reports, 8-K current reports)\n"
     "- Earnings call transcripts\n"
     "- Financial news from verified sources\n\n"
     
     "QUERY GENERATION GUIDELINES:\n"
     "1. When a user inquires about a company or financial topic:\n"
     "   - Extract the primary entity (company name, ticker symbol)\n"
     "   - Identify the specific information requested (financials, performance metrics, executive changes, etc.)\n"
     "   - Determine the relevant time period (specific quarter, year, date range)\n"
     "2. Generate specific search queries that include:\n"
     "   - Both company name AND ticker symbol when known (e.g., 'Apple Inc. AAPL')\n"
     "   - Specific document types needed (e.g., '10-K 2024', 'Q2 earnings transcript')\n"
     "   - Explicit date ranges when applicable (e.g., 'between January 2025 and April 2025')\n"
     "   - Specific financial metrics or events mentioned (e.g., 'revenue growth', 'CEO change')\n"
     "3. Use precise Boolean operators (AND, OR, NOT) to refine results\n"
     "4. For each query, clearly specify the appropriate database:\n"
     "   - SEC EDGAR database for regulatory filings\n"
     "   - Earnings call transcript repository\n"
     "   - Verified financial news sources\n\n"
     
     "RESPONSE FORMATTING REQUIREMENTS:\n"
     "• Begin with a concise summary of key findings (2-3 sentences)\n"
     "• Organize information by source type (SEC Filings, Earnings Calls, News)\n"
     "• Include ONLY dates that appear in the retrieved documents:\n"
     "  - For SEC filings: Include both filing date and period covered (e.g., 'Filed on March 15, 2025, covering Q1 2025')\n"
     "  - For earnings calls: Include the exact call date (e.g., 'Earnings call from April 28, 2025')\n"
     "  - For news: Include the publication date (e.g., 'Published on April 30, 2025')\n"
     "• NEVER present estimated or approximated dates - if a document lacks a clear date, explicitly state 'date not specified'\n"
     "• For SEC filings: Always specify the exact filing type (10-K, 10-Q, 8-K) and the key sections referenced\n"
     "• For news: Always include the specific source name (e.g., Bloomberg, CNBC, Reuters) and author when available\n"
     "• Use direct quotes sparingly and only when particularly significant\n"
     "• Present financial figures with appropriate context (YoY growth, industry comparison)\n"
     "• Include a 'Data Limitations' section noting any gaps in the retrieved information\n\n"
     
     "ANALYSIS GUIDELINES:\n"
     "• Distinguish clearly between factual information and analytical insights\n"
     "• Highlight significant discrepancies between different information sources\n"
     "• Note important trends across multiple reporting periods when visible\n"
     "• Flag potential regulatory concerns or material events that may impact financial outlook\n"
     "• Avoid speculative predictions unless explicitly requested\n\n"
     
     "Present all information in a professional, objective tone with logical organization and clear section headers."
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
    
    IMPORTANT: Return ONLY the 3 rephrased queries, one per line, with no numbering, prefixes, or explanations.
    For example, if the original query is "What are treasury bonds?", respond with exactly:
    How do treasury bonds work?
    Explain treasury bonds and their features
    Treasury bonds definition and characteristics
    """
    
    response = llm.invoke(prompt)
    raw_output = response.content.strip()
    
    # Filter out empty lines and common formatting patterns
    variants = [line.strip() for line in raw_output.split('\n') if line.strip()]
    
    # Remove any remaining bullet points, numbers, or other prefixes
    cleaned_variants = []
    for line in variants:
        # Remove common prefixes like numbers, bullets, etc.
        cleaned_line = re.sub(r'^[\d\-\.\•\*\s]+', '', line).strip()
        if cleaned_line:
            cleaned_variants.append(cleaned_line)
    
    # Return only up to the first 3 valid variants
    return cleaned_variants[:3]


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

# Raw cosine similarity from individual queries, sorted globally, deduplicated, and top-k taken.
# def rag_fusion_search_parallel(vector_store, queries, k=5, filter=None):
#     results = []

#     def search_one_query(q):
#         now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
#         print(f"[{now}] [Thread {threading.get_ident()}] Running query: {q}")
#         return vector_store.similarity_search_with_score(q, k=k, pre_filter=filter or {})

#     with ThreadPoolExecutor() as executor:
#         futures = [executor.submit(search_one_query, q) for q in queries]
#         for future in as_completed(futures):
#             try:
#                 results.extend(future.result())
#             except Exception as e:
#                 print(f"Error during query execution: {e}")

#     # Deduplicate and sort
#     seen = set()
#     fused = []
#     for doc, score in sorted(results, key=lambda x: x[1], reverse=True):
#         if doc.page_content not in seen:
#             fused.append((doc, score))
#             seen.add(doc.page_content)
#         if len(fused) >= k:
#             break

#     return fused



def rag_fusion_rrf(vector_store, queries, k=5, filter=None, rrf_k=60):
    # 1) Underlying client & collection
    client          = vector_store.client
    collection_name = vector_store.collection_name

    # 2) Embed all queries
    embeddings = [vector_store.embeddings.embed_query(q) for q in queries]

    # 3) Build batch search requests
    requests = [
        SearchRequest(
            vector=vec,
            limit=k,
            filter=filter,
            with_payload=True,
            with_vector=False
        )
        for vec in embeddings
    ]

    # 4) Execute batch search
    batch_results = client.search_batch(
        collection_name=collection_name,
        requests=requests
    )  # -> List[List[ScoredPoint]]

    # 5) RRF fusion
    rrf_scores = defaultdict(float)
    doc_map    = {}

    for result_list in batch_results:
        for rank, pt in enumerate(result_list):
            # Extract the text and metadata from payload
            text     = pt.payload.get("content", "")
            metadata = {k: v for k, v in pt.payload.items() if k != "content"}
            metadata["_id"] = pt.id

            # Use text as the fusion key (or pt.id if you prefer)
            key = text  

            # Accumulate RRF score
            rrf_scores[key] += 1.0 / (rank + 1 + rrf_k)
            # Store a LangChain Document for that key
            doc_map[key] = Document(page_content=text, metadata=metadata)

    # 6) Pick top-k results
    top = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(doc_map[key], score) for key, score in top]

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
            name: executor.submit(rag_fusion_rrf, store, queries, k, _filter)
            for name, (store, _filter) in vector_store_tasks.items()
        }
        return {name: future.result() for name, future in futures.items()}

# Define the beginner-friendly tools
@tool(args_schema=CompanyDisclosureQueryInput)
def get_company_disclosures(**kwargs) -> str:
    """
    Retrieves both SEC filings (10-K, 10-Q, etc.) and earnings call transcripts using metadata filters and semantic relevance.
    """
    filter_common_conditions = []
    now = datetime.now()

    ticker = kwargs.get("ticker")
    filling_type = kwargs.get("filling_type")
    quarter = kwargs.get("quarter")
    date_from = kwargs.get("date_from")
    date_to = kwargs.get("date_to")
    last_n_years = kwargs.get("last_n_years")
    desc = kwargs.get("desc") or ""

    query_variants = generate_query_variants(desc, llm)
    print("Query Variants:", query_variants)

    # ── Date filters: use DatetimeRange for ISO-8601 / datetime filtering ─────────
    if last_n_years:
        start = datetime(now.year - last_n_years, 1, 1)
        filter_common_conditions.append(
            FieldCondition(
                key="date",
                range=DatetimeRange(
                    gte=start.isoformat() + "Z",
                    lte=now.isoformat() + "Z"
                )
            )
        )
    elif date_from or date_to:
        dr_kwargs = {}
        if date_from:
            dr_kwargs["gte"] = date_from
        if date_to:
            dr_kwargs["lte"] = date_to
        filter_common_conditions.append(
            FieldCondition(
                key="date",
                range=DatetimeRange(**dr_kwargs)
            )
        )

    # ── Ticker filter ───────────────────────────────────────────────────────────────
    if ticker:
        filter_common_conditions.append(
            FieldCondition(
                key="ticker",
                match=MatchValue(value=ticker.upper())
            )
        )

    # ── Filings‐specific filter ────────────────────────────────────────────────────
    filings_conditions = list(filter_common_conditions)
    if filling_type:
        filings_conditions.append(
            FieldCondition(
                key="filling_type",
                match=MatchAny(any=[ft.upper() for ft in filling_type])
            )
        )

    # ── Earnings‐specific filter ────────────────────────────────────────────────────
    earnings_conditions = list(filter_common_conditions)
    if quarter:
        earnings_conditions.append(
            FieldCondition(
                key="quarter",
                match=MatchValue(value=quarter.upper())
            )
        )

    # ── Build final Filter objects ────────────────────────────────────────────────
    filter_filings  = Filter(must=filings_conditions)
    filter_earnings = Filter(must=earnings_conditions)

    # Vector store tasks for parallel fetching
    vector_store_tasks = {
        "filings": (vector_store_fillings, filter_filings),
        "earnings": (vector_store_earnings, filter_earnings),
    }

    # Fetch all documents using RAG Fusion
    results = fetch_all_sources(vector_store_tasks, queries=query_variants)

    filings_results = results.get("filings", [])
    earnings_results = results.get("earnings", [])

    combined = filings_results + earnings_results
    combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)[:5]

    # Debug
    print("Applied Filters (Filings):", filings_conditions)
    print("Applied Filters (Earnings):", earnings_conditions)
    print("Combined Disclosure Results >>>>", combined_sorted)

    return format_docs(combined_sorted)



@tool(args_schema=NewsQueryInput)
def get_news_articles(**kwargs) -> str:
    """
    Retrieves news articles based on ticker, publisher, and date filters.
    Defaults to the last 7 days if no date is specified.
    """


    now = datetime.now()
    filter_conditions = []

    ticker = kwargs.get("ticker")
    publisher = kwargs.get("publisher")
    date_from = kwargs.get("date_from")
    date_to = kwargs.get("date_to")
    desc = kwargs.get("desc") or ""

    query_variants = generate_query_variants(desc, llm)

    print("Query Variants:", query_variants)

    # Build filters
    if ticker:
        filter_conditions.append(
            FieldCondition(
                key="ticker",
                match=MatchValue(value=ticker.upper())
            )
        )

    if publisher:
        filter_conditions.append(
            FieldCondition(
                key="publisher",
                match=MatchText(text=publisher)  # approximate regex-like match
            )
        )

    # Date Filters
    if date_from and date_to:
        filter_conditions.append(
            FieldCondition(
                key="date",
                range=DatetimeRange(
                    gte=date_from,   # e.g. "2025-04-22T15:47:48Z"
                    lte=date_to      # e.g. "2025-04-29T15:47:48Z"
                )
            )
        )
    elif date_from:
        filter_conditions.append(
            FieldCondition(
                key="date",
                range=DatetimeRange(gte=date_from)
            )
        )
    elif date_to:
        filter_conditions.append(
            FieldCondition(
                key="date",
                range=DatetimeRange(lte=date_to)
            )
        )
    else:
        week_ago = now - timedelta(days=7)
        filter_conditions.append(
            FieldCondition(
                key="date",
                range=DatetimeRange(
                    gte=week_ago.isoformat() + "Z",
                    lte=now.isoformat()      + "Z"
                )
            )
        )


    # Final filter
    filter_news = Filter(must=filter_conditions)

    # Perform retrieval
    results = rag_fusion_rrf(
        vector_store=vector_store_news,
        queries=query_variants,
        k=5,
        filter=filter_news
    )

    print("Filter Conditions Applied:", filter_conditions)
    print("Retrieved News Results >>>>", results)

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
    uvicorn.run("agentic_rag_parallel_rag_fusion_parallel_qdrant_2:app", host="0.0.0.0", port=8000, reload=True) 