import os
from typing import Annotated, List, Literal, TypedDict, Union, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
import json
import concurrent.futures
from copy import deepcopy
import time

# Define our state classes for both agent types
class ParallelAgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    tools_output: Dict[str, str]
    tools_to_run: List[Dict[str, Any]]
    waiting_for_tools: bool

class SequentialAgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    tools_output: List[str]
    next_steps: List[str]

# Tool 1: Web Search Tool
def web_search_tool(query: str) -> str:
    """Simulates a web search tool that finds information online."""
    # Add a small delay to simulate network latency
    time.sleep(1)
    if "weather" in query.lower():
        return "The current weather is sunny with a high of 75°F."
    elif "news" in query.lower():
        return "Top headlines: New AI breakthrough announced today. Stock markets close higher."
    elif "population" in query.lower():
        return "World population is approximately 8 billion as of 2024."
    else:
        return f"Search results for '{query}': Found several relevant websites with information."

# Tool 2: Calculator Tool
def calculator_tool(expression: str) -> str:
    """A tool that evaluates mathematical expressions."""
    # Add a small delay to simulate computation time
    time.sleep(0.5)
    try:
        # Safely evaluate the expression
        sanitized_expression = expression.replace('^', '**')
        result = eval(sanitized_expression, {"__builtins__": {}}, {"abs": abs, "round": round, "max": max, "min": min})
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"

#---------------------------
# PARALLEL AGENT FUNCTIONS
#---------------------------

# Function to execute a tool based on tool type and parameters
def execute_tool(tool_call: Dict[str, Any]) -> Dict[str, str]:
    tool_type = tool_call.get("tool_type")
    tool_params = tool_call.get("parameters", {})
    
    if tool_type == "web_search":
        query = tool_params.get("query", "")
        result = web_search_tool(query)
        return {"tool_type": "web_search", "result": result}
    
    elif tool_type == "calculator":
        expression = tool_params.get("expression", "")
        result = calculator_tool(expression)
        return {"tool_type": "calculator", "result": result}
    
    return {"tool_type": tool_type, "result": "Tool execution failed - unknown tool type"}

# Function to route based on the state for parallel agent
def parallel_route(state: ParallelAgentState) -> Literal["decide_tools", "execute_tools", "respond"]:
    if state["waiting_for_tools"]:
        return "execute_tools"
    elif not state["tools_to_run"] and not state["waiting_for_tools"]:
        return "respond"
    else:
        return "decide_tools"

# Function that decides what tools to use for parallel agent
def decide_tools(state: ParallelAgentState) -> ParallelAgentState:
    # Only process if we haven't decided tools yet
    if state["tools_to_run"] or state["waiting_for_tools"]:
        return state
    
    # Create a system prompt for the agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent assistant with access to two tools:
        1. Web Search Tool: Finds information online
        2. Calculator Tool: Performs mathematical calculations
        
        Based on the user's request, determine which tool(s) to use, if any.
        If you need to use tools, respond with a JSON object in the following format:
        
        ```json
        {{
            "thoughts": "Your reasoning about what tools to use",
            "tools": [
                {{
                    "tool_type": "web_search",
                    "parameters": {{
                        "query": "Your search query"
                    }}
                }},
                {{
                    "tool_type": "calculator",
                    "parameters": {{
                        "expression": "Your math expression"
                    }}
                }}
            ]
        }}
        ```
        
        You can include one or both tools in the "tools" array. 
        If you don't need to use any tools to answer the question, just respond directly.
        """),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Create an LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # Invoke the LLM
    response = llm.invoke(prompt.invoke({"messages": state["messages"]}))
    
    # Check if the response contains a tool call
    tools_to_run = []
    if "```json" in response.content:
        try:
            json_str = response.content.split("```json")[1].split("```")[0]
            action = json.loads(json_str)
            tools_to_run = action.get("tools", [])
            
            # If we have tools to run, set the waiting flag
            waiting_for_tools = len(tools_to_run) > 0
            
            return {
                "messages": state["messages"] + [response],
                "tools_output": state["tools_output"],
                "tools_to_run": tools_to_run,
                "waiting_for_tools": waiting_for_tools
            }
        except Exception as e:
            # If JSON parsing fails, assume no tool is needed
            return {
                "messages": state["messages"] + [response],
                "tools_output": state["tools_output"],
                "tools_to_run": [],
                "waiting_for_tools": False
            }
    else:
        # No tool call needed
        return {
            "messages": state["messages"] + [response],
            "tools_output": state["tools_output"],
            "tools_to_run": [],
            "waiting_for_tools": False
        }

# Function to execute tools in parallel
def execute_tools_in_parallel(state: ParallelAgentState) -> ParallelAgentState:
    if not state["waiting_for_tools"] or not state["tools_to_run"]:
        return state
    
    # Create a copy of the current state
    new_state = deepcopy(state)
    tools_to_run = new_state["tools_to_run"]
    
    # Record start time for performance measurement
    start_time = time.time()
    
    # Execute tools in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all tool executions to the executor
        future_to_tool = {
            executor.submit(execute_tool, tool_call): tool_call
            for tool_call in tools_to_run
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_tool):
            tool_call = future_to_tool[future]
            try:
                result = future.result()
                tool_type = result["tool_type"]
                tool_result = result["result"]
                
                # Store the result in tools_output
                new_state["tools_output"][tool_type] = tool_result
                
                # Add to messages
                new_state["messages"].append(
                    AIMessage(content=f"Tool '{tool_type}' Result: {tool_result}")
                )
            except Exception as e:
                # Handle any exceptions during tool execution
                tool_type = tool_call.get("tool_type", "unknown")
                new_state["tools_output"][tool_type] = f"Error executing {tool_type}: {str(e)}"
                new_state["messages"].append(
                    AIMessage(content=f"Tool '{tool_type}' Error: {str(e)}")
                )
    
    # Calculate and log the execution time
    execution_time = time.time() - start_time
    print(f"Parallel tool execution completed in {execution_time:.2f} seconds")
    
    # Add timing information to the state
    new_state["messages"].append(
        AIMessage(content=f"[System] Parallel tool execution completed in {execution_time:.2f} seconds")
    )
    
    # Reset the state for the next iteration
    new_state["tools_to_run"] = []
    new_state["waiting_for_tools"] = False
    
    return new_state

# Function to generate final response for parallel agent
def parallel_generate_response(state: ParallelAgentState) -> ParallelAgentState:
    # Create a system prompt for generating the final response
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent assistant. Generate a helpful response to the user's query.
        If you've used tools to gather information, incorporate the tool outputs in your response.
        Make your response conversational and helpful.
        """),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Tool outputs: {tool_outputs}\n\nBased on all information above, provide a final response to the user's query.")
    ])
    
    # Create an LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # Get just the human messages for context
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    
    # Format tool outputs for the prompt
    tool_outputs_str = "\n".join([f"{tool}: {result}" for tool, result in state["tools_output"].items()]) if state["tools_output"] else "No tools were used."
    
    # Invoke the LLM
    response = llm.invoke(
        prompt.invoke({
            "messages": human_messages, 
            "tool_outputs": tool_outputs_str
        })
    )
    
    # Update state with final response
    new_state = deepcopy(state)
    new_state["messages"].append(response)
    
    return new_state

#---------------------------
# SEQUENTIAL AGENT FUNCTIONS
#---------------------------

# Function to route based on the agent's decision for sequential agent
def sequential_route(state: SequentialAgentState) -> Literal["web_search", "calculator", "respond"]:
    next_steps = state["next_steps"]
    if not next_steps:
        return "respond"
    next_step = next_steps[0]
    
    if next_step == "web_search":
        return "web_search"
    elif next_step == "calculator":
        return "calculator"
    else:
        return "respond"

# Model that decides on the next steps for sequential agent
def decide_next_steps(state: SequentialAgentState) -> SequentialAgentState:
    # Create a system prompt for the agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent assistant with access to two tools:
        1. Web Search Tool: Finds information online
        2. Calculator Tool: Performs mathematical calculations
        
        Based on the user's request, determine which tool(s) to use, if any.
        If you need to use a tool, respond with a JSON object in the following format:
        
        ```json
        {{
            "thoughts": "Your reasoning about what tools to use",
            "next_steps": ["web_search" and/or "calculator"],
            "query": "Your search query" (if using web_search),
            "expression": "Your math expression" (if using calculator)
        }}
        ```
        
        If you don't need to use any tools to answer the question, just respond directly.
        You can use both tools if needed. List them in the order you want to use them.
        """),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Create an LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # Invoke the LLM
    response = llm.invoke(prompt.invoke({"messages": state["messages"]}))
    
    # Check if the response contains a tool call
    if "```json" in response.content:
        try:
            json_str = response.content.split("```json")[1].split("```")[0]
            action = json.loads(json_str)
            next_steps = action.get("next_steps", [])
            return {
                "messages": state["messages"] + [response],
                "tools_output": state["tools_output"],
                "next_steps": next_steps
            }
        except Exception:
            # If JSON parsing fails, assume no tool is needed
            return {
                "messages": state["messages"] + [response],
                "tools_output": state["tools_output"],
                "next_steps": []
            }
    else:
        # No tool call needed
        return {
            "messages": state["messages"] + [response],
            "tools_output": state["tools_output"],
            "next_steps": []
        }

# Function to use web search tool for sequential agent
def use_web_search(state: SequentialAgentState) -> SequentialAgentState:
    # Record start time
    start_time = time.time()
    
    # Extract the last message to find the query
    last_ai_message = [msg for msg in state["messages"] if isinstance(msg, AIMessage)][-1]
    content = last_ai_message.content
    
    # Extract JSON part
    try:
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0]
        else:
            # Try to find JSON object between curly braces
            start_index = content.find("{")
            end_index = content.rfind("}") + 1
            if start_index != -1 and end_index != 0:
                json_str = content[start_index:end_index]
            else:
                # Fallback to a simple query extraction
                query = content.split("query:")[1].split("\n")[0].strip() if "query:" in content else ""
                result = web_search_tool(query)
                execution_time = time.time() - start_time
                print(f"Sequential web search completed in {execution_time:.2f} seconds")
                return {
                    "messages": state["messages"] + [
                        AIMessage(content=f"Web Search Result: {result}"),
                        AIMessage(content=f"[System] Sequential web search completed in {execution_time:.2f} seconds")
                    ],
                    "tools_output": state["tools_output"] + [result],
                    "next_steps": state["next_steps"][1:]
                }
                
        action_json = json.loads(json_str)
        query = action_json.get("query", "")
    except Exception as e:
        # If JSON parsing fails, try to extract query directly
        if "query:" in content:
            query = content.split("query:")[1].split("\n")[0].strip()
        else:
            query = "general information"
    
    result = web_search_tool(query)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    print(f"Sequential web search completed in {execution_time:.2f} seconds")
    
    # Update state
    return {
        "messages": state["messages"] + [
            AIMessage(content=f"Web Search Result: {result}"),
            AIMessage(content=f"[System] Sequential web search completed in {execution_time:.2f} seconds")
        ],
        "tools_output": state["tools_output"] + [result],
        "next_steps": state["next_steps"][1:]  # Remove the first step as it's been completed
    }

# Function to use calculator tool for sequential agent
def use_calculator(state: SequentialAgentState) -> SequentialAgentState:
    # Record start time
    start_time = time.time()
    
    # Extract the last message to find the expression
    last_ai_message = [msg for msg in state["messages"] if isinstance(msg, AIMessage)][-1]
    content = last_ai_message.content
    
    # Extract JSON part
    try:
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0]
        else:
            # Try to find JSON object between curly braces
            start_index = content.find("{")
            end_index = content.rfind("}") + 1
            if start_index != -1 and end_index != 0:
                json_str = content[start_index:end_index]
            else:
                # Fallback to a simple expression extraction
                expression = content.split("expression:")[1].split("\n")[0].strip() if "expression:" in content else ""
                result = calculator_tool(expression)
                execution_time = time.time() - start_time
                print(f"Sequential calculator completed in {execution_time:.2f} seconds")
                return {
                    "messages": state["messages"] + [
                        AIMessage(content=f"Calculator Result: {result}"),
                        AIMessage(content=f"[System] Sequential calculator completed in {execution_time:.2f} seconds")
                    ],
                    "tools_output": state["tools_output"] + [result],
                    "next_steps": state["next_steps"][1:]
                }
                
        action_json = json.loads(json_str)
        expression = action_json.get("expression", "")
    except Exception as e:
        # If JSON parsing fails, try to extract expression directly
        if "expression:" in content:
            expression = content.split("expression:")[1].split("\n")[0].strip()
        else:
            expression = "0"
    
    result = calculator_tool(expression)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    print(f"Sequential calculator completed in {execution_time:.2f} seconds")
    
    # Update state
    return {
        "messages": state["messages"] + [
            AIMessage(content=f"Calculator Result: {result}"),
            AIMessage(content=f"[System] Sequential calculator completed in {execution_time:.2f} seconds")
        ],
        "tools_output": state["tools_output"] + [result],
        "next_steps": state["next_steps"][1:]  # Remove the first step as it's been completed
    }

# Function to generate final response for sequential agent
def sequential_generate_response(state: SequentialAgentState) -> SequentialAgentState:
    # Create a system prompt for generating the final response
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent assistant. Generate a helpful response to the user's query.
        If you've used tools to gather information, incorporate the tool outputs in your response.
        Make your response conversational and helpful.
        """),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Tool outputs: {tool_outputs}\n\nBased on all information above, provide a final response to the user's query.")
    ])
    
    # Create an LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # Get just the human messages for context
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    
    # Invoke the LLM
    response = llm.invoke(
        prompt.invoke({
            "messages": human_messages, 
            "tool_outputs": "\n".join(state["tools_output"]) if state["tools_output"] else "No tools were used."
        })
    )
    
    # Update state with final response
    return {
        "messages": state["messages"] + [response],
        "tools_output": state["tools_output"],
        "next_steps": []
    }

#---------------------------
# GRAPH BUILDING FUNCTIONS
#---------------------------

# Build the parallel graph
def build_parallel_agent():
    # Create a new graph
    builder = StateGraph(ParallelAgentState)
    
    # Add nodes
    builder.add_node("decide_tools", decide_tools)
    builder.add_node("execute_tools", execute_tools_in_parallel)
    builder.add_node("respond", parallel_generate_response)
    
    # Add conditional edges
    builder.add_conditional_edges(
        "decide_tools",
        parallel_route,
        {
            "execute_tools": "execute_tools",
            "respond": "respond"
        }
    )
    
    builder.add_conditional_edges(
        "execute_tools",
        parallel_route,
        {
            "decide_tools": "decide_tools",
            "respond": "respond"
        }
    )
    
    builder.add_edge("respond", END)
    
    # Set the entry point
    builder.set_entry_point("decide_tools")
    
    # Compile the graph
    return builder.compile()

# Build the sequential graph
def build_sequential_agent():
    # Create a new graph
    builder = StateGraph(SequentialAgentState)
    
    # Add nodes
    builder.add_node("decide", decide_next_steps)
    builder.add_node("web_search", use_web_search)
    builder.add_node("calculator", use_calculator)
    builder.add_node("respond", sequential_generate_response)
    
    # Add edges
    builder.add_conditional_edges(
        "decide",
        sequential_route,
        {
            "web_search": "web_search",
            "calculator": "calculator",
            "respond": "respond",
        }
    )
    builder.add_edge("web_search", "decide")
    builder.add_edge("calculator", "decide")
    builder.add_edge("respond", END)
    
    # Set the entry point
    builder.set_entry_point("decide")
    
    # Compile the graph
    return builder.compile()

# Create agents
parallel_agent = build_parallel_agent()
sequential_agent = build_sequential_agent()

#---------------------------
# USAGE FUNCTIONS
#---------------------------

# Function to run the parallel agent
def chat_with_parallel_agent(user_input: str):
    # Initialize state
    state = {
        "messages": [HumanMessage(content=user_input)],
        "tools_output": {},
        "tools_to_run": [],
        "waiting_for_tools": False
    }
    
    # Record start time
    start_time = time.time()
    
    # Run the agent
    result = parallel_agent.invoke(state)
    
    # Calculate total execution time
    total_time = time.time() - start_time
    print(f"Total parallel agent execution time: {total_time:.2f} seconds")
    
    # Return the last message
    return result["messages"][-1].content

# Function to run the sequential agent
def chat_with_sequential_agent(user_input: str):
    # Initialize state
    state = {
        "messages": [HumanMessage(content=user_input)],
        "tools_output": [],
        "next_steps": []
    }
    
    # Record start time
    start_time = time.time()
    
    # Run the agent
    result = sequential_agent.invoke(state)
    
    # Calculate total execution time
    total_time = time.time() - start_time
    print(f"Total sequential agent execution time: {total_time:.2f} seconds")
    
    # Return the last message
    return result["messages"][-1].content

# Function to compare parallel and sequential execution
def compare_execution_times(query: str):
    print("\n===== TIMING COMPARISON =====")
    print(f"Query: '{query}'")
    print("Running sequential agent...")
    
    # Run sequential agent and measure time
    seq_start = time.time()
    seq_response = chat_with_sequential_agent(query)
    seq_time = time.time() - seq_start
    
    print("\nRunning parallel agent...")
    
    # Run parallel agent and measure time
    par_start = time.time()
    par_response = chat_with_parallel_agent(query)
    par_time = time.time() - par_start
    
    # Calculate time difference and percentage improvement
    time_diff = seq_time - par_time
    percentage_improvement = (time_diff / seq_time) * 100 if seq_time > 0 else 0
    
    print("\n===== RESULTS =====")
    print(f"Sequential execution time: {seq_time:.2f} seconds")
    print(f"Parallel execution time: {par_time:.2f} seconds")
    print(f"Time saved: {time_diff:.2f} seconds ({percentage_improvement:.1f}% faster)")
    
    print("\n===== RESPONSES =====")
    print("Sequential agent response:")
    print(seq_response)
    print("\nParallel agent response:")
    print(par_response)

# Example interaction
if __name__ == "__main__":
    # Example for a query that requires both tools
    complex_query = "If the world population is 8 billion and there are 195 countries, what's the average population per country?"
    
    # Compare execution times
    compare_execution_times(complex_query)
    
    # Additional examples
    simple_queries = [
        "What's the current population of the world?",
        "Calculate 256 * 15 / 2",
        "What's the fastest mammal on earth and how fast can it run in mph?"
    ]
    
    # Test with additional queries if desired
    for i, query in enumerate(simple_queries):
        print(f"\n\nTesting additional query {i+1}: '{query}'")
        compare_execution_times(query)