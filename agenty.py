import os
import sys
import time

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Import CDP Agentkit Langchain Extension.
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from cdp_langchain.tools import CdpTool
import requests
from typing import Optional
from pydantic import BaseModel, Field
PRICE_TOOL_DESCRIPTION = """
Tool for retrieving real-time cryptocurrency token prices. 
Supports fetching current market data including price, market cap, and 24-hour price change.
Can use token symbols or contract addresses.
"""
class TokenPriceTool(BaseModel):
    """Pydantic model for token price retrieval tool."""
    token_identifier: str = Field(
        ..., 
        description="Token name (e.g., 'Bitcoin', 'Ethereum') or contract address",
        examples=["Bitcoin", "Ethereum", "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984"]
    )
    currency: Optional[str] = Field(
        default="USD", 
        description="Target currency for price conversion",
        examples=["USD", "EUR", "GBP"]
    )
def get_token_price(token_identifier: str, currency: str = "USD") -> dict:
    """Retrieve real-time price information for a cryptocurrency token.
    Args:
        token_identifier (str): Token symbol or contract address to fetch price for.
        currency (str, optional): Target currency for price conversion. Defaults to "USD".
    Returns:
        dict: A dictionary containing token price information.
    """
    try:
        # Determine if input is a contract address or a token symbol
        if token_identifier.startswith('0x'):
            # Use contract address lookup
            url = f"https://api.coingecko.com/CG-2pUQLuvdJhQVuWUuyGZ61kRS/v3/simple/token_price/ethereum?contract_addresses={token_identifier}&vs_currencies=USD&include_market_cap=true&include_24hr_change=true"
        else:
            # Use token symbol lookup
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={token_identifier}&vs_currencies=USD&include_market_cap=true&x_cg_demo_api_key=CG-2pUQLuvdJhQVuWUuyGZ61kRS"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        # Process and format the response
        if not data:
            return {"error": "Token not found"}
        # Extract price information
        token_data = list(data.values())[0] if data else {}
        return {
            "price": token_data.get(f"{currency.lower()}"),
            "market_cap": token_data.get(f"{currency.lower()}_market_cap"),
            "24h_change": token_data.get(f"{currency.lower()}_24h_change"),
            "token": token_identifier
        }
    except requests.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}
def run_token_price_tool(token_identifier: str, currency: str = "USD") -> str:
    """Wrapper function to run the token price tool and format output.
    Args:
        token_identifier (str): Token symbol or contract address to fetch price for.
        currency (str, optional): Target currency for price conversion. Defaults to "USD".
    Returns:
        str: Formatted string with token price information.
    """
    result = get_token_price(token_identifier, currency)
    
    if "error" in result:
        return result["error"]
    
    return f"""Token Price Information for {token_identifier}:
Price: {result['price']} {currency}
Market Cap: {result['market_cap']} {currency}
24h Change: {result['24h_change']}%"""

# Configure a file to persist the agent's CDP MPC Wallet Data.
wallet_data_file = "wallet_data.txt"


def initialize_agent():
    """Initialize the agent with CDP Agentkit."""
    # Initialize LLM.
    llm = ChatOpenAI(model="gpt-4o-mini")

    wallet_data = None

    if os.path.exists(wallet_data_file):
        with open(wallet_data_file) as f:
            wallet_data = f.read()

    # Configure CDP Agentkit Langchain Extension.
    values = {}
    if wallet_data is not None:
        # If there is a persisted agentic wallet, load it and pass to the CDP Agentkit Wrapper.
        values = {"cdp_wallet_data": wallet_data}

    agentkit = CdpAgentkitWrapper(**values)

    # persist the agent's CDP MPC Wallet Data.
    wallet_data = agentkit.export_wallet()
    with open(wallet_data_file, "w") as f:
        f.write(wallet_data)

    # Initialize CDP Agentkit Toolkit and get tools.
    cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(agentkit)
    tools = cdp_toolkit.get_tools()

    realTool = CdpTool(
        name="token_price",
        description=PRICE_TOOL_DESCRIPTION,
        cdp_agentkit_wrapper=agentkit,
        func=run_token_price_tool,
        args_schema=TokenPriceTool,
    )
    # Ensure tools is a list and add the new tool
    if tools is None:
        tools = []
    tools.append(realTool)
    # Store buffered conversation history in memory.
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "CDP Agentkit Chatbot Example!"}}

    # Create ReAct Agent using the LLM and CDP Agentkit tools.
    return create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        state_modifier=(
            "You are a helpful agent that can interact onchain using the Coinbase Developer Platform AgentKit. "
            "You are empowered to interact onchain using your tools. If you ever need funds, you can request "
            "them from the faucet if you are on network ID 'base-sepolia'. If not, you can provide your wallet "
            "details and request funds from the user. Before executing your first action, get the wallet details "
            "to see what network you're on. If there is a 5XX (internal) HTTP error code, ask the user to try "
            "again later. If someone asks you to do something you can't do with your currently available tools, "
            "you must say so, and encourage them to implement it themselves using the CDP SDK + Agentkit, "
            "recommend they go to docs.cdp.coinbase.com for more information. Be concise and helpful with your "
            "responses. Refrain from restating your tools' descriptions unless it is explicitly requested."
        ),

    ), config


# Autonomous Mode
def run_autonomous_mode(agent_executor, config, interval=10):
    """Run the agent autonomously with specified intervals."""
    print("Starting autonomous mode...")
    while True:
        try:
            # Provide instructions autonomously
            thought = (
                "Be creative and do something interesting on the blockchain. "
                "Choose an action or set of actions and execute it that highlights your abilities."
            )

            # Run agent in autonomous mode
            for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content=thought)]}, config
            ):
                if "agent" in chunk:
                    print(chunk["agent"]["messages"][0].content)
                elif "tools" in chunk:
                    print(chunk["tools"]["messages"][0].content)
                print("-------------------")

            # Wait before the next action
            time.sleep(interval)

        except KeyboardInterrupt:
            print("Goodbye Agent!")
            sys.exit(0)


# Chat Mode
def run_chat_mode(agent_executor, config):
    """Run the agent interactively based on user input."""
    print("Starting chat mode... Type 'exit' to end.")
    while True:
        try:
            user_input = input("\nPrompt: ")
            if user_input.lower() == "exit":
                break

            # Run agent with the user's input in chat mode
            for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content=user_input)]}, config
            ):
                if "agent" in chunk:
                    print(chunk["agent"]["messages"][0].content)
                elif "tools" in chunk:
                    print(chunk["tools"]["messages"][0].content)
                print("-------------------")

        except KeyboardInterrupt:
            print("Goodbye Agent!")
            sys.exit(0)


# Mode Selection
def choose_mode():
    """Choose whether to run in autonomous or chat mode based on user input."""
    while True:
        print("\nAvailable modes:")
        print("1. chat    - Interactive chat mode")
        print("2. auto    - Autonomous action mode")

        choice = input("\nChoose a mode (enter number or name): ").lower().strip()
        if choice in ["1", "chat"]:
            return "chat"
        elif choice in ["2", "auto"]:
            return "auto"
        print("Invalid choice. Please try again.")


def main():
    """Start the chatbot agent."""
    agent_executor, config = initialize_agent()

    mode = choose_mode()
    if mode == "chat":
        run_chat_mode(agent_executor=agent_executor, config=config)
    elif mode == "auto":
        run_autonomous_mode(agent_executor=agent_executor, config=config)


if __name__ == "__main__":
    print("Starting Agent...")
    main()