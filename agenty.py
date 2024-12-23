import os
import sys
import time
import base64
import requests

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Import CDP Agentkit Langchain Extension.
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from cdp_langchain.tools import CdpTool


from typing import Optional,Dict, Any
from pydantic import BaseModel, Field
from cdp import Wallet
from web3 import Web3
from dotenv import load_dotenv

load_dotenv()

CLAIM_USDC_PROMPT = """
This tool claims USDC using Circle's Cross-Chain Transfer Protocol (CCTP). This only handles attestations and message receiving on the destination chain.
"""

class attestationInput(BaseModel):
    """Input argument schema for bridge USDC action."""
    
    attestation: str = Field(
        ...,
        description="The attestation from Circle's Iris service",
        example="0x4270b7C7DFa9547583779aa88B82ceaE847b863B"
    )
    message_bytes: str = Field(
        ...,
        description="The message bytes from the transaction log",
        example="000000000000000600000003000000000005B94E0000000000000000000000001682AE6375C4E4A97E4B583BC394C861A46D896200000000000000000000000019330D10D9CC8751218EAF51E8885D058642E08A000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000833589FCD6EDB6E08F4C7C32D4F71B54BDA029130000000000000000000000004270B7C7DFA9547583779AA88B82CEAE847B863B00000000000000000000000000000000000000000000000000000000001E848000000000000000000000000090D8E247C115C205E915FA9CA250175A78292EFA00"
    )

def load_wallet_from_file(filename: str) -> Optional[Wallet]:
    """Load wallet details from a file."""
    if os.path.exists(filename):
        with open(filename, "r") as file:
            wallet_data = file.read()
        return Wallet.from_json(wallet_data)
    return None

def recieve_cross_chain_message(destination_wallet: Wallet, message_bytes: str, attestation: str) -> str:
    """Receive cross-chain message. Try with the current wallet, and fall back to stored wallet if necessary."""
    try:
        # Attempt using the provided destination wallet
        receive_tx = destination_wallet.invoke_contract(
            contract_address=ARBITRUM_MESSAGE_TRANSMITTER_ADDRESS,
            method="receiveMessage",
            args={
                "message": message_bytes,
                "attestation": attestation,
            },
            abi=MESSAGE_TRANSMITTER_ABI,
        ).wait()
        return receive_tx.transaction.transaction_hash
    except Exception as e:
        print(f"Error using current wallet: {str(e)}")

        # Fallback: Load wallet from file and retry
        fallback_wallet = load_wallet_from_file("ar_wallet.txt")
        if fallback_wallet is None:
            raise ValueError("Failed to load fallback wallet from ar_wallet.txt.")

        print("Retrying with fallback wallet...")
        try:
            receive_tx = fallback_wallet.invoke_contract(
                contract_address=ARBITRUM_MESSAGE_TRANSMITTER_ADDRESS,
                method="receiveMessage",
                args={
                    "message": message_bytes,
                    "attestation": attestation,
                },
                abi=MESSAGE_TRANSMITTER_ABI,
            ).wait()
            return receive_tx.transaction.transaction_hash
        except Exception as fallback_error:
            raise RuntimeError(
                f"Both current and fallback wallets failed: {str(fallback_error)}"
            )

BRIDGE_USDC_PROMPT = """
This tool bridges USDC from Base-Mainnet to Arbitrum-Mainnet using Circle's Cross-Chain Transfer Protocol (CCTP). 
It handles the complete bridging process including approval, deposit for burn, waiting for attestation, 
and message receiving on the destination chain.
"""

# Contract addresses
BASE_TOKEN_MESSENGER_ADDRESS = "0x1682Ae6375C4E4A97e4B583BC394c861A46D8962"
ARBITRUM_MESSAGE_TRANSMITTER_ADDRESS = "0xC30362313FBBA5cf9163F0bb16a0e01f01A896ca"
USDC_BASE_ADDRESS = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

# ABI definitions
TOKEN_MESSENGER_ABI = [
    {
        "inputs": [
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
            {"internalType": "uint32", "name": "destinationDomain", "type": "uint32"},
            {"internalType": "bytes32", "name": "mintRecipient", "type": "bytes32"},
            {"internalType": "address", "name": "burnToken", "type": "address"},
        ],
        "name": "depositForBurn",
        "outputs": [
            {"internalType": "uint64", "name": "_nonce", "type": "uint64"},
        ],
        "stateMutability": "nonpayable",
        "type": "function",
    }
]

MESSAGE_TRANSMITTER_ABI = [
    {
        "inputs": [
            {"internalType": "bytes", "name": "message", "type": "bytes"},
            {"internalType": "bytes", "name": "attestation", "type": "bytes"},
        ],
        "name": "receiveMessage",
        "outputs": [
            {"internalType": "bool", "name": "success", "type": "bool"},
        ],
        "stateMutability": "nonpayable",
        "type": "function",
    }
]

class BridgeUsdcInput(BaseModel):
    """Input argument schema for bridge USDC action."""
    
    amount: str = Field(
        ...,
        description="Amount of USDC to bridge (in wei)",
        example="1000000"  # 1 USDC
    )

def pad_address(address: str) -> str:
    """Pad an Ethereum address to 32 bytes."""
    address = address.replace("0x", "")
    return "0x" + address.zfill(64)
    max_transactions: int = Field(
        default=50,
        description="Maximum number of transactions to retrieve",
        ge=1,
        le=1000
    )

def wait_for_attestation(message_hash: str, max_attempts: int = 90) -> str:
    """Wait for and retrieve attestation from Circle's Iris service."""
    for _ in range(max_attempts):
        response = requests.get(
            f"https://iris-api.circle.com/attestations/{message_hash}"
        )
        attestation_response = response.json()
        
        if attestation_response["status"] == "complete":
            return attestation_response["attestation"]
            
        time.sleep(20)  # Wait 20 seconds between attempts    
    raise TimeoutError("Attestation wait time exceeded")

def bridge_usdc(source_wallet: Wallet, destination_wallet: Wallet, amount: str, network:str) -> str:
    """Bridge USDC from Base to Arbitrum using Circle's CCTP.

    Args:
        source_wallet (Wallet): The Base network wallet containing USDC to bridge.
        destination_wallet (Wallet): The Arbitrum network wallet to receive the USDC.
        amount (str): Amount of USDC to bridge (in wei).

    Returns:
        str: A message containing the bridge operation details and transaction hashes.
    """
    # Get initial balances
    initial_base_balance = source_wallet.balance("usdc")  
    
    # Step 1: Approve TokenMessenger as spender
    approve_tx = source_wallet.invoke_contract(
        contract_address=USDC_BASE_ADDRESS,
        method="approve",
        args={"spender": BASE_TOKEN_MESSENGER_ADDRESS, "value": amount}
    ).wait()

    # Step 2: Call depositForBurn
    destination_address = pad_address(destination_wallet)

    deposit_tx = source_wallet.invoke_contract(
        contract_address=BASE_TOKEN_MESSENGER_ADDRESS,
        method="depositForBurn",
        args={
            "amount": amount,
            "destinationDomain": "3",  # Arbitrum domain
            "mintRecipient": destination_address,
            "burnToken": USDC_BASE_ADDRESS
        },
        abi=TOKEN_MESSENGER_ABI
    ).wait()

    # Step 3: Get messageHash from logs
    w3 = Web3(Web3.HTTPProvider(f'https://{network}.g.alchemy.com/v2/{os.getenv(ALCHEMY_API_KEY)}'))
    tx_hash = (deposit_tx.transaction_hash)
    data = w3.eth.get_transaction_receipt(tx_hash)# Get transaction receipt
    # Get logs from the transaction receipt
    logs = data['logs']
    event_topic = to_hex(keccak(text="MessageSent(bytes)"))
    message_log = next(log for log in data.get('logs', []) if to_hex(log['topics'][0]) == event_topic) # Extract the first log containing the specific topic
    message_bytes = (message_log['data']).hex()# Extract the 'data' field from the log
    message_bytes = message_bytes[128:624].upper()
    message_hash = f'0x{w3.keccak(hexstr=message_bytes).hex()}'
    
    # Step 4: Wait for attestation
    attestation = wait_for_attestation(message_hash)
    
    receive_tx = recieve_cross_chain_message(destination_wallet, message_bytes, attestation) 
    
    return (
        f"USDC Bridge completed successfully!\n"
        f"Approve TX: {approve_tx.transaction_hash}\n"
        f"Deposit TX: {deposit_tx.transaction_hash}\n"
        f"Receive TX: {receive_tx.transaction_hash}"
    )

WALLET_SEARCH_PROMPT = """
This tool searches for all transactions (trades, swaps, transfers) associated with a given wallet address. It retrieves a summary of recent transaction activities across different protocols and token types.
"""
class EtherscanWalletSearchInput(BaseModel):
    """Input argument schema for Etherscan wallet transaction search."""
    wallet_address: str = Field(
        ...,
        description="The Ethereum wallet address to search for transactions (e.g., '0x742d35Cc6634C0532925a3b844Bc454e4438f44e')",
        example="0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
    )

def search_wallet_transactions(wallet_address: str, max_transactions: int = 50) -> str:
     """
     Search for transactions associated with a specific wallet address.
     Args:
        wallet_address (str): Ethereum wallet address to search.
        api_key (str):  API key.
        max_transactions (int, optional): Maximum number of transactions to retrieve. Defaults to 50.
     Returns:
        str: Summarized transaction information.
    """    
     API_KEY = os.getenv('WALLET_API_KEY')
     encoded_key = base64.b64encode(API_KEY.encode()).decode()


     query = """
     query providerPorfolioQuery($addresses: [Address!]!, $networks: [Network!]!) {
     portfolio(addresses: $addresses, networks: $networks) {
        tokenBalances {
        address
        network
        token {
            balance
            balanceUSD
            baseToken {
            name
            symbol
            }
        }
        }
     }
     }
     """

     response = requests.post(
            'https://public.zapper.xyz/graphql',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Basic {encoded_key}'
            },
            json={
                'query': query,
                'variables': {
                    'addresses': [wallet_address],
                    'networks': ['ETHEREUM_MAINNET','POLYGON_MAINNET','BINANCE_SMART_CHAIN_MAINNET','BASE_MAINNET','METIS_MAINNET']
                }
            },
            timeout=30
        )
     response.raise_for_status()
     data = response.json()
     if 'errors' in data:
        raise ValueError(f"GraphQL Errors: {data['errors']}")
     balance = data['data']['portfolio']['tokenBalances']
     result = ''
     for item in balance:
        result += f"""Network:" {item["network"]}
        Symbol: {item["token"]["baseToken"]["symbol"]}
        Balance: {item["token"]["balance"]}
        Balance USD: {item["token"]["balanceUSD"]}\n"""
     return result   

     
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
    Api = os.getenv('COINGECKO_API_KEY')
    try:
        # Determine if input is a contract address or a token symbol
        if token_identifier.startswith('0x'):
            # Use contract address lookup
            url = f"https://api.coingecko.com/api/v3/simple/token_price/id={token_identifier}include_market_cap=true&include_24hr_vol=true&include_24hr_change=true&x_cg_demo_api_key={Api}"
        else:
            # Use token symbol lookup
             url = f"https://api.coingecko.com/api/v3/simple/price?ids={token_identifier}&vs_currencies=USD&include_market_cap=true&x_cg_demo_api_key={Api}"
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

    cross_chain_tool = CdpTool(
        name="bridge_usdc",
        description=BRIDGE_USDC_PROMPT,
        cdp_agentkit_wrapper=agentkit,
        func=bridge_usdc,
        args_schema=BridgeUsdcInput,
    )

    receive_cross_chain_message_tool = CdpTool(
        name="recieve_cross_chain_message",
        description=CLAIM_USDC_PROMPT,
        cdp_agentkit_wrapper=agentkit,
        func=recieve_cross_chain_message,
        args_schema=attestationInput
    )

    visualizer = CdpTool(
        name="visualize_wallet",
        description=WALLET_SEARCH_PROMPT,
        cdp_agentkit_wrapper=agentkit,
        func=search_wallet_transactions,
        args_schema=EtherscanWalletSearchInput,
    )

    # Ensure tools is a list and add the new tool
    if tools is None:
        tools = []
    tools.append(realTool)
    tools.append(cross_chain_tool)
    tools.append(receive_cross_chain_message_tool)
    tools.append(visualizer)
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