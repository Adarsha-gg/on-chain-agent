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

from typing import Dict
import time
import requests
from eth_utils import to_hex, keccak
from pydantic import BaseModel, Field
from cdp import Wallet
from web3 import Web3, HTTPProvider
w3 = Web3(HTTPProvider('https://young-skilled-sunset.base-mainnet.quiknode.pro/b8e701a962a9f1f33ead6bea5a9a2d261317763e'))
data = w3.eth.get_transaction_receipt('0x9f1e7ffcc041e4af17f1f81b08bc113d39b96b8051e02bc77a4a17e98f761e5d')

event_topic = to_hex(keccak(text="MessageSent(bytes)"))
message_log = next(log for log in data.get('logs', []) if to_hex(log['topics'][0]) == event_topic)
message_bytes = message_log['data']
message_hash = to_hex(keccak(message_bytes))
# Print the extracted 'data'
print(message_bytes)
print(message_hash)



