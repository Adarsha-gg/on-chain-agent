from eth_utils import to_hex, keccak
from web3 import Web3,HTTPProvider
w3 = Web3(HTTPProvider('https://young-skilled-sunset.base-mainnet.quiknode.pro/b8e701a962a9f1f33ead6bea5a9a2d261317763e'))


message_hash = '{
  "method": null,
  "types": [
    "uint32",
    "address",
    "uint32",
    "uint32"
  ],
  "inputs": [
    3181390099,
    "0x4270b7C7DFa9547583779aa88B82ceaE847b863B",
    2000000,
    2015964922
  ],
  "names": [
    "_localDomain",
    "_attester",
    "_maxMessageBodySize",
    "_version"
  ]
}'
hasher = w3.solidity_keccak(['bytes'], message_hash)
print(message_hash)
print(hasher)