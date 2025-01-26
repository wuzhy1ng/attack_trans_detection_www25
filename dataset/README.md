# Quickly start

Use the following code to load the transaction graph by `networkx`:

```python
from dataset import NetworkxDataset

dataset = NetworkxDataset(path=r'/path/to/dataset')
for txhash, label, graph in dataset.iter_read():
    # `txhash` means transaction hash, identifying the transactions.
    # `label` means the label of the transaction identified by `txhash`.
    # `graph` is the transaction graph, with the type as `networkx.MultiDiGraph`. 
    # e.g. ('0x16d2628be31b9e507223add500a70b118951af6eb7ca2744953060dbe0f5ecfa', 'Honeypot', <networkx.classes.multidigraph.MultiDiGraph object at 0x000001F2502D3970>)
    print(txhash, label, graph)

```

# Data description

We model each transaction as a heterogeneous graph, namely `transaction graph`. The nodes and edges description as
follows:

## Node: Account

| field   | type | description                                    |
|---------|------|------------------------------------------------|
| address | str  | (**key**) The address of the external account. |

# Node: Contract

| field        | type      | description                                                      |
|--------------|-----------|------------------------------------------------------------------|
| address      | str       | (**key**) The address of the contract.                           |
| event_names  | List[str] | An list of event names, where the event emitted by the contract. |
| opcodes      | List[str] | An opcode list of the contract code.                             |                                          |

# Edge: Transaction

| field              | type      | description                                                                       |
|--------------------|-----------|-----------------------------------------------------------------------------------|
| address_from       | str       | (**key**) The from address of the external transaction.                           |
| address_to         | str       | (**key**) The to address of the external transaction.                             |
| value              | int       | The external transaction value of naive token, e.g. ETH, BNB.                     |
| gas                | int       | The gas fee of the external transaction.                                          |
| gas_price          | int       | The gas price of the external transaction.                                        |
| is_create_contract | bool      | Indicate a contract was created, if `True`, the `address_to` is created contract. |
| is_error           | bool      | Indicate whether the external transaction got an error.                           |
| func_name          | List[str] | The function name triggered by the external transaction.                          |

# Edge: Trace (i.e. Internal transaction)

| field        | type      | description                                                                   |
|--------------|-----------|-------------------------------------------------------------------------------|
| address_from | str       | (**key**) The from address of the internal transaction.                       |
| address_to   | str       | (**key**) The to address of the internal transaction.                         |
| trace_id     | int       | The id for identifying the triggered order of each internal transaction item. |
| trace_type   | str       | The debug trace type, e.g. `CALL`, `STATICCALL`, etc.                         |
| value        | int       | The internal transaction value of naive token.                                |
| gas          | int       | The gas fee of the internal transaction.                                      |
| gas_price    | int       | The gas price of the internal transaction.                                    |
| func_name    | List[str] | The function name triggered by the internal transaction.                      |

# Edge: Token20Transfer

| field            | type      | description                                                             |
|------------------|-----------|-------------------------------------------------------------------------|
| address_from     | str       | (**key**) The from address of the token transfer.                       |
| address_to       | str       | (**key**) The to address of the token transfer.                         |
| log_index        | int       | (**key**) The id for identifying the triggered order of each event log. |
| value            | int       | The transferred value of specific token.                                |
| contract_address | str       | The token contract address, emitting the token transfer.                |
| name             | List[str] | The token name.                                                         |
| token_symbol     | str       | The token symbol.                                                       |
| decimals         | int       | The token decimals, based on `10`.                                      |
| total_supply     | int       | The token total supply.                                                 |

# Edge: Token721Transfer

| field            | type      | description                                                             |
|------------------|-----------|-------------------------------------------------------------------------|
| address_from     | str       | (**key**) The from address of the token transfer.                       |
| address_to       | str       | (**key**) The to address of the token transfer.                         |
| log_index        | int       | (**key**) The id for identifying the triggered order of each event log. |
| token_id         | int       | The transferred token id of specific token.                             |
| contract_address | str       | The token contract address, emitting the token transfer.                |
| metadata         | List[str] | The NFT metadata, containing a list of words.                           |
| name             | List[str] | The token name.                                                         |
| token_symbol     | str       | The token symbol.                                                       |
| total_supply     | int       | The token total supply.                                                 |

# Edge: Token1155Transfer

| field            | type      | description                                                             |
|------------------|-----------|-------------------------------------------------------------------------|
| address_from     | str       | (**key**) The from address of the token transfer.                       |
| address_to       | str       | (**key**) The to address of the token transfer.                         |
| log_index        | int       | (**key**) The id for identifying the triggered order of each event log. |
| token_id         | int       | The transferred token id of specific token.                             |
| value            | int       | The transferred value of specific token.                                |
| contract_address | str       | The token contract address, emitting the token transfer.                |
| metadata         | List[str] | The NFT metadata, containing a list of words.                           |
| name             | List[str] | The token name.                                                         |
| token_symbol     | str       | The token symbol.                                                       |
| decimals         | int       | The token decimals, based on `10`.                                      |
| total_supply     | int       | The token total supply.                                                 |

# Edge: TokenApproval

| field            | type      | description                                                             |
|------------------|-----------|-------------------------------------------------------------------------|
| address_from     | str       | (**key**) The from address of the token approval action.                |
| address_to       | str       | (**key**) The to address of the token approval action.                  |
| log_index        | int       | (**key**) The id for identifying the triggered order of each event log. |
| value            | int       | The approved value of specific token.                                   |
| contract_address | str       | The token contract address, emitting the token approval.                |
| name             | List[str] | The token name.                                                         |
| token_symbol     | str       | The token symbol.                                                       |
| decimals         | int       | The token decimals, based on `10`.                                      |
| total_supply     | int       | The token total supply.                                                 |

# Edge: TokenApprovalAll

| field            | type      | description                                                             |
|------------------|-----------|-------------------------------------------------------------------------|
| address_from     | str       | (**key**) The from address of the token approval action.                |
| address_to       | str       | (**key**) The to address of the token approval action.                  |
| log_index        | int       | (**key**) The id for identifying the triggered order of each event log. |
| approved         | bool      | Is approved all token for `address_to`.                                 |
| contract_address | str       | The token contract address, emitting the token approval.                |
| name             | List[str] | The token name.                                                         |
| token_symbol     | str       | The token symbol.                                                       |
| total_supply     | int       | The token total supply.                                                 |