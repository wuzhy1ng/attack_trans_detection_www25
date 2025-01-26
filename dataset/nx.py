from typing import Iterator, Tuple

import networkx as nx

from daos import EventLogReader, TransactionReader, \
    Token20TransferReader, Token721TransferReader, Token1155TransferReader, \
    TokenApprovalReader, TokenApprovalAllReader, \
    DCFGBlockReader, DCFGEdgeReader, JointReader
from utils.signature import load_signatures

call_ops = {
    'CALL': True, 'CALLCODE': True,
    'STATICCALL': True, 'DELEGATECALL': True,
    'CREATE': True, 'CREATE2': True,
    'SELFDESTRUCT': True,
}


class NetworkxDataset:
    def __init__(self, data_path: str, signature_path: str):
        self.data_path = data_path
        self.signature2keyword = load_signatures(signature_path)

    def iter_read(self) -> Iterator[Tuple[str, nx.MultiDiGraph]]:
        """
        load and read transaction graph iteratively.
        Each returned tuple contains (`transaction_hash`, `transaction_graph`)
        """
        yield from self._load_transaction_graphs(path=self.data_path)

    def _load_transaction_graphs(self, path: str) -> Iterator[nx.MultiDiGraph]:
        tx2graph = dict()
        tx2top_func_name = dict()  # record the first triggerred func name
        tx2block_path = dict()  # record the triggerred blocks
        block2log_cnt = dict()  # record the block triggerred log or not

        # load transaction
        reader = TransactionReader(path, signature2keyword=self.signature2keyword)
        reader = JointReader(
            path, 'TransactionReceiptItem.csv',
            joint_reader=reader,
            joint_key='transaction_hash'
        )
        for item in reader.iter_read():
            g = nx.MultiDiGraph()
            g.add_node(item['address_from'], type='Account')
            g.add_node(item['address_to'], type='Account')
            g.add_edge(
                item['address_from'], item['address_to'],
                value=int(item['value']),
                gas=int(item['gas']),
                gas_price=int(item['gas_price']),
                timestamp=int(item['timestamp']),
                block_number=int(item['block_number']),
                is_create_contract=item.get('created_contract') != '',
                is_error=item.get('is_error') == 'True',
                type='Transaction',
                transaction_index=int(item['transaction_index']),
            )
            tx2graph[item['transaction_hash']] = g
            tx2top_func_name[item['transaction_hash']] = item['func_name']
            tx2block_path[item['transaction_hash']] = dict()

        # load dcfg
        bid2operations = dict()
        for block in DCFGBlockReader(path).iter_read():
            bid2operations[block['block_id']] = block['operations']
            for op in block['operations']:
                if not op.startswith('LOG'):
                    continue
                block2log_cnt[block['block_id']] = block2log_cnt.get(block['block_id'], 0) + 1
        reader = DCFGEdgeReader(path, signature2keyword=self.signature2keyword)
        for control_flow in reader.iter_read():
            transaction_hash = control_flow['transaction_hash']
            from_block_id = control_flow['from_block_id']
            to_block_id = control_flow['to_block_id']
            g = tx2graph.get(transaction_hash)
            if g is None:
                continue
            if not g.has_node(from_block_id):
                operations = bid2operations.get(from_block_id, [])
                g.add_node(
                    from_block_id,
                    operations=operations,
                    type='Block',
                )
            if not g.has_node(to_block_id):
                operations = bid2operations.get(to_block_id, [])
                g.add_node(
                    to_block_id,
                    operations=operations,
                    type='Block',
                )

            # add select edge in the index of 0
            if control_flow['index'] == 0:
                g.add_node(
                    control_flow['address_from'],
                    type='Contract',
                )
                g.add_edge(
                    control_flow['address_from'], from_block_id,
                    func_name=tx2top_func_name[transaction_hash],
                    type='Select',
                    index=0,
                )
                tx2block_path[transaction_hash][0] = from_block_id
            tx2block_path[transaction_hash][control_flow['index']] = to_block_id

            # add select edge for the non-zero indices
            if call_ops.get(control_flow['flow_type']):
                g.add_node(control_flow['address_from'], type='Contract')
                g.add_node(control_flow['address_to'], type='Contract')
                g.add_edge(
                    control_flow['address_from'], control_flow['address_to'],
                    value=int(control_flow['value']),
                    gas=int(control_flow['gas']),
                    type=control_flow['flow_type'],
                    index=control_flow['index'],
                )
                g.add_edge(
                    control_flow['address_to'], to_block_id,
                    func_name=control_flow['func_name'],
                    type='Select',
                    index=control_flow['index'],
                )
                continue

            # add edge for other control flows
            g.add_edge(
                from_block_id, to_block_id,
                type=control_flow['flow_type'],
                index=control_flow['index'],
            )

        # load event logs
        txhash2logs = dict()
        for item in EventLogReader(path, signature2keyword=self.signature2keyword).iter_read():
            txhash = item['transaction_hash']
            if not txhash2logs.get(txhash):
                txhash2logs[txhash] = list()
            txhash2logs[txhash].append(item)
        for txhash, logs in txhash2logs.items():
            len_log = len(logs)
            if len_log == 0:
                continue
            logs.sort(key=lambda _log: int(_log['log_index']))
            emit_index, g = 0, tx2graph.get(txhash)
            if g is None:
                continue
            block_path = list(tx2block_path[txhash].items())
            block_path.sort(key=lambda _t: _t[0])
            block_path = [item[1] for item in block_path]
            for block_id in block_path:
                num_logs = block2log_cnt.get(block_id)
                if not num_logs:
                    continue
                while num_logs > 0 and emit_index < len_log:
                    num_logs -= 1
                    log = logs[emit_index]
                    topic0 = log['topics'][0] if len(log['topics']) > 0 else ''
                    log_id = '{}@{}'.format(txhash, topic0)
                    if not g.has_node(log_id):
                        g.add_node(
                            log_id,
                            event_name=log['event_name'],
                            type='Log',
                        )
                    g.add_edge(
                        block_id, log_id,
                        timestamp=int(log['timestamp']),
                        removed=log['removed'] == 'True',
                        type='Emit',
                        emit_index=emit_index,
                    )
                    emit_index += 1

        # load token20 transfer
        reader = JointReader(
            path, 'TokenPropertyItem.csv',
            joint_reader=Token20TransferReader(path),
            joint_key='contract_address',
        )
        for item in reader.iter_read():
            g = tx2graph.get(item['transaction_hash'])
            if g is None:
                continue
            if not g.has_node(item['address_from']):
                g.add_node(item['address_from'], type='Account')
            if not g.has_node(item['address_to']):
                g.add_node(item['address_to'], type='Account')
            g.add_edge(
                item['address_from'], item['address_to'],
                value=int(float(item['value'])),
                contract_address=item['contract_address'],
                name=item.get('name', ''),
                token_symbol=item.get('token_symbol', ''),
                decimals=int(item.get('decimals', -1)),
                total_supply=int(float(item.get('total_supply', -1))),
                type='Token20Transfer',
                log_index=int(item['log_index']),
            )

        # load token721 transfer
        reader = JointReader(
            path, 'TokenPropertyItem.csv',
            joint_reader=Token721TransferReader(path),
            joint_key='contract_address',
        )
        for item in reader.iter_read():
            g = tx2graph.get(item['transaction_hash'])
            if g is None:
                continue
            if not g.has_node(item['address_from']):
                g.add_node(item['address_from'], type='Account')
            if not g.has_node(item['address_to']):
                g.add_node(item['address_to'], type='Account')
            g.add_edge(
                item['address_from'], item['address_to'],
                token_id=int(item['token_id']),
                contract_address=item['contract_address'],
                name=item.get('name', ''),
                token_symbol=item.get('token_symbol', ''),
                total_supply=int(float(item.get('total_supply', -1))),
                type='Token721Transfer',
                log_index=int(item['log_index']),
            )

        # load token1155 transfer
        reader = JointReader(
            path, 'TokenPropertyItem.csv',
            joint_reader=Token1155TransferReader(path),
            joint_key='contract_address',
        )
        for item in reader.iter_read():
            g = tx2graph.get(item['transaction_hash'])
            if g is None:
                continue
            if not g.has_node(item['address_from']):
                g.add_node(item['address_from'], type='Account')
            if not g.has_node(item['address_to']):
                g.add_node(item['address_to'], type='Account')
            g.add_edge(
                item['address_from'], item['address_to'],
                token_id=int(item['token_id']),
                value=int(float(item['value'])),
                contract_address=item['contract_address'],
                name=item.get('name', ''),
                token_symbol=item.get('token_symbol', ''),
                decimals=int(item.get('decimals', -1)),
                total_supply=int(float(item.get('total_supply', -1))),
                type='Token1155Transfer',
                log_index=int(item['log_index']),
            )

        # load token approval
        reader = JointReader(
            path, 'TokenPropertyItem.csv',
            joint_reader=TokenApprovalReader(path),
            joint_key='contract_address',
        )
        for item in reader.iter_read():
            g = tx2graph.get(item['transaction_hash'])
            if g is None:
                continue
            if not g.has_node(item['address_from']):
                g.add_node(item['address_from'], type='Account')
            if not g.has_node(item['address_to']):
                g.add_node(item['address_to'], type='Account')
            g.add_edge(
                item['address_from'], item['address_to'],
                value=int(float(item['value'])),
                contract_address=item['contract_address'],
                name=item.get('name', ''),
                token_symbol=item.get('token_symbol', ''),
                decimals=int(item.get('decimals', -1)),
                total_supply=int(float(item.get('total_supply', -1))),
                type='TokenApproval',
                log_index=int(item['log_index']),
            )

        # load token approval all
        reader = JointReader(
            path, 'TokenPropertyItem.csv',
            joint_reader=TokenApprovalAllReader(path),
            joint_key='contract_address',
        )
        for item in reader.iter_read():
            g = tx2graph.get(item['transaction_hash'])
            if g is None:
                continue
            if not g.has_node(item['address_from']):
                g.add_node(item['address_from'], type='Account')
            if not g.has_node(item['address_to']):
                g.add_node(item['address_to'], type='Account')
            g.add_edge(
                item['address_from'], item['address_to'],
                approved=bool(item['approved'] == 'True'),
                contract_address=item['contract_address'],
                name=item.get('name', ''),
                token_symbol=item.get('token_symbol', ''),
                total_supply=int(float(item.get('total_supply', -1))),
                type='TokenApprovalAll',
                log_index=int(item['log_index']),
            )

        # return data
        for txhash, g in tx2graph.items():
            yield txhash, g


if __name__ == '__main__':
    for txhash, g in NetworkxDataset(
            data_path=r'C:\Users\87016\Downloads\tmp\raw\0',
            signature_path=r'D:\transCLR_data\signatures.csv'
    ).iter_read():
        print(txhash, g.number_of_nodes(), g.number_of_edges())
        node_type2cnt, edge_type2cnt = dict(), dict()
        for _, attr in g.nodes(data=True):
            node_type2cnt[attr['type']] = node_type2cnt.get(attr['type'], 0) + 1
        print(node_type2cnt)
        for _, _, attr in g.edges(data=True):
            edge_type2cnt[attr['type']] = edge_type2cnt.get(attr['type'], 0) + 1
        print(edge_type2cnt)
