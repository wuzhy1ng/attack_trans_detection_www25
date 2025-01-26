import numpy as np
import networkx as nx
from scipy.sparse import lil_matrix


class HighOrderBlockMotifCounter:
    single_counters = {}

    def __init__(self, **kwargs):
        self.counter = HighOrderMotifCounter(**kwargs)

    async def count(self, edges: [dict]):
        hash_txs = dict()
        for edge in edges:
            txs = hash_txs.get(edge.get('transaction_hash'))
            if txs is None:
                txs = hash_txs[edge.get('transaction_hash')] = list()
            txs.append(edge)

        for txhash, txs in hash_txs.items():
            yield {
                'transaction_hash': txhash,
                'frequency': self.counter.count(txs),
            }


class HighOrderMotifCounter:
    def __init__(self, motif_size=3):
        self.motif_edges_num = list()
        self.motif_size = motif_size

        # 2-nodes
        if self.motif_size >= 2:
            self.motif_edges_num.extend([1, 2])

        # 3-nodes
        if self.motif_size >= 3:
            self.motif_edges_num.extend([
                3, 4, 5, 6, 3,
                4, 4, 2, 2, 2,
                3, 3, 4,
            ])

        # 4-nodes
        if self.motif_size >= 4:
            self.motif_edges_num.extend([4])

    def count(self, edges: list):
        if self.motif_size < 2:
            return dict()

        # init input graph
        g = nx.MultiDiGraph()
        _edges = [(
            e['address_from'] if e.get('address_from') else '',
            e['address_to'] if e.get('address_to') else ''
        ) for e in edges]
        g.add_edges_from(_edges)
        if g.number_of_edges() == 1:
            return {1: 1.0, **{i + 1: 0.0 for i in range(1, len(self.motif_edges_num))}}

        # convert multidigraph to digraph
        nodes_num = {node: i for i, node in enumerate(g.nodes())}
        edges = dict()
        for u, v, k in g.edges(keys=True):
            edges[(nodes_num[u], nodes_num[v])] = max(edges.get((u, v), 0), k)
        gg = nx.DiGraph()
        gg.add_weighted_edges_from([(edge[0], edge[1], weight + 1) for edge, weight in edges.items()])

        return self._count(gg)

    def _count(self, g: nx.DiGraph) -> dict:
        adj = nx.to_scipy_sparse_matrix(g)

        motif_matries = list()
        A = adj.astype(np.bool8).astype(np.int32)
        N = g.number_of_nodes()

        if self.motif_size < 2:
            return {}

        # compute 2-nodes motif adjacency matrix
        # U: unidirectional matrix
        # B: bidirectional matrix
        B, U = self._calc_BU(A)
        motif_matries.append(U)
        motif_matries.append(B)

        if self.motif_size < 3:
            return {
                i + 1: adj.multiply(motif_matries[i]).sum() / self.motif_edges_num[i]
                for i in range(len(motif_matries))
            }

        # compute M1 motif adjacency matrix
        C = U.dot(U).multiply(U.transpose())
        M1 = C + C.transpose()
        motif_matries.append(M1)

        # compute M2 motif adjacency matrix
        C = B.dot(U).multiply(U.transpose()) + U.dot(B).multiply(U.transpose()) + U.dot(U).multiply(B)
        M2 = C + C.transpose()
        motif_matries.append(M2)

        # compute M3 motif adjacency matrix
        C = B.dot(B).multiply(U) + B.dot(U).multiply(B) + U.dot(B).multiply(B)
        M3 = C + C.transpose()
        motif_matries.append(M3)

        # compute M4 motif adjacency matrix
        M4 = B.dot(B).multiply(B)
        motif_matries.append(M4)

        # compute M5 motif adjacency matrix
        C = U.dot(U).multiply(U) + U.dot(U.transpose()).multiply(U) + U.transpose().dot(U).multiply(U)
        M5 = C + C.transpose()
        motif_matries.append(M5)

        # compute M6 motif adjacency matrix
        M6 = U.dot(B).multiply(U) + B.dot(U.transpose()).multiply(U.transpose()) + U.transpose().dot(U).multiply(B)
        motif_matries.append(M6)

        # compute M7 motif adjacency matrix
        M7 = U.transpose().dot(B).multiply(U.transpose()) + B.dot(U).multiply(U) + U.dot(U.transpose()).multiply(B)
        motif_matries.append(M7)

        # compute M8 motif adjacency matrix
        M8 = self._M8(A)
        motif_matries.append(M8)

        # compute M9 motif adjacency matrix
        W = lil_matrix(A.shape)
        for i in range(N):
            J1 = (U[i, :] != 0).toarray().flatten().nonzero()[0]
            J2 = (U[:, i] != 0).toarray().flatten().nonzero()[0]
            for j1 in range(len(J1)):
                for j2 in range(len(J2)):
                    k1, k2 = J1[j1], J2[j2]
                    if A[k1, k2] == 0 and A[k2, k1] == 0:
                        W[i, k1] = W[i, k1] + 1
                        W[i, k2] = W[i, k2] + 1
                        W[k1, k2] = W[k1, k2] + 1
        M9 = W + W.transpose()
        motif_matries.append(M9)

        # compute M10 motif adjacency matrix
        M10 = self._M8(A.transpose())
        motif_matries.append(M10)

        # compute M11 motif adjacency matrix
        M11 = self._M11(A)
        motif_matries.append(M11)

        # compute M12 motif adjacency matrix
        M12 = self._M11(A.transpose())
        motif_matries.append(M12)

        # compute M13 motif adjacency matrix
        W = lil_matrix(A.shape)
        for i in range(N):
            J = (B[i, :] != 0).toarray().flatten().nonzero()[0]
            for j1 in range(len(J)):
                for j2 in range(j1 + 1, len(J)):
                    k1, k2 = J[j1], J[j2]
                    if A[k1, k2] == 0 and A[k2, k1] == 0:
                        W[i, k1] = W[i, k1] + 1
                        W[i, k2] = W[i, k2] + 1
                        W[k1, k2] = W[k1, k2] + 1
        M13 = W + W.transpose()
        motif_matries.append(M13)

        if self.motif_size < 4:
            return {
                i + 1: adj.multiply(motif_matries[i]).sum() / self.motif_edges_num[i]
                for i in range(len(motif_matries))
            }

        # compute M_bifan motif adjacency matrix
        A = A.astype(np.bool8).toarray()
        NA = ~A & ~A.transpose()
        W = lil_matrix(A.shape)
        ai, aj = np.triu(NA, 1).nonzero()
        for ind in range(len(ai)):
            x, y = ai[ind], aj[ind]
            xout = (U[x, :] != 0).toarray().flatten().nonzero()[0]
            yout = (U[y, :] != 0).toarray().flatten().nonzero()[0]
            common = set(xout).intersection(set(yout))
            common = list(common)
            nc = len(common)
            for i in range(nc):
                for j in range(i + 1, nc):
                    w, v = common[i], common[j]
                    if NA[w, v] == 1:
                        W[x, y] = W[x, y] + 1
                        W[x, w] = W[x, w] + 1
                        W[x, v] = W[x, v] + 1
                        W[y, w] = W[y, w] + 1
                        W[y, v] = W[y, v] + 1
                        W[w, v] = W[w, v] + 1
        M_bifan = W + W.transpose()
        motif_matries.append(M_bifan)

        return {
            i + 1: adj.multiply(motif_matries[i]).sum() / self.motif_edges_num[i]
            for i in range(len(motif_matries))
        }

    @staticmethod
    def _calc_BU(A):
        B = A.multiply(A.transpose())
        U = A - B
        return B, U

    @staticmethod
    def _M8(A):
        B, U = HighOrderMotifCounter._calc_BU(A)
        W = lil_matrix(A.shape)
        N = A.shape[0]
        for i in range(N):
            J = (U[i, :] != 0).toarray().flatten().nonzero()[0]
            for j1 in range(len(J)):
                for j2 in range(j1 + 1, len(J)):
                    k1, k2 = J[j1], J[j2]
                    if A[k1, k2] == 0 and A[k2, k1] == 0:
                        W[i, k1] = W[i, k1] + 1
                        W[i, k2] = W[i, k2] + 1
                        W[k1, k2] = W[k1, k2] + 1
        return W + W.transpose()

    @staticmethod
    def _M11(A):
        B, U = HighOrderMotifCounter._calc_BU(A)
        W = lil_matrix(A.shape)
        N = A.shape[0]
        for i in range(N):
            J1 = (B[i, :] != 0).toarray().flatten().nonzero()[0]
            J2 = (U[i, :] != 0).toarray().flatten().nonzero()[0]
            for j1 in range(len(J1)):
                for j2 in range(len(J2)):
                    k1, k2 = J1[j1], J2[j2]
                    if A[k1, k2] == 0 and A[k2, k1] == 0:
                        W[i, k1] = W[i, k1] + 1
                        W[i, k2] = W[i, k2] + 1
                        W[k1, k2] = W[k1, k2] + 1
        return W + W.transpose()


if __name__ == '__main__':
    c = HighOrderMotifCounter()
    rlt = c.count([
        {
            'address_from': '0xce1',
            'address_to': '0x38c',
        }, {
            'address_from': '0x38c',
            'address_to': '0xce1',
        }, {
            'address_from': '0x38c',
            'address_to': '0x21b',
        }, {
            'address_from': '0x21b',
            'address_to': '0x38c',
        }, {
            'address_from': '0x38c',
            'address_to': '0xc02',
        }, {
            'address_from': '0xc02',
            'address_to': '0x38c',
        }, {
            'address_from': '0x38c',
            'address_to': '0x0ec',
        }, {
            'address_from': '0x0ec',
            'address_to': '0x38c',
        }, {
            'address_from': '0x0ec',
            'address_to': '0xc02',
        }, {
            'address_from': '0xce1',
            'address_to': '0x0ec',
        }, {
            'address_from': '0x0ec',
            'address_to': '0xce1',
        }
    ])
    print(rlt)