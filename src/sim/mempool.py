import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Union


@dataclass
class SimTx:
    resource_dict: Dict[str, float]
    tx_fee: float = 0.0
    arrival_ts: int = 0
    label: str = None

    def __lt__(self, other):
        # Order of transactions if based on descending transaction fees
        # i.e., highest paying transaction if first
        return self.tx_fee > other.tx_fee


class TransferSimMempool:
    _resource_dict = {
        "Compute": 8500.0,
        "History": 6500.0,
        "Access": 300.0,
        "Bandwidth": 5700.0,
    }

    def get_next_tx(self) -> SimTx:
        return SimTx(self._resource_dict)

    # Need to use the same fingerprint as HistoricalSimMempool
    def refresh(self):
        pass

    def txs_count(self):
        return np.inf


class HistoricalSimMempool:
    def __init__(
        self,
        historical_txs: List[SimTx],
        demand_type: str,
        demand_lambda: float = None,
        block_time: int = None,
    ):
        self.mempool_txs: List[SimTx] = []
        self.historical_txs = historical_txs
        self.demand_type = demand_type
        self.demand_lambda = demand_lambda
        self.block_time = block_time
        self.refresh_times = 0
        if (demand_type == "parametric") & (demand_lambda is None):
            raise ValueError(
                "`demand_lambda` must be set when using `parametric` demand"
            )
        if (demand_type == "historical") & (block_time is None):
            raise ValueError("`block_time` must be set when using `historical` demand")
        if demand_type not in ["infinite", "historical", "parametric"]:
            raise ValueError(
                "`demand_type` can only take the values `infinite`, `historical` or `parametric` "
            )

    def get_next_tx(self) -> Union[SimTx, None]:
        if self.demand_type == "infinite":
            tx = random.choice(self.historical_txs)
            return tx
        else:  # "historical" or "parametric"
            if len(self.mempool_txs) == 0:
                return None
            else:
                tx = self.mempool_txs.pop()
                return tx

    def refresh(self):
        if self.demand_type == "infinite":
            pass
        elif self.demand_type == "historical":
            arr_start = self.refresh_times * self.block_time
            arr_end = (self.refresh_times + 1) * self.block_time
            new_txs = [
                tx for tx in self.historical_txs if arr_start <= tx.arrival_ts < arr_end
            ]
            self.mempool_txs += new_txs
            self.mempool_txs.sort()
        else:  # "parametric"
            tx_sample_size = np.random.poisson(self.demand_lambda)
            new_txs = random.choices(self.historical_txs, k=tx_sample_size)
            self.mempool_txs += new_txs
            self.mempool_txs.sort()
        self.refresh_times + 1

    def txs_count(self):
        if self.demand_type == "infinite":
            return np.inf
        else:  # "historical" or "parametric"
            return len(self.mempool_txs)
