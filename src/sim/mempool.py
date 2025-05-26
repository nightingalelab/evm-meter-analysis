import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Union
from scipy.stats import gaussian_kde


@dataclass
class SimTx:
    resource_dict: Dict[str, float]
    tx_fee: float = 0.0
    arrival_ts: int = 0
    label: str = None


class TransferSimMempool:
    _resource_dict = {
        "Compute": 8500.0,
        "History": 6500.0,
        "Access": 300.0,
        "Bandwidth": 5700.0,
    }

    def get_next_tx(self) -> SimTx:
        return SimTx(self._resource_dict)

    def get_next_tx_batch(self, batch_size: int) -> List[SimTx]:
        return [SimTx(self._resource_dict)] * batch_size

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
        demand_base_kernel: gaussian_kde = None,
        demand_mul: float = None,
        block_time: int = None,
    ):
        self.mempool_txs: List[SimTx] = []
        self.historical_txs = historical_txs
        self.historical_txs_by_slot: List[List[SimTx]] = []
        self.demand_type = demand_type
        self.demand_base_kernel = demand_base_kernel
        self.demand_mul = demand_mul
        self.demand_adj = None
        self.block_time = block_time
        self.refresh_times = 0
        if (demand_type == "parametric") & (demand_mul is None):
            raise ValueError("`demand_mul` must be set when using `parametric` demand")
        if (demand_type == "parametric") & (demand_base_kernel is None):
            raise ValueError(
                "`demand_base_kernel` must be set when using `parametric` demand"
            )
        if (demand_type == "historical") & (block_time is None):
            raise ValueError("`block_time` must be set when using `historical` demand")
        if demand_type not in ["infinite", "historical", "parametric"]:
            raise ValueError(
                "`demand_type` can only take the values `infinite`, `historical` or `parametric` "
            )
        if demand_type == "historical":
            self._group_historical_txs_by_slot()
        if demand_type == "parametric":
            average = self.demand_base_kernel.resample(size=10000).mean()
            self.demand_adj = average * (demand_mul - 1)

    def get_next_tx(self) -> Union[SimTx, None]:
        if self.demand_type == "infinite":
            tx = random.choice(self.historical_txs)
            return tx
        else:  # "historical" or "parametric"
            if len(self.mempool_txs) == 0:
                return None
            else:
                tx = self.mempool_txs.pop(0)
                return tx

    def get_next_tx_batch(self, batch_size: int) -> List[SimTx]:
        if self.demand_type == "infinite":
            tx_batch = [random.choice(self.historical_txs) for i in range(batch_size)]
            return tx_batch
        else:  # "historical" or "parametric"
            if len(self.mempool_txs) == 0:
                return []
            elif len(self.mempool_txs) < batch_size:
                tx_batch = self.mempool_txs
                self.mempool_txs = []
                return tx_batch
            else:
                tx_batch = self.mempool_txs[:batch_size]
                self.mempool_txs = self.mempool_txs[batch_size:]
                return tx_batch

    def refresh(self):
        if self.demand_type == "infinite":
            pass
        elif self.demand_type == "historical":
            slot = self.refresh_times
            new_txs = self.historical_txs_by_slot[slot]
            self.mempool_txs = sorted(
                self.mempool_txs + new_txs, key=lambda tx: tx.tx_fee, reverse=True
            )
        else:  # "parametric"
            tx_sample_size = int(
                self.demand_base_kernel.resample(size=1)[0][0] + self.demand_adj
            )
            new_txs = random.choices(self.historical_txs, k=tx_sample_size)
            self.mempool_txs = sorted(
                self.mempool_txs + new_txs, key=lambda tx: tx.tx_fee, reverse=True
            )
        self.refresh_times += 1

    def txs_count(self):
        if self.demand_type == "infinite":
            return np.inf
        else:  # "historical" or "parametric"
            return len(self.mempool_txs)

    def _group_historical_txs_by_slot(self):
        slot = 0
        slot_list = []
        for tx in self.historical_txs:
            # Note: assumes historical_txs are sorted by arrival time!!!
            tx_slot = int(np.floor(tx.arrival_ts / self.block_time))
            if tx_slot == slot:
                slot_list.append(tx)
            else:
                self.historical_txs_by_slot.append(slot_list)
                slot += 1
                slot_list = [tx]
