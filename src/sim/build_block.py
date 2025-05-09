import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Union, Tuple
from collections.abc import Callable

sys.path.append(str(Path(__file__).parent.parent))
from sim.mempool import SimTx, TransferSimMempool, HistoricalSimMempool
from sim.meter import one_dim_scheme


def build_block(
    mempool: Union[TransferSimMempool, HistoricalSimMempool],
    meter_func: Callable[[List[SimTx], float], float],
    meter_limit: float,
) -> Tuple[List[SimTx], float]:
    mempool.refresh()
    block_txs = []
    utilization = 0.0
    while True:
        next_tx = mempool.get_next_tx()
        if next_tx is None:  # if mempool is empty, we close the block as is
            break
        candidate_txs = block_txs + [next_tx]
        candidate_utilization = meter_func(candidate_txs, meter_limit)
        if candidate_utilization <= 1:
            block_txs = candidate_txs
            utilization = candidate_utilization
        else:
            break
    return block_txs, utilization


def build_blocks_from_historic_scenario(
    n_iter: int,
    n_blocks: int,
    tx_set: List[SimTx],
    demand_type: str,
    meter_func: Callable[[List[SimTx], float], float],
    meter_limit: float,
    demand_lambda: float = None,
    block_time: int = None,
) -> pd.DataFrame:
    sim_df = pd.DataFrame()
    for iter in tqdm(range(n_iter)):
        mempool = HistoricalSimMempool(tx_set, demand_type, demand_lambda, block_time)
        for i in range(n_blocks):
            block_txs, utilization = build_block(mempool, meter_func, meter_limit)
            block_dict = {
                "iter": iter,
                "block": i,
                "utilization": utilization,
                "gas_used": utilization * meter_limit,
                "one_dim_utilization": one_dim_scheme(block_txs, meter_limit),
                "throughput": len(block_txs),
                "mempool_size": mempool.txs_count(),
            }
            sim_df = pd.concat([sim_df, pd.DataFrame([block_dict])], ignore_index=True)
    return sim_df


def build_block_from_eth_transfer_scenario(
    meter_func: Callable[[List[SimTx], float], float], meter_limit: float
):
    mempool = TransferSimMempool()
    block_txs, utilization = build_block(mempool, meter_func, meter_limit)
    block_dict = {
        "utilization": utilization,
        "gas_used": utilization * meter_limit,
        "one_dim_utilization": one_dim_scheme(block_txs, meter_limit),
        "throughput": len(block_txs),
        "mempool_size": mempool.txs_count(),
    }
    sim_df = pd.DataFrame([block_dict])
    return sim_df
