import sys
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
from scipy.stats import gaussian_kde
from typing import List, Union, Tuple
from collections.abc import Callable

sys.path.append(str(Path(__file__).parent.parent))
from sim.mempool import SimTx, TransferSimMempool, HistoricalSimMempool
from sim.meter import one_dim_scheme


def build_block(
    mempool: Union[TransferSimMempool, HistoricalSimMempool],
    meter_func: Callable[[List[SimTx], float], float],
    meter_limit: float,
    tx_batch_size: int,
) -> Tuple[List[SimTx], float]:
    mempool.refresh()
    block_txs = []
    utilization = 0.0
    while True:
        next_tx_batch = mempool.get_next_tx_batch(tx_batch_size)
        # if mempool is empty, we close the block as is
        if len(next_tx_batch) == 0:
            break
        # if not, then:
        candidate_txs = block_txs + next_tx_batch
        candidate_utilization = meter_func(candidate_txs, meter_limit)
        # if batch does not fill the block, we add it and continue
        if candidate_utilization < 1:
            block_txs = candidate_txs
            utilization = candidate_utilization
        # if the batch fills the block,
        # we remove candidate txs until it is about to be full
        else:
            for i in range(tx_batch_size, 0, -1):
                candidate_txs = block_txs + next_tx_batch[:i]
                candidate_utilization = meter_func(candidate_txs, meter_limit)
                if candidate_utilization <= 1:
                    block_txs = candidate_txs
                    utilization = candidate_utilization
                    break
            break
    return block_txs, utilization


def build_blocks_from_historic_scenario(
    n_iter: int,
    n_blocks: int,
    tx_set: List[SimTx],
    demand_type: str,
    meter_func: Callable[[List[SimTx], float], float],
    meter_limit: float,
    demand_base_kernel: gaussian_kde,
    demand_lambda: float = None,
    block_time: int = None,
    tx_batch_size: int = 20,
    thread_pool_size: int = 8,
) -> pd.DataFrame:
    sim_df = pd.DataFrame()
    # Historical demand scnerio only needs 1 iteration
    if demand_type == "historical":
        sim_df = _get_historic_scenario_blocks_for_iter(
            1,
            n_blocks,
            tx_set,
            demand_type,
            meter_func,
            meter_limit,
            demand_base_kernel,
            demand_lambda,
            block_time,
            tx_batch_size,
        )
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=thread_pool_size
        ) as executor:
            futures = [
                executor.submit(
                    _get_historic_scenario_blocks_for_iter,
                    iter,
                    n_blocks,
                    tx_set,
                    demand_type,
                    meter_func,
                    meter_limit,
                    demand_base_kernel,
                    demand_lambda,
                    block_time,
                    tx_batch_size,
                )
                for iter in range(n_iter)
            ]
            df_list = []
            for future in tqdm(concurrent.futures.as_completed(futures), total=n_iter):
                df_list.append(future.result())
            sim_df = pd.concat(df_list, ignore_index=True).sort_values(
                ["iter", "block"]
            )

    return sim_df


def _get_historic_scenario_blocks_for_iter(
    iter: int,
    n_blocks: int,
    tx_set: List[SimTx],
    demand_type: str,
    meter_func: Callable[[List[SimTx], float], float],
    meter_limit: float,
    demand_base_kernel: gaussian_kde,
    demand_lambda: float = None,
    block_time: int = None,
    tx_batch_size: int = 20,
):
    mempool = HistoricalSimMempool(
        tx_set,
        demand_type,
        demand_base_kernel,
        demand_lambda,
        block_time,
    )
    df = pd.DataFrame()
    for block in range(n_blocks):
        block_txs, utilization = build_block(
            mempool, meter_func, meter_limit, tx_batch_size
        )
        one_dim_utilization = one_dim_scheme(block_txs, meter_limit)
        block_df = pd.DataFrame(
            [
                {
                    "iter": iter,
                    "block": block,
                    "utilization": utilization,
                    "gas_used": one_dim_utilization * meter_limit,
                    "one_dim_utilization": one_dim_utilization,
                    "throughput": len(block_txs),
                    "mempool_size": mempool.txs_count(),
                }
            ]
        )
        df = pd.concat([df, block_df], ignore_index=True)
    return df


def build_block_from_eth_transfer_scenario(
    meter_func: Callable[[List[SimTx], float], float],
    meter_limit: float,
    tx_batch_size: int = 20,
):
    mempool = TransferSimMempool()
    block_txs, utilization = build_block(
        mempool, meter_func, meter_limit, tx_batch_size
    )
    one_dim_utilization = one_dim_scheme(block_txs, meter_limit)
    block_dict = {
        "utilization": utilization,
        "gas_used": one_dim_utilization * meter_limit,
        "one_dim_utilization": one_dim_utilization,
        "throughput": len(block_txs),
        "mempool_size": mempool.txs_count(),
    }
    sim_df = pd.DataFrame([block_dict])
    return sim_df
