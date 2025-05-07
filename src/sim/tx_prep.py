import os
import sys
import pickle
import duckdb
import logging
import numpy as np
import pandas as pd
from typing import List
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from mempool import SimTx
from data.gas_cost import compute_component_gas_costs_per_tx
from resource_gas_split import compute_resource_gas_cost_per_tx


def load_agg_trace_df(op_files_dir: str) -> pd.DataFrame:
    # Load parquet files with duckdb
    query = f"""
    SELECT *
    FROM read_parquet(
        '{op_files_dir}', 
        hive_partitioning=True, 
        union_by_name=True
        );
    """
    agg_trace_df = duckdb.connect().execute(query).fetchdf()
    # Drop columns
    agg_trace_df = agg_trace_df.drop(columns=["block_range"])
    # Clean up repeated opcodes
    agg_trace_df["op"] = np.where(
        agg_trace_df["op"].str.startswith("DUP"), "DUP", agg_trace_df["op"]
    )
    agg_trace_df["op"] = np.where(
        agg_trace_df["op"].str.startswith("SWAP"), "SWAP", agg_trace_df["op"]
    )
    agg_trace_df["op"] = np.where(
        (agg_trace_df["op"].str.startswith("PUSH")) & (agg_trace_df["op"] != "PUSH0"),
        "PUSH",
        agg_trace_df["op"],
    )
    # Compute total gas cost for opcode
    agg_trace_df["op_total_gas_cost"] = (
        agg_trace_df["op_gas_cost"] * agg_trace_df["op_gas_pair_count"]
    )
    return agg_trace_df


def process_and_save_sim_txs(
    op_files_dir: str, tx_info_dir: str, sim_txs_file_dir: str
) -> None:
    # Load opcode and tx data
    agg_trace_df = load_agg_trace_df(op_files_dir)
    tx_gas_info_df = pd.read_parquet(tx_info_dir)
    # Compute component gas usage
    comp_df = compute_component_gas_costs_per_tx(agg_trace_df, tx_gas_info_df)
    # Set txs to ignore
    strange_access_txs = comp_df[~comp_df["intrinsic_access_cost"].between(0, 4000000)][
        "tx_hash"
    ].values.tolist()
    fail_txs = tx_gas_info_df[~tx_gas_info_df["is_success"]]["tx_hash"].values.tolist()
    ignore_txs = fail_txs + strange_access_txs
    # Get gas costs by resource
    gas_by_resource_df = compute_resource_gas_cost_per_tx(
        agg_trace_df,
        tx_gas_info_df,
        comp_df,
        ignore_txs,
    )
    # Build sim txs
    tx_hash_list = gas_by_resource_df["tx_hash"].unique().tolist()
    sim_tx_list = []
    for tx_hash in tx_hash_list:
        resource_dict = (
            gas_by_resource_df[gas_by_resource_df["tx_hash"] == tx_hash]
            .drop(columns=["tx_hash", "block_height", "State (exc. Refunds)"])
            .iloc[0]
            .to_dict()
        )
        tx_fee = 0.0  # TODO: add tx fees
        arrival_time = 0.0  # TODO: add arrival times
        sim_tx = SimTx(resource_dict, tx_fee, arrival_time, tx_hash)
        sim_tx_list.append(sim_tx)
    # Save sim txs as pickle
    with open(sim_txs_file_dir, "wb") as f:
        pickle.dump(sim_tx_list, f)


def get_sim_txs(op_files_dir: str, tx_info_dir: str, sim_txs_dir: str) -> List[SimTx]:
    # if the file does not exist, process data and save
    if not os.path.isfile(sim_txs_dir):
        logging.info(
            "Historical transactions for simulation are not yet processed. Processing now."
        )
        process_and_save_sim_txs(op_files_dir, tx_info_dir, sim_txs_dir)
    # load list from pickle file
    logging.info("Loading historical transactions for simulation.")
    with open(sim_txs_dir, "rb") as f:
        sim_tx_list = pickle.load(f)
    return sim_tx_list
