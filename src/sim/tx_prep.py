import os
import sys
import pickle
import duckdb
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from datetime import timedelta
from sqlalchemy import text, create_engine


sys.path.append(str(Path(__file__).parent.parent))
from sim.mempool import SimTx
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


def get_tx_info_df(
    start_block: int, end_block: int, secrets_dict: Dict[str, str]
) -> pd.DataFrame:
    # Download data from Xatu
    tx_gas_df = download_tx_gas_data(start_block, end_block, secrets_dict)
    block_times_df = download_block_times_data(start_block, end_block, secrets_dict)
    start_time = block_times_df["block_time"].min() - timedelta(hours=24)
    end_time = block_times_df["block_time"].max() + timedelta(hours=1)
    arrival_times_df = download_tx_arrival_times_data(
        start_time, end_time, secrets_dict
    )
    # Join datasets
    df = tx_gas_df.merge(block_times_df, on="block_height", how="left")
    df = df.merge(arrival_times_df, on="tx_hash", how="left")
    # Replace empty arrival times with block time
    df["arrival_time"] = np.where(
        df["arrival_time"].isna(), df["block_time"], df["arrival_time"]
    )
    # Convert arrival_time to seconds starting from the first block
    # Earlier arrivals are assigned to zero
    df["block_ts"] = df["block_time"].apply(lambda dt: int(np.floor(dt.timestamp())))
    df["arrival_ts"] = df["arrival_time"].apply(
        lambda dt: int(np.floor(dt.timestamp()))
    )
    df["arrival_ts"] = np.maximum(0, df["arrival_ts"] - df["block_ts"].min())
    return df


def download_tx_gas_data(
    start_block: int, end_block: int, secrets_dict: Dict[str, str]
) -> pd.DataFrame:
    # Credentials for xatu clickhouse
    xatu_user = secrets_dict["xatu_username"]
    xatu_pass = secrets_dict["xatu_password"]
    # Define SQL query
    query = text(
        """
        SELECT 
            block_number AS block_height, 
            transaction_hash AS tx_hash, 
            gas_used AS tx_gas_cost, 
            gas_limit AS tx_gas_limit,
            n_input_zero_bytes AS tx_input_zero_bytes,
            n_input_nonzero_bytes AS tx_input_nonzero_bytes,
            4 * n_input_zero_bytes + 16 * n_input_nonzero_bytes AS tx_input_data_cost,
            to_address IS NULL AS is_contract_creation,
            success = true AS is_success,
            max_fee_per_gas
        FROM default.canonical_execution_transaction
        WHERE block_number BETWEEN toUInt64(:start_block) AND toUInt64(:end_block)
                AND meta_network_name = :network
        ORDER BY block_number ASC, transaction_index ASC
    """
    )
    # Run query
    db_url = f"clickhouse+http://{xatu_user}:{xatu_pass}@clickhouse.xatu.ethpandaops.io:443/default?protocol=https"
    engine = create_engine(db_url)
    connection = engine.connect()
    query_result = connection.execute(
        query,
        {
            "start_block": start_block,
            "end_block": end_block,
            "network": "mainnet",
        },
    )
    # Fecth query result to pandas
    df = pd.DataFrame(query_result.fetchall())
    # Transform booleans to correct type
    df["is_contract_creation"] = df["is_contract_creation"].astype(bool)
    df["is_success"] = df["is_success"].astype(bool)
    return df


def download_block_times_data(
    start_block: int, end_block: int, secrets_dict: Dict[str, str]
):
    # Credentials for xatu clickhouse
    xatu_user = secrets_dict["xatu_username"]
    xatu_pass = secrets_dict["xatu_password"]
    # Define SQL query
    query = text(
        """
        SELECT 
            block_number AS block_height, 
            block_date_time AS block_time
        FROM default.canonical_execution_block FINAL
        WHERE block_number BETWEEN toUInt64(:start_block) AND toUInt64(:end_block)
            AND meta_network_name = :network
    """
    )
    # Run query
    db_url = f"clickhouse+http://{xatu_user}:{xatu_pass}@clickhouse.xatu.ethpandaops.io:443/default?protocol=https"
    engine = create_engine(db_url)
    connection = engine.connect()
    query_result = connection.execute(
        query,
        {
            "start_block": start_block,
            "end_block": end_block,
            "network": "mainnet",
        },
    )
    # Fecth query result to pandas
    df = pd.DataFrame(query_result.fetchall())
    return df


def download_tx_arrival_times_data(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    secrets_dict: Dict[str, str],
):
    # Credentials for xatu clickhouse
    xatu_user = secrets_dict["xatu_username"]
    xatu_pass = secrets_dict["xatu_password"]
    # Define SQL query
    query = text(
        """
        SELECT 
            hash AS tx_hash, 
            MAX(event_date_time) AS arrival_time
        FROM default.mempool_transaction FINAL
        WHERE event_date_time BETWEEN :start_time AND :end_time
            AND meta_network_name = 'mainnet'
        GROUP BY hash;
    """
    )
    # Run query
    db_url = f"clickhouse+http://{xatu_user}:{xatu_pass}@clickhouse.xatu.ethpandaops.io:443/default?protocol=https"
    engine = create_engine(db_url)
    connection = engine.connect()
    query_result = connection.execute(
        query,
        {"start_time": start_time, "end_time": end_time, "network": "mainnet"},
    )
    # Fecth query result to pandas
    df = pd.DataFrame(query_result.fetchall())
    return df


def process_and_save_sim_txs(
    op_files_dir: str,
    sim_txs_dir: str,
    secrets_dict: Dict[str, str],
) -> None:
    # Load opcode and tx data
    logging.info("Loading aggregated traces.")
    agg_trace_df = load_agg_trace_df(op_files_dir)
    start_block = int(agg_trace_df["block_height"].min())
    end_block = int(agg_trace_df["block_height"].max())
    logging.info("Downloading additional transaction info.")
    tx_info_df = get_tx_info_df(start_block, end_block, secrets_dict)
    # Compute component gas usage
    logging.info("Computing gas costs by component.")
    comp_df = compute_component_gas_costs_per_tx(agg_trace_df, tx_info_df)
    # Set txs to ignore
    strange_access_txs = comp_df[~comp_df["intrinsic_access_cost"].between(0, 4000000)][
        "tx_hash"
    ].values.tolist()
    fail_txs = tx_info_df[~tx_info_df["is_success"]]["tx_hash"].values.tolist()
    ignore_txs = fail_txs + strange_access_txs
    # Get gas costs by resource
    logging.info("Computing gas costs by resource.")
    gas_by_resource_df = compute_resource_gas_cost_per_tx(
        agg_trace_df,
        tx_info_df,
        comp_df,
        ignore_txs,
    )
    # Build sim txs
    logging.info("Building list of simTxs.")
    tx_hash_list = gas_by_resource_df["tx_hash"].unique().tolist()
    sim_tx_list = []
    for tx_hash in tqdm(tx_hash_list):
        resource_dict = (
            gas_by_resource_df[gas_by_resource_df["tx_hash"] == tx_hash]
            .drop(columns=["tx_hash", "block_height", "State (exc. Refunds)"])
            .iloc[0]
            .to_dict()
        )
        tx_info = tx_info_df[tx_info_df["tx_hash"] == tx_hash].iloc[0]
        tx_fee = tx_info["max_fee_per_gas"] * tx_info["tx_gas_cost"]
        arrival_ts = tx_info["arrival_ts"]
        sim_tx = SimTx(resource_dict, tx_fee, arrival_ts, tx_hash)
        sim_tx_list.append(sim_tx)
    # Sort by arrival time
    sorted_sim_tx_list = sorted(sim_tx_list, key=lambda tx: tx.arrival_ts)
    # Save sorted_sim_tx_list as pickle
    with open(sim_txs_dir, "wb") as f:
        pickle.dump(sorted_sim_tx_list, f)
    logging.info("Sucessfully pickled list of simTxs.")


def get_sim_txs(
    op_files_dir: str, sim_txs_dir: str, secrets_dict: Dict[str, str], reprocess: bool
) -> List[SimTx]:
    # if the file does not exist, process data and save
    if reprocess:
        logging.info(
            "Reprocess flag set to True. Reprocessing transactions for the simulation."
        )
        process_and_save_sim_txs(op_files_dir, sim_txs_dir, secrets_dict)
    elif not os.path.isfile(sim_txs_dir):
        logging.info(
            "Historical transactions for simulation are not yet processed. Processing now."
        )
        process_and_save_sim_txs(op_files_dir, sim_txs_dir, secrets_dict)
    # load list from pickle file
    logging.info("Loading historical transactions for simulation.")
    with open(sim_txs_dir, "rb") as f:
        sim_tx_list = pickle.load(f)
    return sim_tx_list
