import os
import sys
import json
import logging
import argparse
import itertools
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from sim.build_block import (
    build_block_from_eth_transfer_scenario,
    build_blocks_from_historic_scenario,
)
from sim.tx_prep import get_sim_txs
import sim.meter as meter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_configuration():
    """
    Parses command line arguments and secrets, and returns a configuration dictionary.
    """
    file_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Runs a set of simulations to test metering schemes under different demand scenarios"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.abspath(os.path.join(file_dir, "..", "..", "data")),
        help="Data directory (default: ./data). Parquet files will be stored in a folder there.",
    )
    parser.add_argument(
        "--reprocess",
        type=bool,
        default=False,
        help=(
            "Whether to reprocess the transaction set for the historical transaction "
            "scenarios (default: False)"
        ),
    )
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=6000,
        help="Number of blocks built in the simulation (default: 6000)",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=50,
        help=(
            "Number of Monte-Carlo iterations for the historical transactions"
            "simulation (default: 50)"
        ),
    )
    parser.add_argument(
        "--secrets_path",
        type=str,
        default=os.path.abspath(os.path.join(file_dir, "..", "..", "secrets.json")),
        help="Path to secrets.json file (default: ./secrets.json)",
    )
    parser.add_argument(
        "--xatu_username",
        type=str,
        help="Xatu Clickhouse username (can be provided in secrets.json)",
    )
    parser.add_argument(
        "--xatu_password",
        type=str,
        help="Xatu Clickhouse password (can be provided in secrets.json)",
    )
    parser.add_argument(
        "--thread_pool_size",
        type=int,
        default=8,
        help="Number of threads to use for processing transactions (default: 8)",
    )
    parser.add_argument(
        "--tx_batch_size",
        type=int,
        default=20,
        help=(
            "Number of transactions that we tentatively add to the block at each iteration."
            "Used to speed up run time. (default: 10)"
        ),
    )
    args = parser.parse_args()
    config = {
        "data_dir": args.data_dir,
        "reprocess": args.reprocess,
        "n_blocks": args.n_blocks,
        "n_iter": args.n_iter,
        "thread_pool_size": args.thread_pool_size,
        "tx_batch_size": args.tx_batch_size,
    }
    # Load secrets form file, if it exists
    if os.path.isfile(args.secrets_path):
        with open(args.secrets_path, "r") as file:
            secrets_dict = json.load(file)
    # If not, secrets will be defined by xatu_username and xatu_password
    else:
        logging.warning(
            f"Secrets file not found at {config['secrets_path']}."
            f"Secrets might be missing if not provided via command line."
        )
        secrets_dict = {
            "xatu_username": args.xatu_username,
            "xatu_password": args.xatu_password,
        }
    if (
        args.xatu_username
    ):  # if xatu_username is provided, it will override secrets_dict
        secrets_dict["xatu_username"] = args.xatu_username
    if (
        args.xatu_password
    ):  # if xatu_password is provided, it will override secrets_dict
        secrets_dict["xatu_password"] = args.xatu_password
    config["secrets_dict"] = secrets_dict
    return config


def main():
    config = parse_configuration()
    # Directories
    data_dir = config["data_dir"]
    op_files_dir = os.path.join(data_dir, "aggregated_opcodes_v3", "*", "file.parquet")
    sim_dir = os.path.join(data_dir, "sim")
    sim_txs_dir = os.path.join(sim_dir, "sim_txs_22000000_22006000.pickle")
    # Set metering schemes and other variables
    meter_fn_list = [
        meter.one_dim_scheme,
        meter.compute_vs_others,
        meter.state_vs_others,
        meter.access_vs_others,
        meter.bandwidth_vs_others,
        meter.state_vs_compute_vs_others,
        meter.state_vs_compute_vs_access_vs_others,
    ]
    meter_limit_list = [36_000_000.0, 36_000_000.0 * 0.5]
    tx_batch_size = config["tx_batch_size"]
    # Run ETH transfers simulation
    logging.info("Running simulation for the ETH transfers scenario.")
    eth_transfer_sim_df = pd.DataFrame()
    for meter_func, meter_limit in itertools.product(meter_fn_list, meter_limit_list):
        iter_sim_df = build_block_from_eth_transfer_scenario(
            meter_func, meter_limit, tx_batch_size
        )
        iter_sim_df["meter_scheme"] = meter_func.__name__
        iter_sim_df["limit"] = meter_limit
        eth_transfer_sim_df = pd.concat([eth_transfer_sim_df, iter_sim_df])
    out_file = os.path.join(sim_dir, "eth_transfer_sim_results.csv")
    os.makedirs(sim_dir, exist_ok=True)
    eth_transfer_sim_df.to_csv(out_file, index=False)
    # Run historical transaction simulations
    tx_set = get_sim_txs(
        op_files_dir, sim_txs_dir, config["secrets_dict"], config["reprocess"]
    )
    demand_dict_list = [
        {"demand_type": "infinite", "demand_lambda": None},
        {"demand_type": "historical", "demand_lambda": None},
        {"demand_type": "parametric", "demand_lambda": 155},
        {"demand_type": "parametric", "demand_lambda": 155 * 2},
        {"demand_type": "parametric", "demand_lambda": 155 * 5},
    ]
    block_time = 12
    for demand_dict in demand_dict_list:
        historical_sim_df = pd.DataFrame()
        demand_type = demand_dict["demand_type"]
        demand_lambda = demand_dict["demand_lambda"]
        logging.info(
            f"Running simulation for the historical transactions scenarios with "
            f"demand_type={demand_type} & demand_lambda={demand_lambda}."
        )
        for meter_func, meter_limit in itertools.product(
            meter_fn_list, meter_limit_list
        ):
            logging.info(
                f"Monte-Carlo run for meter_func={meter_func.__name__}"
                f" & meter_limit={int(meter_limit)}."
            )
            iter_sim_df = build_blocks_from_historic_scenario(
                config["n_iter"],
                config["n_blocks"],
                tx_set,
                demand_type,
                meter_func,
                meter_limit,
                demand_lambda,
                block_time,
                config["thread_pool_size"],
                tx_batch_size,
            )
            iter_sim_df["demand_type"] = demand_type
            iter_sim_df["demand_lambda"] = demand_lambda
            iter_sim_df["meter_scheme"] = meter_func.__name__
            iter_sim_df["limit"] = meter_limit
            historical_sim_df = pd.concat([historical_sim_df, iter_sim_df])
        out_file = os.path.join(
            sim_dir,
            f"historical_txs_sim_results_demand={demand_type}_lambda={demand_lambda}.csv",
        )
        os.makedirs(sim_dir, exist_ok=True)
        historical_sim_df.to_csv(out_file, index=False)


if __name__ == "__main__":
    main()
