import os
import sys
import duckdb
import logging
import argparse
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from data.path_mng import chunks, get_parquet_path_patterns


pd.options.mode.chained_assignment = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def compute_gas_cost_for_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a new column to the raw debug trace data (df) with the correct gas cost for the CALL-type opcodes.
    The dataframe can have multiple transactions.
    """
    unique_txs = df["tx_hash"].unique()
    new_df = pd.DataFrame()
    for tx_hash in unique_txs:
        tx_df = df[df["tx_hash"] == tx_hash]
        try:
            new_tx_df = compute_gas_costs_for_single_tx(tx_df)
        except:
            logging.error(f"Error at transaction: {tx_hash}")
            logging.error("Traceback:")
            traceback.print_exc()
            new_tx_df = tx_df.copy()
            new_tx_df["op_gas_cost"] = new_tx_df["gasCost"]
            logging.error(
                "The column op_gas_cost will be set to original gasCost. Continuing to process chunk."
            )
        new_df = pd.concat([new_df, new_tx_df], ignore_index=True)
    return new_df


def compute_gas_costs_for_single_tx(initial_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a new column to the raw debug trace data (df) with the correct gas cost for the CALL-type opcodes.
    Note that this code assume df only contains data for one transaction
    """
    df = initial_df.copy()
    if df["depth"].nunique() == 1:  # if transaction does not have depth changes
        # We only need to fix calls
        df["op_gas_cost"] = np.where(
            df["op"].isin(["DELEGATECALL", "STATICCALL", "CALL", "CALLCODE"]),
            df["gas"] - df["gas"].shift(-1),
            df["gasCost"],
        )
        return df
    else:  # if the transaction has depth changes...
        df["has_depth_increase"] = df["depth"] < df["depth"].shift(-1)
        df["has_depth_decrease"] = df["depth"] > df["depth"].shift(-1)
        df["has_depth_change"] = df["depth"] != df["depth"].shift(-1)
        inner_gas_diff_list = []
        outer_gas_diff_list = []
        change_depth_row_numbers = df[df["has_depth_change"]]["file_row_number"].values
        filter_row_numbers = np.unique(
            np.concat(
                (
                    change_depth_row_numbers,
                    change_depth_row_numbers + 1,
                    change_depth_row_numbers - 1,
                )
            )
        )
        filter_rows_df = df[df["file_row_number"].isin(filter_row_numbers)].sort_values(
            "file_row_number"
        )
        for i in range(len(filter_rows_df) - 1):
            row = filter_rows_df.iloc[i]
            next_row = filter_rows_df.iloc[i + 1]
            # if the row is a new depth starter
            if row["has_depth_increase"]:
                # Get the next row that decreases the depth
                return_row = filter_rows_df[
                    (filter_rows_df["file_row_number"] > row["file_row_number"])
                    & (filter_rows_df["has_depth_decrease"])
                    & (filter_rows_df["depth"] == row["depth"] + 1)
                ].iloc[0]
                # Get and save the available gas at entry and exit
                ## inside the call
                inner_entry_gas = next_row["gas"]
                inner_exit_gas = filter_rows_df[
                    filter_rows_df["file_row_number"] == return_row["file_row_number"]
                ].iloc[0]["gas"]
                inner_gas_diff_list.append(inner_entry_gas - inner_exit_gas)
                ## right outside the call
                outer_entry_gas = row["gas"]
                outer_exit_gas = filter_rows_df[
                    filter_rows_df["file_row_number"]
                    == return_row["file_row_number"] + 1
                ].iloc[0]["gas"]
                outer_gas_diff_list.append(outer_entry_gas - outer_exit_gas)
            else:
                inner_gas_diff_list.append(np.nan)
                outer_gas_diff_list.append(np.nan)
        # Compute the correct cost for the depth starter rows
        depth_starter_cost_df = pd.DataFrame(
            {
                "file_row_number": filter_rows_df["file_row_number"].values,
                "op_gas_cost": np.array(outer_gas_diff_list + [np.nan])
                - np.array(inner_gas_diff_list + [np.nan]),
            }
        )
        # Merge with original df
        df = df.merge(depth_starter_cost_df, on="file_row_number", how="left").drop(
            columns=["has_depth_change", "has_depth_increase", "has_depth_decrease"]
        )
        df["op_gas_cost"] = np.where(
            df["op_gas_cost"].isna(), df["gasCost"], df["op_gas_cost"]
        )
        # Now, we only need to fix calls of same depth
        df["op_gas_cost"] = np.where(
            (df["op"].isin(["DELEGATECALL", "STATICCALL", "CALL", "CALLCODE"]))
            & (df["depth"] == df["depth"].shift(-1)),
            df["gas"] - df["gas"].shift(-1),
            df["op_gas_cost"],
        )
        return df


def parse_configuration():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Loads data from EVM debug traces calculates the correct gas costs and aggregates the results."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.abspath(os.path.join(src_dir, "..", "..", "data")),
        help="Data directory (default: ./data). Used for both input and output.",
    )
    parser.add_argument(
        "--reprocess",
        type=bool,
        default=False,
        help="Whether to reprocessed blocks. Default is False.",
    )
    args = parser.parse_args()
    config = {
        "data_dir": args.data_dir,
        "reprocess": args.reprocess,
    }
    return config


def main():
    config = parse_configuration()
    # Directories
    data_dir = config["data_dir"]
    block_data_dir = os.path.join(data_dir, "block_data")
    block_dirs = get_parquet_path_patterns(block_data_dir)
    # block_dirs now contains paths to parquet files in valid block_height directories
    chunk_size = 5
    block_dirs_chunks = list(chunks(block_dirs, chunk_size))
    total_chunks = len(block_dirs_chunks)
    for dirs_chunk in tqdm(
        block_dirs_chunks, total=total_chunks, desc="Processing chunks"
    ):
        # Define output directory path
        start_block_height = dirs_chunk[0].split("/")[-3].split("=")[-1]
        end_block_height = dirs_chunk[-1].split("/")[-3].split("=")[-1]
        block_range = f"{start_block_height}...{end_block_height}"
        output_dir = os.path.join(
            data_dir,
            "aggregated_opcodes_v2",
            f"block_range={block_range}",
        )
        output_file_path = os.path.join(output_dir, "file.parquet")
        if os.path.isfile(output_file_path) and config["reprocess"] == False:
            logging.info(f"Block range {block_range} is already processed. Skipping...")
            continue
        logging.info(f"Start processing block range {block_range} ...")
        # Define query
        query = f"""
        SELECT block_height,
            tx_hash,
            file_row_number,
            op,
            gas,
            gasCost,
            depth
        FROM read_parquet(
                { dirs_chunk },
                hive_partitioning = TRUE,
                filename = True,
                file_row_number = True,
                union_by_name = True
            );
        """
        # Execute the query and convert the result to a DataFrame
        raw_df = duckdb.query(query).to_df()
        # Fix issues with gas costs
        clean_df = compute_gas_cost_for_chunk(raw_df)
        # Aggregate data for memory efficiency
        df = (
            clean_df.groupby(["block_height", "tx_hash", "op", "op_gas_cost"])
            .size()
            .reset_index()
        )
        df.columns = [
            "block_height",
            "tx_hash",
            "op",
            "op_gas_cost",
            "op_gas_pair_count",
        ]
        # Save DataFrame to parquet
        os.makedirs(output_dir, exist_ok=True)
        df.to_parquet(output_file_path)


if __name__ == "__main__":
    main()
