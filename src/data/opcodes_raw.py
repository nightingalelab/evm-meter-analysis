import os
import re
import gc
import sys
import json
import logging
import argparse
import requests
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from typing import List
from pathlib import Path
from sqlalchemy import create_engine, pool, text

sys.path.append(str(Path(__file__).parent.parent))
from data.tracer import CUSTOM_TRACER

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class XatuClickhouse:
    def __init__(self, db_url, pool_size=5, max_overflow=10, pool_timeout=30):
        self.db_engine = create_engine(
            db_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            poolclass=pool.QueuePool,
        )

    def get_tx_hashes_for_block(
        self,
        block_height: int,
    ) -> list:
        logging.debug(f"Fetching transaction hashes for block {block_height}")
        query = text(
            """
            SELECT transaction_hash
            FROM default.canonical_execution_transaction
            WHERE block_number BETWEEN toUInt64(:start_block) AND toUInt64(:end_block)
                  AND meta_network_name = :network
            ORDER BY block_number ASC, transaction_index ASC
        """
        )
        with self.db_engine.connect() as connection:
            logging.debug(f"Connected to Clickhouse for block {block_height}")
            query_result = connection.execute(
                query,
                {
                    "start_block": block_height,
                    "end_block": block_height,
                    "network": "mainnet",
                },
            )
            logging.debug(f"Query executed for block {block_height}")
            transaction_hashes = [row[0] for row in query_result.fetchall()]
            logging.debug(
                f"Fetched {len(transaction_hashes)} transaction hashes for block {block_height}"
            )
            return transaction_hashes


class ErigonRPC:
    def __init__(
        self,
        erigon_rpc_url: str,
        erigon_rpc_user: str,
        erigon_rpc_pass: str,
        erigon_rpc_response_max_size: int,
    ):
        self.erigon_rpc_url = erigon_rpc_url
        self.erigon_rpc_session = requests.Session()
        self.erigon_rpc_session.auth = (erigon_rpc_user, erigon_rpc_pass)
        self.erigon_rpc_response_max_size = erigon_rpc_response_max_size

    class ResponseTooLargeError(Exception):
        """Custom exception raised when response size exceeds the limit."""

    def _fetch_rpc_response(self, payload: dict) -> str | None:
        """Fetches RPC response from Erigon with error handling and size limits."""
        logging.debug(f"Fetching RPC response with payload: {payload}")
        try:
            response = self.erigon_rpc_session.post(
                self.erigon_rpc_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                stream=True,
            )
            logging.debug(f"RPC request sent to {self.erigon_rpc_url}")
            response.raise_for_status()
            logging.debug("RPC response status OK")
            size = 0
            response_str = ""
            for chunk in response.iter_content(16384):
                size += len(chunk)
                if size > self.erigon_rpc_response_max_size:
                    raise self.ResponseTooLargeError(
                        f"Response size exceeded {self.erigon_rpc_response_max_size} bytes"
                    )
                decoded_chunk = chunk.decode("utf-8", errors="ignore")
                response_str += decoded_chunk
            logging.debug("RPC response fetched and processed successfully")
            return response_str
        except self.ResponseTooLargeError as e:
            logging.error(f"Response too large: {e}")
            return None
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Request exception: {type(e)}: {e}")
            return None

    def _process_trace_response(self, response_str: str) -> list | None:
        """Processes the RPC response string to extract transaction traces."""
        logging.debug("Processing trace response string")
        try:
            trace = json.loads(response_str)
            logging.debug("JSON response loaded successfully")
            if "result" in trace and "data" in trace["result"]:
                data = trace["result"]["data"]
                del trace
                logging.debug("data extracted")
                return data
            else:
                logging.warning("No 'data' found in RPC response.")
                del trace
                return None
        except json.JSONDecodeError:
            logging.error("Failed to decode JSON response.")
            return None

    def fetch_transaction_traces(
        self,
        tx_hash: str,
        block_height: int = 0,
    ) -> list | None:
        """Fetches transaction trace from Erigon RPC."""
        logging.debug(
            f"Fetching transaction trace for tx_hash: {tx_hash}, block_height: {block_height}"
        )
        payload = {
            "jsonrpc": "2.0",
            "method": "debug_traceTransaction",
            "params": [tx_hash, {"tracer": CUSTOM_TRACER}],
            "id": 1,
        }
        response_str = self._fetch_rpc_response(payload)
        if not response_str:
            logging.error(
                f"Transaction trace failed for block {block_height}, tx {tx_hash} due to RPC error."
            )
            return [
                {
                    "op": "RPC_FETCH_ERROR",
                    "gas": 0,
                    "gas_cost": 0,
                    "depth": 0,
                    "memory_expansion": 0,
                    "memory_size": 0,
                    "cum_refund": 0,
                    "call_address": "",
                }
            ]
        else:
            data = self._process_trace_response(response_str)
            if data:
                logging.info(f"Trace fetched for block {block_height}, tx {tx_hash}")
                return data
            else:
                return [
                    {
                        "op": "NO_TRACE",
                        "gas": 0,
                        "gas_cost": 0,
                        "depth": 0,
                        "memory_expansion": 0,
                        "memory_size": 0,
                        "cum_refund": 0,
                        "call_address": "",
                    }
                ]


class BlockProcessor:
    """Manages block processing status."""

    def __init__(
        self,
        block_data_dir: str,
        xatu_clickhouse_fetcher: XatuClickhouse,
        erigon_rpc: ErigonRPC,
        thread_pool_size: int = 8,
    ):
        self.block_data_dir = block_data_dir
        self.xatu_clickhouse_fetcher = xatu_clickhouse_fetcher
        self.erigon_rpc = erigon_rpc
        self.thread_pool_size = thread_pool_size

    def get_block_dir(self, block_height):
        return os.path.join(self.block_data_dir, f"block_height={block_height}")

    def get_tx_hashes_to_process(self, block_height: int, reprocess: bool) -> List[str]:
        """
        Return the list of transaction hashes to process.
        If reprocess=False, checks if a block is fully processed or whether there are missing transactions.
        """
        transaction_hashes = self.xatu_clickhouse_fetcher.get_tx_hashes_for_block(
            block_height
        )
        block_dir = self.get_block_dir(block_height)
        # if we want to reprocess, return all transactions
        if reprocess:
            logging.info(f"Reprocessing block {block_height}.")
            return transaction_hashes
        # if the block folder does not exist, return all transactions
        elif not os.path.exists(block_dir):
            logging.info(
                f"Block {block_height} not yet processed. Processing it for the first time."
            )
            return transaction_hashes
        # if the block folder exists, return only missing transactions
        else:
            missing_tx_hashes = []
            for tx_hash in transaction_hashes:
                tx_dir = os.path.join(block_dir, f"tx_hash={tx_hash}")
                parquet_file_path = os.path.join(tx_dir, "file.parquet")
                if not os.path.exists(parquet_file_path):
                    missing_tx_hashes.append(tx_hash)
            return missing_tx_hashes

    def _write_traces_to_parquet(self, traces, block_height, tx_hash):
        """Writes transaction traces to Parquet file."""
        logging.debug(
            f"Writing traces to Parquet for block {block_height}, tx_hash: {tx_hash}"
        )
        block_dir = self.get_block_dir(block_height)
        tx_dir = os.path.join(block_dir, f"tx_hash={tx_hash}")
        os.makedirs(tx_dir, exist_ok=True)
        parquet_file_path = os.path.join(tx_dir, "file.parquet")
        try:
            df = pd.DataFrame(traces)
            df.to_parquet(parquet_file_path)
            logging.debug(
                f"Traces written to Parquet for block {block_height}, tx_hash: {tx_hash}"
            )
        except Exception as e:
            logging.error(
                f"Failed to write traces to Parquet: {parquet_file_path}. Error: {e}"
            )
            return
        logging.info(f"Transaction traces written to Parquet file: {parquet_file_path}")

    def _process_transaction(self, tx_hash, block_height):
        """Processes a single transaction: fetches trace and writes to parquet."""
        logging.debug(f"Fetching trace for tx_hash: {tx_hash} in block {block_height}")
        traces = self.erigon_rpc.fetch_transaction_traces(
            tx_hash,
            block_height=block_height,
        )
        self._write_traces_to_parquet(traces, block_height, tx_hash)
        gc.collect()

    def process_block_range(
        self,
        block_start: int,
        block_count: int,
        reprocess: bool,
    ):
        """Processes a range of blocks to fetch and stores the transaction traces."""
        logging.debug(
            f"Processing block range from {block_start} to {block_start + block_count - 1}"
        )
        for block_height in range(block_start, block_start + block_count):
            logging.debug(f"Processing block {block_height}")
            tx_hashes_to_process = self.get_tx_hashes_to_process(
                block_height, reprocess
            )
            if len(tx_hashes_to_process) == 0:
                logging.info(
                    f"Block {block_height} is already processed. Skipping block."
                )
                continue
            else:
                logging.info(
                    f"Parquet files missing for transactions {tx_hashes_to_process} in block {block_height}. Reprocessing these transactions."
                )
                block_dir = self.get_block_dir(block_height)
                if not os.path.exists(block_dir):
                    os.makedirs(block_dir, exist_ok=True)
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.thread_pool_size
                ) as executor:
                    futures = [
                        executor.submit(
                            self._process_transaction, tx_hash, block_height
                        )
                        for tx_hash in tx_hashes_to_process
                    ]
                    for future in tqdm(
                        concurrent.futures.as_completed(futures),
                        total=len(futures),
                        desc=f"Processing block {block_height}",
                    ):
                        future.result()
                logging.debug(f"Finished processing block {block_height}")
            logging.debug("Finished processing block range")


def parse_configuration():
    """
    Parses command line arguments and secrets, and returns a configuration dictionary.
    """
    file_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Process Ethereum blocks and transaction traces and saves them to parquet files."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.abspath(os.path.join(file_dir, "..", "..", "data")),
        help="Data directory (default: ./data). Parquet files will be stored here.",
    )

    parser.add_argument(
        "--block_start",
        type=int,
        default=22000000,
        help="Starting block number (default: 22000000)",
    )
    parser.add_argument(
        "--block_count",
        type=int,
        default=2,
        help="Number of blocks to process (default: 6000)",
    )
    parser.add_argument(
        "--reprocess",
        type=bool,
        default=True,
        help="Whether to reprocessed traces. Default is False.",
    )
    parser.add_argument(
        "--thread_pool_size",
        type=int,
        default=8,
        help="Number of threads to use for processing transactions (default: 8)",
    )
    parser.add_argument(
        "--clickhouse_pool_size",
        type=int,
        default=5,
        help="Clickhouse connection pool size (default: 5)",
    )
    parser.add_argument(
        "--clickhouse_max_overflow",
        type=int,
        default=10,
        help="Clickhouse connection pool max overflow (default: 10)",
    )
    parser.add_argument(
        "--clickhouse_pool_timeout",
        type=int,
        default=30,
        help="Clickhouse connection pool timeout (default: 30)",
    )

    parser.add_argument(
        "--secrets_path",
        type=str,
        default=os.path.abspath(os.path.join(file_dir, "..", "..", "secrets.json")),
        help="Path to secrets.json file (default: ./secrets.json)",
    )

    parser.add_argument(
        "--erigon_rpc_url",
        type=str,
        default="https://rpc-mainnet-teku-erigon-001.utility.production.platform.ethpandaops.io",
        help="Erigon RPC URL (default: https://rpc-mainnet-teku-erigon-001.utility.production.platform.ethpandaops.io)",
    )
    parser.add_argument(
        "--erigon_username",
        type=str,
        help="Erigon RPC username (can be provided in secrets.json)",
    )
    parser.add_argument(
        "--erigon_password",
        type=str,
        help="Erigon RPC password (can be provided in secrets.json)",
    )
    parser.add_argument(
        "--erigon_rpc_response_max_size",
        type=int,
        default=int(1e9),
        help="Maximum response size for Erigon RPC calls (default: 1GB)",
    )

    parser.add_argument(
        "--xatu_clickhouse_url_base",
        type=str,
        default="clickhouse+http://clickhouse.xatu.ethpandaops.io:443/default?protocol=https",
        help="Clickhouse URL base (default: clickhouse+http://clickhouse.xatu.ethpandaops.io:443/default?protocol=https)",
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

    args = parser.parse_args()

    config = {}

    config["data_dir"] = args.data_dir
    config["secrets_path"] = args.secrets_path

    config["block_start"] = args.block_start
    config["block_count"] = args.block_count
    config["reprocess"] = args.reprocess
    config["thread_pool_size"] = args.thread_pool_size
    config["clickhouse_pool_size"] = args.clickhouse_pool_size
    config["clickhouse_max_overflow"] = args.clickhouse_max_overflow
    config["clickhouse_pool_timeout"] = args.clickhouse_pool_timeout

    config["erigon_rpc_url"] = args.erigon_rpc_url
    config["erigon_username"] = args.erigon_username
    config["erigon_password"] = args.erigon_password
    config["erigon_rpc_response_max_size"] = args.erigon_rpc_response_max_size

    config["xatu_clickhouse_url_base"] = args.xatu_clickhouse_url_base
    config["xatu_username"] = args.xatu_username
    config["xatu_password"] = args.xatu_password

    try:
        with open(config["secrets_path"], "r") as file:
            secrets_dict = json.load(file)
        logging.debug(f"Secrets loaded from {config['secrets_path']}")

        if not config["xatu_username"]:
            config["xatu_username"] = secrets_dict.get("xatu_username")
        if not config["xatu_password"]:
            config["xatu_password"] = secrets_dict.get("xatu_password")
        if not config["erigon_username"]:
            config["erigon_username"] = secrets_dict.get("erigon_username")
        if not config["erigon_password"]:
            config["erigon_password"] = secrets_dict.get("erigon_password")

    except FileNotFoundError:
        logging.warning(
            f"Secrets file not found at {config['secrets_path']}. Secrets might be missing if not provided via command line."
        )

    return config


def main():
    logging.debug("Starting main function")

    config = parse_configuration()
    data_dir = config["data_dir"]
    block_data_dir = os.path.join(data_dir, "block_data")

    block_start = config["block_start"]
    block_count = config["block_count"]
    reprocess = config["reprocess"]
    thread_pool_size = config["thread_pool_size"]
    clickhouse_pool_size = config["clickhouse_pool_size"]
    clickhouse_max_overflow = config["clickhouse_max_overflow"]
    clickhouse_pool_timeout = config["clickhouse_pool_timeout"]

    os.makedirs(block_data_dir, exist_ok=True)
    logging.debug(f"Block data directory created or exists: {block_data_dir}")

    xatu_username = config.get("xatu_username")
    xatu_password = config.get("xatu_password")
    if not xatu_username or not xatu_password:
        logging.error(
            "Xatu Clickhouse credentials not found. Please provide them via command line, environment variables, or secrets.json."
        )
        return

    xatu_clickhouse_url_base = config["xatu_clickhouse_url_base"]
    db_url = f"{xatu_clickhouse_url_base.split('://', 1)[0]}://{xatu_username}:{xatu_password}@{xatu_clickhouse_url_base.split('://', 1)[1]}"

    erigon_rpc_url = config["erigon_rpc_url"]
    erigon_rpc_response_max_size = config["erigon_rpc_response_max_size"]
    erigon_username = config.get("erigon_username")
    erigon_password = config.get("erigon_password")
    if not erigon_username or not erigon_password:
        logging.error(
            "Erigon RPC credentials not found. Please provide them via command line, environment variables, or secrets.json."
        )
        return

    erigon_rpc = ErigonRPC(
        erigon_rpc_url, erigon_username, erigon_password, erigon_rpc_response_max_size
    )
    xatu_clickhouse_fetcher = XatuClickhouse(
        db_url,
        pool_size=clickhouse_pool_size,
        max_overflow=clickhouse_max_overflow,
        pool_timeout=clickhouse_pool_timeout,
    )
    logging.debug("ErigonRPC and XatuClickhouse instances created")

    block_processor = BlockProcessor(
        block_data_dir, xatu_clickhouse_fetcher, erigon_rpc, thread_pool_size
    )
    block_processor.process_block_range(
        block_start,
        block_count,
        reprocess,
    )
    logging.debug("Ending main function")


if __name__ == "__main__":
    main()
