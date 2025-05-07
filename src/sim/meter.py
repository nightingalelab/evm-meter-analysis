import sys
from pathlib import Path
from typing import List, Dict


sys.path.append(str(Path(__file__).parent.parent))
from sim.mempool import SimTx


def one_dim_scheme(tx_list: List[SimTx], limit: float) -> float:
    total_gas = 0.0
    for tx in tx_list:
        total_tx_gas = sum(tx.resource_dict.values())
        total_gas += total_tx_gas
    utilization = total_gas / limit
    return utilization


def state_vs_others(tx_list: List[SimTx], limit: float) -> float:
    utilization = two_dim_scheme(tx_list, limit, "State")
    return utilization


def compute_vs_others(tx_list: List[SimTx], limit: float) -> float:
    utilization = two_dim_scheme(tx_list, limit, "Compute")
    return utilization


def access_vs_others(tx_list: List[SimTx], limit: float) -> float:
    utilization = two_dim_scheme(tx_list, limit, "Access")
    return utilization


def bandwidth_vs_others(tx_list: List[SimTx], limit: float) -> float:
    utilization = two_dim_scheme(tx_list, limit, "Bandwidth")
    return utilization


def state_vs_compute_vs_others(tx_list: List[SimTx], limit: float) -> float:
    compute_gas = 0.0
    state_gas = 0.0
    others_gas = 0.0
    for tx in tx_list:
        tx_compute_gas = get_resource_safe(tx.resource_dict, "Compute")
        compute_gas += tx_compute_gas
        tx_state_gas = get_resource_safe(tx.resource_dict, "State")
        state_gas += tx_state_gas
        others_gas += sum(tx.resource_dict.values()) - tx_compute_gas - tx_state_gas
    utilization = max(state_gas, compute_gas, others_gas) / limit
    return utilization


def state_vs_compute_vs_access_vs_others(tx_list: List[SimTx], limit: float) -> float:
    compute_gas = 0.0
    state_gas = 0.0
    access_gas = 0.0
    others_gas = 0.0
    for tx in tx_list:
        tx_compute_gas = get_resource_safe(tx.resource_dict, "Compute")
        compute_gas += tx_compute_gas
        tx_state_gas = get_resource_safe(tx.resource_dict, "State")
        state_gas += tx_state_gas
        tx_access_gas = get_resource_safe(tx.resource_dict, "Access")
        access_gas += tx_access_gas
        others_gas += (
            sum(tx.resource_dict.values())
            - tx_compute_gas
            - tx_state_gas
            - tx_access_gas
        )
    utilization = max(state_gas, compute_gas, others_gas, access_gas) / limit
    return utilization


def two_dim_scheme(tx_list: List[SimTx], limit: float, resource: str) -> float:
    resource_gas = 0.0
    others_gas = 0.0
    for tx in tx_list:
        tx_resource_gas = get_resource_safe(tx.resource_dict, resource)
        resource_gas += tx_resource_gas
        others_gas += sum(tx.resource_dict.values()) - tx_resource_gas
    utilization = max(resource_gas, others_gas) / limit
    return utilization


def get_resource_safe(resource_dict: Dict[str, float], resource_str: str):
    if resource_str not in resource_dict:
        return 0.0
    else:
        return resource_dict[resource_str]
