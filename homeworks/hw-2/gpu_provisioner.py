#!/usr/bin/env python3

import argparse
import time
import uuid
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm

from google.cloud import compute_v1
from google.api_core import exceptions


def normalize_gpu_name(name: str) -> str:
    return name.strip().lower()


def gpu_to_quota_metric(gpu_name: str) -> str:
    n = gpu_name.upper().replace("NVIDIA-", "").replace("TESLA-", "").replace("-", "_")
    return f"NVIDIA_{n}_GPUS"


def zone_to_region(zone: str) -> str:
    parts = zone.split("-")
    return "-".join(parts[:2])


class DynamicGPUProvisioner:
    def __init__(self, project_id: str, gpu_types: List[str], keep_count: int = 0):
        self.project_id = project_id
        self.gpu_types = [normalize_gpu_name(g) for g in gpu_types]
        self.gpu_targets = {g: gpu_to_quota_metric(g) for g in self.gpu_types}
        self.keep_count = max(0, keep_count)
        self.current_kept = 0

        # Stats
        self.api_calls = 0
        self.passive_time = 0
        self.active_time = 0
        self.active_attempts = 0
        self.active_successes = 0

    # ------------------------------------------------------------------
    # METHOD 1 — Passive (Hardware + Quota Listing)
    # ------------------------------------------------------------------
    def method_1_passive_scan(self) -> Tuple[List[Dict], float]:
        start = time.time()

        accel_client = compute_v1.AcceleratorTypesClient()
        region_client = compute_v1.RegionsClient()

        # 1 API call — aggregated accelerator listing
        self.api_calls += 1
        zone_to_gpus = {}
        for scope, scoped_list in accel_client.aggregated_list(project=self.project_id):
            if scope.startswith("zones/") and scoped_list.accelerator_types:
                zone = scope.replace("zones/", "")
                zone_to_gpus[zone] = [
                    acc.name.lower() for acc in scoped_list.accelerator_types
                ]

        # 1 API call — region quotas
        self.api_calls += 1
        region_quota_remaining = {}
        for region in region_client.list(project=self.project_id):
            quotas = {}
            for q in region.quotas:
                quotas[q.metric] = int(q.limit - q.usage)
            region_quota_remaining[region.name] = quotas

        # Filter candidate zones
        candidates = []
        for zone, gpu_list in zone_to_gpus.items():
            region = zone_to_region(zone)
            for gpu_name, quota_metric in self.gpu_targets.items():
                if gpu_name in gpu_list:
                    remaining = region_quota_remaining.get(region, {}).get(quota_metric, 0)
                    if remaining > 0:
                        candidates.append({
                            "zone": zone,
                            "region": region,
                            "gpu_type": gpu_name
                        })

        elapsed = round(time.time() - start, 2)
        self.passive_time = elapsed
        return candidates, elapsed

    # ------------------------------------------------------------------
    # Failure Classification
    # ------------------------------------------------------------------
    def classify_error(self, msg: str) -> str:
        m = msg.lower()

        if "resource_pool_exhausted" in m or "zone resource pool exhausted" in m:
            return "No GPUs available (Stockout)"
        if "quota" in m:
            return "Quota exceeded"
        if "billing" in m or "account" in m:
            return "Billing / Pricing issue"

        return "Other API error"

    # ------------------------------------------------------------------
    # METHOD 2 — Active (Sequential VM Attempt)
    # ------------------------------------------------------------------
    def method_2_active_attempt(self, node: Dict) -> Dict:
        self.active_attempts += 1
        client = compute_v1.InstancesClient()
        start = time.time()

        vm_name = f"gpu-test-{uuid.uuid4().hex[:8]}"
        zone = node["zone"]
        gpu_type = node["gpu_type"]

        machine_type = "g2-standard-4" if "l4" in gpu_type else "n1-standard-4"

        instance = compute_v1.Instance(
            name=vm_name,
            machine_type=f"projects/{self.project_id}/zones/{zone}/machineTypes/{machine_type}",
            guest_accelerators=[
                compute_v1.AcceleratorConfig(
                    accelerator_count=1,
                    accelerator_type=f"projects/{self.project_id}/zones/{zone}/acceleratorTypes/{gpu_type}"
                )
            ],
            disks=[
                compute_v1.AttachedDisk(
                    auto_delete=True,
                    boot=True,
                    initialize_params=compute_v1.AttachedDiskInitializeParams(
                        source_image="projects/debian-cloud/global/images/family/debian-11"
                    )
                )
            ],
            network_interfaces=[compute_v1.NetworkInterface(network="global/networks/default")],
            scheduling=compute_v1.Scheduling(on_host_maintenance="TERMINATE"),
        )

        try:
            self.api_calls += 1
            op = client.insert(
                project=self.project_id,
                zone=zone,
                instance_resource=instance
            )
            op.result(timeout=120)

            elapsed = round(time.time() - start, 2)
            self.active_successes += 1

            kept = False
            if self.keep_count > 0 and self.current_kept < self.keep_count:
                self.current_kept += 1
                kept = True
            else:
                try:
                    self.api_calls += 1
                    del_op = client.delete(
                        project=self.project_id,
                        zone=zone,
                        instance=vm_name
                    )
                    del_op.result(timeout=120)
                except Exception:
                    pass

            return {
                "Zone": zone,
                "GPU": gpu_type,
                "GPU available (Yes/No)": "Yes",
                "GPU allocated successfully (Yes/No)": "Yes",
                "Reason for failure": "N/A",
                "Time taken for each check": elapsed
            }

        except exceptions.GoogleAPIError as e:
            elapsed = round(time.time() - start, 2)
            msg = str(e)
            return {
                "Zone": zone,
                "GPU": gpu_type,
                "GPU available (Yes/No)": "Yes",
                "GPU allocated successfully (Yes/No)": "No",
                "Reason for failure": self.classify_error(msg),
                "Time taken for each check": elapsed
            }

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self):
        candidates, passive_time = self.method_1_passive_scan()
        print(f"Method 1 completed in {passive_time}s")

        results = []
        start_active = time.time()

        for node in tqdm(candidates, desc="Active GPU Allocation Attempts"):
            res = self.method_2_active_attempt(node)
            results.append(res)

        self.active_time = round(time.time() - start_active, 2)

        print("\n=== Efficiency Comparison ===")
        print(f"Passive scan time: {self.passive_time}s (2 API calls total)")
        print(f"Active attempt time: {self.active_time}s")
        print(f"Total API calls: {self.api_calls}")
        print(f"Active attempts: {self.active_attempts}")
        print(f"Successful allocations: {self.active_successes}")
        print(f"VMs kept: {self.current_kept}")

        return pd.DataFrame(results)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential GCP GPU Discovery")
    parser.add_argument("--project", required=True)
    parser.add_argument("--gpu-types", nargs="+", default=["nvidia-tesla-t4", "nvidia-l4"])
    parser.add_argument("--keep", type=int, default=0)
    args = parser.parse_args()

    provisioner = DynamicGPUProvisioner(
        args.project,
        args.gpu_types,
        args.keep
    )

    df = provisioner.run()
    print("\n")
    print(df.to_markdown(index=False))