# Dynamic GCP GPU Provisioner

A synchronous Python tool to discover and validate GPU availability across an entire GCP project footprint.  
It is designed for low API overhead operation and safe, sequential validation of real-time GPU allocation.

## Overview

The tool uses a two-stage funnel:

1. **Passive filtering (Method 1)**  
   - Performs exactly two logical API calls: an aggregated accelerator-types listing (to learn which accelerator types are advertised per zone) and a regions list call (to collect per-region GPU quotas).
   - Produces a filtered candidate set of `(region, zone, gpu_type)` where both hardware is advertised and region quota appears available.
   - This stage minimizes API usage and eliminates obviously invalid allocation targets.

2. **Active sequential validation (Method 2)**  
   - Sequentially attempts instance creation for the selected candidates to confirm ground-truth availability (handles `RESOURCE_POOL_EXHAUSTED`, quota and billing errors).
   - Runs sequentially (no parallel insertion attempts) to avoid provoking quota contention artifacts.
   - Successful test instances are deleted immediately by default to avoid costs; you can optionally keep a small number.

## Why this approach

- **Minimal API calls:** Passive scan is implemented with two logical requests for the full project, avoiding per-zone or per-region polling.
- **Correctness:** Active sequential allocation is the only reliable way to detect zonal stockouts.
- **Deterministic behavior:** Sequential attempts prevent accidental `QUOTA_EXCEEDED` responses caused by concurrent attempts in a low-quota environment.

## Prerequisites

- **Python:** 3.10+ recommended.
- **GCP Permissions:** Service account or user must have sufficient compute permissions (e.g., `roles/compute.admin`) and billing must be enabled for instance creation.
- **Dependencies:** `google-cloud-compute`, `pandas`, and `tqdm`.

### Install
```bash
pip install google-cloud-compute pandas tabulate tqdm
````

### Authenticate

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

## Usage

```bash
python gpu_provisioner.py --project YOUR_PROJECT_ID
```

### CLI Arguments

| Argument      |     Type | Default                     | Description                                                    |
| ------------- | -------: | --------------------------- | -------------------------------------------------------------- |
| `--project`   | `string` | **Required**                | GCP project id.                                                |
| `--gpu-types` |   `list` | `nvidia-tesla-t4 nvidia-l4` | GPU types to search for (space-separated).                     |
| `--keep`      |    `int` | `0`                         | Number of successfully provisioned VMs to retain (default: 0). |

## Typical workflows

* **Audit availability across a project:** run with default flags — passive scan + sequential validation will report where GPUs can be provisioned.
* **Secure a single GPU for a job:** run with `--keep 1`. The script marks the first successful allocation as kept and stops keeping further VMs.
* **Hunt for a specific high-end accelerator:** pass `--gpu-types` with specific hardware (e.g., `nvidia-tesla-v100`) to restrict the passive filter and active checks.

## Output & error classification

* The script prints a summary table of attempt results and a small efficiency comparison:

  * Passive scan time and API-call counts
  * Active attempt counts, successes, and number of VMs retained
* Active failures are categorized for easy triage:

  * `No GPUs available (stockout)` — zonal resource exhaustion
  * `Quota exceeded` — region-level quota reached
  * `Billing / Pricing issue` — account or billing-related rejections
  * `Other API error` — miscellaneous API responses

## Operational notes & caveats

* **Method 1 cannot detect transient, zonal resource pool exhaustion.** A zone advertising a GPU type and a positive region quota do not guarantee a successful allocation in Method 2.
* **Sequential active checks** are intentionally conservative to avoid producing quota contention errors in low-quota projects.
* There is no reliable universal “dry-run” for instance placement that guarantees physical stock validation without an allocation attempt; therefore, a controlled allocation attempt is required for ground-truth confirmation.

## Example

Availability audit across project:

```bash
python gpu_provisioner.py --project my-project
```

Find and keep one GPU:

```bash
python gpu_provisioner.py --project my-project --keep 1
```