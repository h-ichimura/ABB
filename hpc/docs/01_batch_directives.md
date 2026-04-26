# UA HPC — SBATCH Directives Reference

Source: https://hpcdocs.hpc.arizona.edu/running_jobs/batch_jobs/batch_directives/
Fetched: 2026-04-26

## Core Directives

| Directive | Example | Notes |
|-----------|---------|-------|
| `--job-name` | `--job-name=hello_world` | |
| `--account` | `--account=<PI GROUP>` | **Omit for windfall** |
| `--partition` | `--partition=standard` | See valid list below |
| `--nodes` | `--nodes=1` | |
| `--ntasks` | `--ntasks=1` | |
| `--cpus-per-task` | `--cpus-per-task=10` | |
| `--mem` | `--mem=50gb` | Or `--mem-per-cpu=Xgb` |
| `--time` | `--time=DD-HH:MM:SS` | Max 10 days |
| `--array` | `--array=<N>-<M>` | |
| `--gres` | `--gres=gpu:1` | GPU only; see partition table |

## Valid Partitions

| Partition | --account? | --qos? | --gres? |
|-----------|-----------|--------|---------|
| `standard` | required | — | no |
| `windfall` | **must NOT include** | — | no |
| `high_priority` | required | `--qos=user_qos_<PI GROUP>` | no |
| `gpu_standard` | required | — | yes |
| `gpu_windfall` | **must NOT include** | — | yes |
| `gpu_high_priority` | required | `--qos=user_qos_<PI GROUP>` | yes |

## Windfall Rules (verbatim)

- "Unlimited access. Preemptible"
- "Do not include an `--account` flag when requesting this partition"
- No QOS directive required for windfall

## GPU --gres on Puma

```
#SBATCH --gres=gpu:1
#SBATCH --gres=gpu:volta:N
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb
```

## GPU --gres on Ocelote

```
#SBATCH --gres=gpu:N
#SBATCH --mem-per-cpu=8gb
```

## Job Dependencies

`--dependency=<type:jobid>` where type is `after`, `afterany`, `afterok`, `afternotok`.

## Output filename patterns

`%A` = array main jobid; `%a` = array task index; `%j` = job id; `%x` = job name.

## QOS Errors

- `QOSGrpSubmitJobsLimit`: missing `--qos` directive when in a buy-in group.
- `QOSMaxSubmitJobPerUserLimit`: hit the per-user cap (1000 on standard/windfall per `sacctmgr`).
