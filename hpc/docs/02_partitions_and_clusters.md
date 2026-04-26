# UA HPC — Partitions, Clusters, and Cluster-Switching

## Clusters

| Cluster | Default | GPUs | Notes |
|---------|---------|------|-------|
| Puma    | yes     | 56 V100S + 8 A100-40GB MIG | 30,720 schedulable CPU cores |
| Ocelote | no      | 36 P100 (16GB) | "Research groups are limited to using a maximum of 10 GPUs simultaneously" |
| ElGato  | no      | older Tesla GPUs | (smallest) |
| TheNewCat | no    | 40 H200 (2026) | newest |

## Switching Clusters from Puma Login

Source: https://hpcdocs.hpc.arizona.edu/quick_start/accessing_compute/

> The default cluster for job submission is Puma  
> Shortcut commands change the target cluster:  
>   Puma:    `$ puma`     → `(puma) $`  
>   Ocelote: `$ ocelote`  → `(ocelote) $`  
>   ElGato:  `$ elgato`   → `(elgato) $`

The shortcuts are **shell aliases** that source `/usr/local/bin/slurm-selector.sh`.
Aliases require an interactive shell or `shopt -s expand_aliases` to take effect.

### Implications for tmux/non-interactive scripts

- Non-interactive bash (e.g. `tmux new -d -s name 'cmd'`) does NOT load `~/.bashrc` and aliases are not defined.
- `bash -lc "cmd"` (login shell) sources `.bash_profile`, which usually sources `.bashrc` — aliases get defined, but `expand_aliases` must be on for them to be invoked from inside a script.
- Safest: invoke the script directly: `source /usr/local/bin/slurm-selector.sh ocelote`. This does not depend on aliases.
- Cluster context is set via env vars exported by the script — sourcing keeps them in the parent shell; executing as a child does not.

## Partitions Per Cluster

Source: https://hpcdocs.hpc.arizona.edu/running_jobs/batch_jobs/batch_directives/

Same partition names exist on every cluster:

| Partition          | --account?       | --qos?                       | --gres? |
|--------------------|------------------|------------------------------|---------|
| `standard`         | required         | —                            | no      |
| `windfall`         | **must NOT have**| —                            | no      |
| `high_priority`    | required         | `--qos=user_qos_<PI GROUP>`  | no      |
| `gpu_standard`     | required         | —                            | yes     |
| `gpu_windfall`     | **must NOT have**| —                            | yes     |
| `gpu_high_priority`| required         | `--qos=user_qos_<PI GROUP>`  | yes     |

## Windfall Rules (verbatim)

Source: https://hpcdocs.hpc.arizona.edu/resources/allocations/

> Windfall is a partition available to jobs that enables them to run **without consuming your allocation**, but it also **reduces their priority**.
>
> The `--account` flag should be omitted when using the Windfall partition.
>
> Windfall jobs are preemptible, meaning standard and high-priority jobs can interrupt a running windfall job, effectively placing it back in the queue.

CPU-only windfall: `#SBATCH --partition=windfall`
GPU windfall:      `#SBATCH --partition=gpu_windfall` plus `#SBATCH --gres=gpu:N`
