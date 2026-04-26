# UA HPC — `slurm-selector.sh` Behavior (verified)

Confirmed by inspecting `/usr/local/bin/slurm-selector.sh` on `junonia.hpc.arizona.edu` (Puma login) on 2026-04-26.

## Summary

`slurm-selector.sh <cluster>` sets `SLURM_CONF` so that subsequent `squeue`/`sbatch`/etc. talk to the chosen cluster's controller.

For each cluster it exports a path:

| Cluster | `SLURM_CONF` |
|---------|--------------|
| puma    | `/etc/slurm-puma/slurm.conf`    |
| ocelote | (`/etc/slurm-ocelote/slurm.conf` — pattern matches) |
| elgato  | (`/etc/slurm-elgato/slurm.conf` — pattern matches) |

Also sets `PROMPT_COMMAND='echo -n "(<cluster>) "'` and various MPI / fabric env vars.

## Aliases

The user-facing shortcuts are bash aliases that source the script:

```
alias ocelote='. /usr/local/bin/slurm-selector.sh ocelote'
alias puma='. /usr/local/bin/slurm-selector.sh puma'
alias elgato='. /usr/local/bin/slurm-selector.sh elgato'
```

(verified by `type ocelote` on Puma login)

## Implication: how to switch in scripts / tmux

Aliases are not loaded in non-interactive shells, so inside tmux's command (or a `.sh` file) the bare `ocelote` will fail. The robust replacement:

```bash
. /usr/local/bin/slurm-selector.sh ocelote   # source — runs in current shell
```

This is the same thing the alias does. Works in any bash context (interactive, login, plain `bash -c '...'`, tmux command, etc.) because we're using `source` (`.`) on a known absolute path.

Do **NOT** run it as a child process (`/usr/local/bin/slurm-selector.sh ocelote` without the dot) — exports happen in a child shell that exits without affecting the calling shell.

## Cross-cluster `-M` flag

Tested earlier from Puma login: `squeue -M ocelote -u $USER -h` returns 0 results even when Ocelote actually has jobs in queue. Conclusion: `-M` cross-cluster lookup is NOT functional in UA's setup. Always source `slurm-selector.sh` to switch contexts.
