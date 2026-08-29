# trading-harness

## Agent Reach

This project's Claude Code sessions run in disposable containers, so any
tool installed outside this repo (like [Agent Reach](https://github.com/Panniantong/Agent-Reach),
which gives the agent read access to the web, YouTube, GitHub, RSS, etc.)
disappears when the container is recycled and needs to be reinstalled each
fresh session. To set it back up:

```bash
bash scripts/install_agent_reach.sh --system
```

Drop `--system` to only run the safe, read-only dependency check without
installing anything. See the script's header comment for details, including
the fallback for sandboxes that block direct archive downloads.