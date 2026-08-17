#!/usr/bin/bash
source .venv/bin/activate
# Run with an agent that does nothing, to make sure the environment works as intended
WANDB_MODE=disabled uv run play Mjlab-LocoManip-BRAVE-G1 --agent zero

