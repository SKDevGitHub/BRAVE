#!/usr/bin/bash
source .venv/bin/activate
WANDB_MODE=disabled uv run train Mjlab-LocoManip-BRAVE-G1 --gpu-ids all

