#!/bin/bash
# check-progress.sh - Check FL training progress from checkpoints
# Usage: ./check-progress.sh

CHECKPOINT_DIR="$HOME/coldstart_runs/flower_bigmodel"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "No checkpoint directory found at: $CHECKPOINT_DIR"
    exit 1
fi

echo "=== Federated Learning Training Progress ==="
echo ""

# Check server checkpoints
echo "Server Checkpoints:"
if [ -f "$CHECKPOINT_DIR/latest.pt" ]; then
    python3 <<EOF
import pickle
import os
from pathlib import Path

checkpoint_dir = Path('$CHECKPOINT_DIR')
latest_file = checkpoint_dir / 'latest.pt'

try:
    with open(latest_file, 'rb') as f:
        ckpt = pickle.load(f)
    
    current_round = ckpt['round']
    strategy_state = ckpt.get('strategy_state', {})
    
    print(f"  Current round: {current_round}")
    print(f"  Checkpoint file: {latest_file}")
    
    # List all server checkpoints
    server_ckpts = sorted(checkpoint_dir.glob('server_round_*.pt'))
    if server_ckpts:
        print(f"  Total checkpoints: {len(server_ckpts)}")
        print(f"  First: {server_ckpts[0].name}")
        print(f"  Latest: {server_ckpts[-1].name}")
    
    if strategy_state:
        print(f"  Strategy state: {strategy_state}")
    
except Exception as e:
    print(f"  Error reading checkpoint: {e}")
EOF
else
    echo "  No server checkpoints found"
fi

echo ""
echo "Client Checkpoints:"
for client_dir in "$CHECKPOINT_DIR"/client_*; do
    if [ -d "$client_dir" ]; then
        client_id=$(basename "$client_dir" | sed 's/client_//')
        num_ckpts=$(ls "$client_dir"/local_round_*.pt 2>/dev/null | wc -l)
        echo "  Client $client_id: $num_ckpts checkpoints"
        
        if [ -f "$client_dir/latest.pt" ]; then
            python3 <<EOF
import torch
try:
    ckpt = torch.load('$client_dir/latest.pt', map_location='cpu')
    print(f"    Latest round: {ckpt['server_round']}")
    if 'meta' in ckpt:
        meta = ckpt['meta']
        print(f"    Partition: {meta.get('partition', 'N/A')}")
        print(f"    Dataset: {meta.get('dataset', 'N/A')}")
        if 'train_loss' in meta:
            print(f"    Train loss: {meta['train_loss']:.4f}")
except Exception as e:
    print(f"    Error: {e}")
EOF
        fi
    fi
done

if [ ! -d "$CHECKPOINT_DIR/client_0" ]; then
    echo "  No client checkpoints found"
fi

echo ""
echo "Disk usage:"
du -sh "$CHECKPOINT_DIR"
