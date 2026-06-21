#!/usr/bin/env bash
#
# greenland-sync.sh — Push/pull code and results to/from Greenland
#
# Usage:
#   ./scripts/greenland-sync.sh push              # push local code
#   ./scripts/greenland-sync.sh pull [path]       # pull results back
#   ./scripts/greenland-sync.sh run "command"     # run command on Greenland
#   ./scripts/greenland-sync.sh deploy            # push + install deps
#   ./scripts/greenland-sync.sh experiment        # push + run topological experiment
set -euo pipefail

LOCAL_PORT="1053"
SSH_USER="greenland-user"
SSH_HOST="localhost"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

LOCAL_PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_PROJECT_DIR="~/inference-time-uncertainty"

EXCLUDE=(
    ".git"
    "__pycache__"
    "*.pyc"
    ".venv"
    "node_modules"
    "build"
    ".eggs"
    "*.egg-info"
    "wandb"
    ".DS_Store"
    "data/inference_outputs"
    "data/calibration_outputs"
    "runs"
)

EXCLUDE_FLAGS=""
for pattern in "${EXCLUDE[@]}"; do
    EXCLUDE_FLAGS="$EXCLUDE_FLAGS --exclude=$pattern"
done

SSH_CMD="ssh -p $LOCAL_PORT $SSH_OPTS"

check_tunnel() {
    if ! ssh -p "$LOCAL_PORT" $SSH_OPTS -o ConnectTimeout=3 "$SSH_USER@$SSH_HOST" "echo ok" > /dev/null 2>&1; then
        echo "✗ Cannot reach Greenland. Is the SSM tunnel running?"
        echo "  Start it with: ./scripts/greenland-connect.sh tunnel"
        exit 1
    fi
}

push() {
    echo ">> Pushing local code to Greenland..."
    check_tunnel
    rsync -avz --delete \
        $EXCLUDE_FLAGS \
        -e "$SSH_CMD" \
        "$LOCAL_PROJECT_DIR/" \
        "$SSH_USER@$SSH_HOST:$REMOTE_PROJECT_DIR/"
    echo ">> ✓ Code pushed to $REMOTE_PROJECT_DIR"
}

pull() {
    local remote_path="${1:-data/topological_outputs/}"
    local local_path="${2:-$LOCAL_PROJECT_DIR/data/topological_outputs/}"
    echo ">> Pulling from Greenland: $remote_path..."
    check_tunnel
    mkdir -p "$local_path"
    rsync -avz \
        -e "$SSH_CMD" \
        "$SSH_USER@$SSH_HOST:$REMOTE_PROJECT_DIR/$remote_path" \
        "$local_path"
    echo ">> ✓ Pulled to $local_path"
}

run_remote() {
    local cmd="$1"
    echo ">> Running on Greenland: $cmd"
    check_tunnel
    ssh -p "$LOCAL_PORT" $SSH_OPTS "$SSH_USER@$SSH_HOST" \
        "cd $REMOTE_PROJECT_DIR && source ~/miniconda3/etc/profile.d/conda.sh && conda activate digte && $cmd"
}

deploy() {
    push
    echo ""
    echo ">> Installing dependencies on Greenland..."
    run_remote "pip install ripser persim gudhi vllm matplotlib scipy"
    echo ">> ✓ Deploy complete"
}

experiment() {
    local extra_args="${1:-}"
    push
    echo ""
    echo ">> Running topological persistence experiment..."
    run_remote "python -m topological_persistence.run --model Qwen/Qwen3-32B --dataset aime_2024 --n-problems 5 --n-chains 8 --representation curve $extra_args"
    echo ""
    pull "data/topological_outputs/" "$LOCAL_PROJECT_DIR/data/topological_outputs/"
}

case "${1:-help}" in
    push)       push ;;
    pull)       pull "${2:-data/topological_outputs/}" "${3:-$LOCAL_PROJECT_DIR/data/topological_outputs/}" ;;
    run)        run_remote "${2:?Usage: $0 run \"command\"}" ;;
    deploy)     deploy ;;
    experiment) experiment "${2:-}" ;;
    help|*)
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  push                Sync local code → Greenland"
        echo "  pull [path]         Pull results from Greenland (default: data/topological_outputs/)"
        echo "  run \"command\"       Execute a command on Greenland"
        echo "  deploy              Push code + install dependencies"
        echo "  experiment [args]   Push → run topological persistence → pull results"
        echo ""
        echo "Prerequisites:"
        echo "  1. Auth must be done:    ./scripts/greenland-auth.sh"
        echo "  2. Tunnel must be open:  ./scripts/greenland-connect.sh tunnel"
        ;;
esac
