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
        "cd $REMOTE_PROJECT_DIR && source ~/miniconda3/etc/profile.d/conda.sh && conda activate topo && $cmd"
}

run_detached() {
    local cmd="$1"
    local logfile="${2:-experiment.log}"
    echo ">> Launching detached on Greenland (survives disconnect)..."
    echo ">> Log: ~/$logfile"
    check_tunnel
    ssh -p "$LOCAL_PORT" $SSH_OPTS "$SSH_USER@$SSH_HOST" \
        "cd $REMOTE_PROJECT_DIR && nohup bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate topo && $cmd' > ~/$logfile 2>&1 &"
    echo ">> ✓ Process launched in background"
    echo ""
    echo "  Monitor:  ./scripts/greenland-sync.sh logs"
    echo "  Status:   ./scripts/greenland-sync.sh status"
    echo "  Pull:     ./scripts/greenland-sync.sh pull"
}

logs() {
    local logfile="${1:-experiment.log}"
    check_tunnel
    ssh -p "$LOCAL_PORT" $SSH_OPTS "$SSH_USER@$SSH_HOST" "tail -50 ~/$logfile"
}

status() {
    check_tunnel
    ssh -p "$LOCAL_PORT" $SSH_OPTS "$SSH_USER@$SSH_HOST" \
        "ps aux | grep 'topological_persistence\|python -m' | grep -v grep || echo 'No running experiment found'"
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
    echo ">> Launching topological persistence experiment (detached)..."
    run_detached "python -m topological_persistence.run --model Qwen/Qwen3-32B --dataset aime_2024 --n-problems 5 --n-chains 8 --representation curve $extra_args"
}

case "${1:-help}" in
    push)       push ;;
    pull)       pull "${2:-data/topological_outputs/}" "${3:-$LOCAL_PROJECT_DIR/data/topological_outputs/}" ;;
    run)        run_remote "${2:?Usage: $0 run \"command\"}" ;;
    runbg)      run_detached "${2:?Usage: $0 runbg \"command\"}" "${3:-experiment.log}" ;;
    deploy)     deploy ;;
    experiment) experiment "${2:-}" ;;
    logs)       logs "${2:-experiment.log}" ;;
    status)     status ;;
    help|*)
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  push              Sync local code → Greenland"
        echo "  pull [path]       Pull results (default: data/topological_outputs/)"
        echo "  run \"command\"     Run on Greenland (foreground, dies on disconnect)"
        echo "  runbg \"cmd\" [log] Run detached with nohup (survives disconnect)"
        echo "  deploy            Push code + install deps"
        echo "  experiment [args] Push + launch experiment (detached)"
        echo "  logs [file]       Tail experiment log (default: experiment.log)"
        echo "  status            Check if experiment is still running"
        echo ""
        echo "Prerequisites:"
        echo "  1. Auth:    ./scripts/greenland-auth.sh"
        echo "  2. Tunnel:  ./scripts/greenland-connect.sh tunnel"
        ;;
esac
