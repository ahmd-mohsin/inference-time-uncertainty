#!/usr/bin/env bash
#
# greenland-connect.sh — Auth + SSM tunnel + SSH to Greenland SDB
#
# Usage:
#   ./scripts/greenland-connect.sh          # auth + tunnel
#   ./scripts/greenland-connect.sh auth     # auth only
#   ./scripts/greenland-connect.sh tunnel   # tunnel only (auth must be done)
#   ./scripts/greenland-connect.sh ssh      # SSH in (tunnel must be running)
set -euo pipefail

ACCOUNT="703671891219"
CUSTOMER_ROLE="Intern"
PROVIDER="isengard"
PROFILE="greenland"
REGION="us-east-2"
JOB_ROLE_ARN="arn:aws:iam::072510399842:role/greenland-access-37f871283e3e69fdbfe97939a34079a8bfdfdd85"
SSM_TARGET="mi-08fed06edba1c35a4"
REMOTE_PORT="2222"
LOCAL_PORT="1053"
SSH_USER="greenland-user"

auth() {
  echo ">> Authenticating with Midway + Isengard..."
  mwinit -f
  ada credentials update --account="$ACCOUNT" --provider="$PROVIDER" --role="$CUSTOMER_ROLE" --once
  aws configure set --profile "$PROFILE" source_profile default
  aws configure set --profile "$PROFILE" region "$REGION"
  aws configure set --profile "$PROFILE" role_arn "$JOB_ROLE_ARN"
  aws sts get-caller-identity --profile "$PROFILE"
  echo ">> ✓ Auth OK. '$PROFILE' profile ready."
}

tunnel() {
  echo ">> Opening SSM port-forward $LOCAL_PORT -> $REMOTE_PORT (keep this terminal open)..."
  aws ssm start-session \
    --target "$SSM_TARGET" \
    --document-name AWS-StartPortForwardingSession \
    --parameters "{\"portNumber\":[\"$REMOTE_PORT\"],\"localPortNumber\":[\"$LOCAL_PORT\"]}" \
    --profile "$PROFILE" \
    --region "$REGION"
}

ssh_in() {
  echo ">> SSH into Greenland on localhost:$LOCAL_PORT..."
  ssh -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o ServerAliveInterval=60 \
      -p "$LOCAL_PORT" "$SSH_USER@localhost"
}

case "${1:-all}" in
  auth)   auth ;;
  tunnel) tunnel ;;
  ssh)    ssh_in ;;
  all)    auth; tunnel ;;
  *) echo "Usage: $0 [auth|tunnel|ssh|all]"; exit 1 ;;
esac
