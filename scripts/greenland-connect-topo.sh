#!/usr/bin/env bash
#
# greenland-connect-topo.sh — Connect to Instance 2 (topological experiment)
# Current job: cmohsinm-workspace-uncertainity (submitted 2026-06-25T23:32Z)
#   SSM: mi-0a7b3e00847947566 / EC2: i-0832beffb99d1dd09 / IP: 10.3.33.7 / p4d.24xlarge
# Previous: mi-0e220ab52563a081e (10.3.219.28), mi-02b46ccb515620db7 (10.3.228.25)
set -euo pipefail

ACCOUNT="703671891219"
CUSTOMER_ROLE="Intern"
PROVIDER="isengard"
PROFILE="greenland"
REGION="us-east-2"
JOB_ROLE_ARN="arn:aws:iam::072510399842:role/greenland-access-37f871283e3e69fdbfe97939a34079a8bfdfdd85"
SSM_TARGET="mi-0a7b3e00847947566"
REMOTE_PORT="2222"
LOCAL_PORT="1056"
SSH_USER="greenland-user"

auth() {
  echo ">> Authenticating with Midway + Isengard..."
  mwinit -f
  ada credentials update --account="$ACCOUNT" --provider="$PROVIDER" --role="$CUSTOMER_ROLE" --once
  aws configure set --profile "$PROFILE" source_profile default
  aws configure set --profile "$PROFILE" region "$REGION"
  aws configure set --profile "$PROFILE" role_arn "$JOB_ROLE_ARN"
  aws sts get-caller-identity --profile "$PROFILE"
  echo ">> ✓ Auth OK."
}

tunnel() {
  echo ">> Opening SSM tunnel to Instance 2 (topo) on local port $LOCAL_PORT..."
  echo ">> Target: $SSM_TARGET (10.3.33.7)"
  aws ssm start-session \
    --target "$SSM_TARGET" \
    --document-name AWS-StartPortForwardingSession \
    --parameters "{\"portNumber\":[\"$REMOTE_PORT\"],\"localPortNumber\":[\"$LOCAL_PORT\"]}" \
    --profile "$PROFILE" \
    --region "$REGION"
}

ssh_in() {
  echo ">> SSH into Instance 2 (topo) on localhost:$LOCAL_PORT..."
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
