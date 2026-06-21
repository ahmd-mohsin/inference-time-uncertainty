#!/usr/bin/env bash
#
# greenland-auth.sh — Run this from your LOCAL LAPTOP daily
set -euo pipefail

ACCOUNT="703671891219"
CUSTOMER_ROLE="Intern"
PROVIDER="isengard"
PROFILE="greenland"
REGION="us-east-2"
JOB_ROLE_ARN="arn:aws:iam::072510399842:role/greenland-access-37f871283e3e69fdbfe97939a34079a8bfdfdd85"
# Job: cmohsinm-workspace | Instance: p4d.24xlarge (8x A100 GPU)
# SSM Managed Instance: mi-00b4a7a2d8b022958
# EC2 Instance: i-0bbb43ee1228d971f

echo "============================================"
echo " Greenland Daily Auth (Local Laptop)"
echo "============================================"
echo ""

echo "[1/4] Refreshing Midway credentials..."
mwinit -f
echo "  ✓ Midway OK"
echo ""

echo "[2/4] Assuming role '$CUSTOMER_ROLE' on account $ACCOUNT (provider: $PROVIDER)..."
ada credentials update \
  --account "$ACCOUNT" \
  --role "$CUSTOMER_ROLE" \
  --provider "$PROVIDER" \
  --once
echo "  ✓ Credentials updated in default profile"
echo ""

echo "[3/4] Configuring '$PROFILE' AWS profile..."
aws configure set --profile "$PROFILE" source_profile default
aws configure set --profile "$PROFILE" region "$REGION"
aws configure set --profile "$PROFILE" role_arn "$JOB_ROLE_ARN"
echo "  ✓ Profile '$PROFILE' configured"
echo ""

echo "[4/4] Verifying auth chain (default -> greenland job role)..."
echo ""
echo "  Caller identity:"
aws sts get-caller-identity --profile "$PROFILE"
echo ""

echo "============================================"
echo " ✅ Auth complete!"
echo "============================================"
echo ""
echo "Next: ./scripts/greenland-connect.sh tunnel"
