# Active us-west-2 diversity instances (2026-08-25) — profile greenlandw, account 144991380388

Reconnect a tunnel:
  pkill -f "localPortNumber.*<PORT>"; sleep 2
  nohup aws ssm start-session --target <SSM> --document-name AWS-StartPortForwardingSession \
    --parameters '{"portNumber":["2222"],"localPortNumber":["<PORT>"]}' --profile greenlandw --region us-west-2 &
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p <PORT> greenland-user@localhost
Creds expired -> ada credentials update --account 703671891219 --role Intern --provider isengard --once

| inst | SSM | tunnel | autopull | node | IP | cell |
|---|---|---|---|---|---|---|
| I1 | mi-05eb80e2290593395 | 1030 | 1032 | main | 10.2.181.240 | Qwen3-8B × Olympiad @2048 |
| I1 | | | | worker | 10.2.171.115 | DeepSeek-Math-7B × Omni-MATH |
| I1 | | | | worker | 10.2.193.52 | Qwen2.5-7B-Instruct × Olympiad |
| I2 | mi-0a0a489493371c215 | 1031 | 1033 | main | 10.2.217.1 | Qwen3-8B × Omni-MATH @2048 |
| I2 | | | | worker | 10.2.210.0 | Qwen2.5-7B-Instruct × Omni-MATH |
| I2 | | | | worker | 10.2.101.89 | Qwen2.5-Math-1.5B × Olympiad |

Per-node: /tmp/instance_storage/gu/cell_<name>_<dataset>/ + logs/. Pull -> runs_pulled/round2_eval/diversity/.
Workers reached from main via: ssh -p 2222 greenland-user@<workerIP>.
