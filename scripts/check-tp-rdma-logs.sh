#!/bin/bash
# Prove that both ranks used the requested RDMA provider without transport or
# kernel failures. Shared by timed results and pre-timing diagnostic gates.
set -euo pipefail

COORD_LOG=${1:?usage: check-tp-rdma-logs.sh COORD_LOG WORKER_LOG RDMA_PROFILE [COORD_DEVICE [RDMA_GID_INDEX [WORKER_DEVICE [RUN_ID]]]]}
WORKER_LOG=${2:?missing worker log}
RDMA_PROFILE=${3:-odinlink}
RDMA_DEVICE=${4:-}
RDMA_GID_INDEX=${5:-}
WORKER_RDMA_DEVICE=${6:-$RDMA_DEVICE}
EXPECTED_RUN_ID=${7:-}

for path in "$COORD_LOG" "$WORKER_LOG"; do
  [[ -r $path ]] || { echo "error: missing TP log: $path" >&2; exit 1; }
done
if [[ -n $EXPECTED_RUN_ID ]]; then
  for log in "$COORD_LOG" "$WORKER_LOG"; do
    grep -qF "ds4-tp: benchmark run_id=$EXPECTED_RUN_ID" "$log" || {
      echo "error: TP log does not bind benchmark run ID $EXPECTED_RUN_ID: $log" >&2
      exit 1
    }
  done
fi

grep -q 'worker connected, transport=rdma' "$COORD_LOG" || {
  echo "error: diagnostic did not use coordinator RDMA" >&2; exit 1;
}
grep -q 'leader connected, transport=rdma' "$WORKER_LOG" || {
  echo "error: diagnostic did not use worker RDMA" >&2; exit 1;
}

case $RDMA_PROFILE in
  odinlink)
    grep -q '"fallback_calls":0' "$COORD_LOG" || {
      echo "error: coordinator OdinLink provider reported fallback traffic" >&2; exit 1;
    }
    grep -q '"fallback_calls":0' "$WORKER_LOG" || {
      echo "error: worker OdinLink provider reported fallback traffic" >&2; exit 1;
    }
    ;;
  roce-v2)
    [[ $RDMA_DEVICE == mlx5_* && $WORKER_RDMA_DEVICE == mlx5_* &&
       $RDMA_GID_INDEX =~ ^[0-9]+$ ]] || {
      echo "error: RoCE v2 validation requires both mlx5 devices and a numeric GID index" >&2
      exit 2
    }
    for item in "$COORD_LOG:$RDMA_DEVICE" "$WORKER_LOG:$WORKER_RDMA_DEVICE"; do
      log=${item%%:*}
      device=${item#*:}
      if ! { grep -qF "rdma device $device " "$log" &&
             grep -qF "rdma GID index $RDMA_GID_INDEX (RoCE v2)" "$log" &&
             grep -q 'mlx5 queue pair uses RC' "$log" &&
             grep -q 'mlx5 registered host slab as 3 MRs' "$log"; }; then
        echo "error: log does not prove $device/GID $RDMA_GID_INDEX RoCE v2 RC with segmented MR: $log" >&2
        exit 1
      fi
      ! grep -q 'rdma device odl_tb5_' "$log" || {
        echo "error: RoCE run unexpectedly used OdinLink: $log" >&2
        exit 1
      }
    done
    ;;
  ib-mlx4)
    [[ -n $RDMA_DEVICE ]] || {
      echo "error: ib-mlx4 profile requires the RDMA device name as the 4th argument" >&2
      exit 2
    }
    for log in "$COORD_LOG" "$WORKER_LOG"; do
      if ! { grep -qF "rdma device $RDMA_DEVICE " "$log" &&
             grep -Eq "rdma GID index ${RDMA_GID_INDEX:-0}([[:space:]]|$)" "$log" &&
             grep -q 'registered slab as 1 MR' "$log" &&
             grep -qF "$RDMA_DEVICE queue pair uses RC" "$log" &&
             grep -qF 'rdma decode message policy 16384 bytes (generic provider)' "$log"; }; then
        echo "error: log does not prove mlx4 native-InfiniBand with single-MR registration: $log" >&2
        exit 1
      fi
      ! grep -q 'rdma device odl_tb5_' "$log" || {
        echo "error: InfiniBand run unexpectedly used OdinLink: $log" >&2
        exit 1
      }
      ! grep -q 'RoCE v2' "$log" || {
        echo "error: InfiniBand run unexpectedly used a RoCE v2 GID: $log" >&2
        exit 1
      }
    done
    ;;
  *)
    echo "error: unknown RDMA profile: $RDMA_PROFILE" >&2
    exit 2
    ;;
esac

if grep -Eqi 'timeout waiting|transport failed|decode .* failed|kernel (launch )?failed|nan detected' \
     "$COORD_LOG" "$WORKER_LOG"; then
  echo "error: TP logs contain a transport, decode, or kernel failure" >&2
  exit 1
fi

for log in "$COORD_LOG" "$WORKER_LOG"; do
  grep -q 'transport proof requested=rdma active=rdma payload_fallback_calls=0 failed=0' \
    "$log" || {
      echo "error: log lacks successful zero-payload-fallback proof: $log" >&2
      exit 1
    }
done

echo "validated_rdma_profile=$RDMA_PROFILE"
