#!/bin/bash
set -e

EXTERNAL_IP="${1:-$EXTERNAL_IP}"

if [ -z "$EXTERNAL_IP" ]; then
  echo "Usage: $0 <EXTERNAL_IP>"
  echo "   or: EXTERNAL_IP=<ip> $0"
  exit 1
fi

BASE_URL="http://$EXTERNAL_IP/predict"

run_test() {
  local label="$1"
  local path="$2"
  printf "%-35s -> " "$label"
  curl -s -X POST "$BASE_URL" -F "image=@$path" | python3 -c "import sys,json; print(json.load(sys.stdin)['prediction'])"
}

echo "=== CIFAR-10 (in-distribution) ==="
for cls in airplane automobile bird cat deer dog frog horse ship truck; do
  run_test "$cls" "test_images/cifar10/${cls}.png"
done

echo ""
echo "=== Out-of-distribution ==="
for num in 1 2 3; do
  run_test "flower_${num}" "test_images/out_of_distribution/flower_${num}.jpg"
done
for num in zero three seven; do
  run_test "digit_${num} (MNIST)" "test_images/out_of_distribution/digit_${num}.png"
done