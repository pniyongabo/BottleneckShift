#!/usr/bin/env bash
# Build the technical report with Tectonic: paper/bottleneckshift-report.pdf.
set -euo pipefail
cd "$(dirname "$0")"
tectonic -X compile main.tex --keep-logs
mv main.pdf bottleneckshift-report.pdf
echo "built paper/bottleneckshift-report.pdf"
