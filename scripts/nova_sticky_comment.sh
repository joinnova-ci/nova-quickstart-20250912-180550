#!/usr/bin/env bash
set -euo pipefail
TITLE="${1:-Nova CI-Rescue}"
BODY="${2:-}" 
TAG="<!-- nova-sticky:status -->"
pr_number="${PR_NUMBER:-${GITHUB_REF##*/}}"
cid="$(gh api repos/${GITHUB_REPOSITORY}/issues/${pr_number}/comments --jq ".[] | select(.body|contains(\"$TAG\")) | .id" | head -n1 || true)"
markdown="### ${TITLE}
${TAG}

${BODY}
"
if [ -n "${cid}" ]; then
  gh api repos/${GITHUB_REPOSITORY}/issues/comments/${cid} -X PATCH -f body="${markdown}" >/dev/null
else
  gh api repos/${GITHUB_REPOSITORY}/issues/${pr_number}/comments -f body="${markdown}" >/dev/null
fi
echo "Sticky comment updated."
