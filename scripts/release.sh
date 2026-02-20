#!/usr/bin/env bash
set -euo pipefail

# ─── Usage ───
# ./scripts/release.sh          # patch (0.1.0 → 0.1.1)
# ./scripts/release.sh -mi      # minor (0.1.0 → 0.2.0)
# ./scripts/release.sh -ma      # major (0.1.0 → 1.0.0)

BUMP="patch"
case "${1:-}" in
  -mi) BUMP="minor" ;;
  -ma) BUMP="major" ;;
  -h|--help)
    echo "Usage: $0 [-mi | -ma]"
    echo "  (default)  patch bump"
    echo "  -mi        minor bump"
    echo "  -ma        major bump"
    exit 0
    ;;
esac

cd "$(git rev-parse --show-toplevel)"

# ─── Preflight checks ───
if [ -n "$(git status --porcelain)" ]; then
  echo "ERROR: Working tree is dirty. Commit or stash changes first."
  exit 1
fi

if ! command -v gh &>/dev/null; then
  echo "ERROR: gh CLI not found. Install: https://cli.github.com"
  exit 1
fi

# ─── Version bump ───
OLD_VERSION=$(node -p "require('./packages/core/package.json').version")
cd packages/core
npm version "$BUMP" --no-git-tag-version
NEW_VERSION=$(node -p "require('./package.json').version")
cd ../..

echo "Version: $OLD_VERSION → $NEW_VERSION"

# ─── Build ───
echo "Building..."
npx turbo run build

# ─── Generate release notes from commits since last tag ───
LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
if [ -n "$LAST_TAG" ]; then
  NOTES=$(git log "$LAST_TAG..HEAD" --oneline --no-decorate)
else
  NOTES=$(git log --oneline --no-decorate -20)
fi

# ─── Commit, tag ───
TAG="v$NEW_VERSION"
git add packages/core/package.json
git commit -m "Release $TAG"
git tag -a "$TAG" -m "Release $TAG"

echo ""
echo "=== Release $TAG ==="
echo ""
echo "$NOTES"
echo ""

# ─── Publish to npm ───
read -rp "Publish to npm? [y/N] " CONFIRM
if [[ "$CONFIRM" =~ ^[yY]$ ]]; then
  cd packages/core
  npm publish --access public
  cd ../..
  echo "Published to npm."
else
  echo "Skipped npm publish."
fi

# ─── Push tag ───
read -rp "Push tag and create GitHub release? [y/N] " CONFIRM
if [[ "$CONFIRM" =~ ^[yY]$ ]]; then
  git push
  git push origin "$TAG"

  # ─── GitHub Release ───
  gh release create "$TAG" \
    --title "$TAG" \
    --notes "$(cat <<EOF
## Changes since ${LAST_TAG:-initial}

$NOTES

**Full Changelog**: https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/compare/${LAST_TAG:-$(git rev-list --max-parents=0 HEAD | head -1)}...$TAG
EOF
)"

  echo "GitHub release created: $TAG"
else
  echo "Skipped push. To push manually:"
  echo "  git push && git push origin $TAG"
fi
