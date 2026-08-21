#!/usr/bin/env bash
#
# PreToolUse hook for Read|Write|Edit.
# Denies file_path targets outside the project root, /tmp, and $HOME/.claude.

input=$(cat)
fp=$(echo "$input" | jq -r '.tool_input.file_path // empty')
[ -z "$fp" ] && exit 0

root=$(realpath "$CLAUDE_PROJECT_DIR")
claude_dir=$(realpath "$HOME/.claude" 2>/dev/null)
rp=$(realpath -m "$fp" 2>/dev/null)

case "$rp" in
  "$root"/*|"$root")
    exit 0 ;;
  /tmp/*|/tmp)
    exit 0 ;;
  "$claude_dir"/*|"$claude_dir")
    exit 0 ;;
  *)
    echo '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"Read/Write/Edit outside the project root is not allowed."}}'
    ;;
esac
