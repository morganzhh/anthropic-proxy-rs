#!/usr/bin/env bash

set -euo pipefail

DEFAULT_REPO_URL="https://github.com/m0n0x41d/anthropic-proxy-rs"
DEFAULT_PACKAGE_NAME="anthropic-proxy"

usage() {
    cat <<'EOF'
Usage: install.sh [options]

Install anthropic-proxy from git using cargo.

Options:
  --repo <url>      Git repository URL
  --root <dir>      Cargo install root
  --branch <name>   Install from a git branch
  --tag <name>      Install from a git tag
  --rev <sha>       Install from a git revision
  --force           Reinstall even if already installed
  -h, --help        Show this help

Environment overrides:
  ANTHROPIC_PROXY_INSTALL_REPO
  ANTHROPIC_PROXY_INSTALL_ROOT
  ANTHROPIC_PROXY_INSTALL_BRANCH
  ANTHROPIC_PROXY_INSTALL_TAG
  ANTHROPIC_PROXY_INSTALL_REV
  ANTHROPIC_PROXY_INSTALL_FORCE=1
EOF
}

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 127
    fi
}

repo_url="${ANTHROPIC_PROXY_INSTALL_REPO:-$DEFAULT_REPO_URL}"
install_root="${ANTHROPIC_PROXY_INSTALL_ROOT:-}"
git_branch="${ANTHROPIC_PROXY_INSTALL_BRANCH:-}"
git_tag="${ANTHROPIC_PROXY_INSTALL_TAG:-}"
git_rev="${ANTHROPIC_PROXY_INSTALL_REV:-}"
force_install="${ANTHROPIC_PROXY_INSTALL_FORCE:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)
            repo_url="${2:?missing value for --repo}"
            shift 2
            ;;
        --root)
            install_root="${2:?missing value for --root}"
            shift 2
            ;;
        --branch)
            git_branch="${2:?missing value for --branch}"
            shift 2
            ;;
        --tag)
            git_tag="${2:?missing value for --tag}"
            shift 2
            ;;
        --rev)
            git_rev="${2:?missing value for --rev}"
            shift 2
            ;;
        --force)
            force_install=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

need_cmd cargo

selected_refs=0
[[ -n "$git_branch" ]] && selected_refs=$((selected_refs + 1))
[[ -n "$git_tag" ]] && selected_refs=$((selected_refs + 1))
[[ -n "$git_rev" ]] && selected_refs=$((selected_refs + 1))

if [[ $selected_refs -gt 1 ]]; then
    echo "Use only one of --branch, --tag, or --rev" >&2
    exit 1
fi

cmd=(cargo install --locked --git "$repo_url")

if [[ -n "$install_root" ]]; then
    cmd+=(--root "$install_root")
fi

if [[ -n "$git_branch" ]]; then
    cmd+=(--branch "$git_branch")
elif [[ -n "$git_tag" ]]; then
    cmd+=(--tag "$git_tag")
elif [[ -n "$git_rev" ]]; then
    cmd+=(--rev "$git_rev")
fi

if [[ "$force_install" == "1" ]]; then
    cmd+=(--force)
fi

cmd+=("$DEFAULT_PACKAGE_NAME")

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"

if [[ -n "$install_root" ]]; then
    binary_path="$install_root/bin/$DEFAULT_PACKAGE_NAME"
else
    binary_path="${CARGO_HOME:-$HOME/.cargo}/bin/$DEFAULT_PACKAGE_NAME"
fi

echo "Installed $DEFAULT_PACKAGE_NAME to $binary_path"
