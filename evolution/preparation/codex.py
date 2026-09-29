"""Install KDA skills and guard hooks into a generated Codex worktree."""

from __future__ import annotations

import json
from pathlib import Path

from .declare import Toolset
from .guards import (
    BANNED_PATHS_HOOK_NAME,
    PROCESS_GUARD_HOOK_NAME,
    _remove_path,
    _sanitize_banned_paths,
    _write_banned_paths_hook,
    _write_process_guard_hook,
)

CODEX_SKILLS_DIR = Path(".agents") / "skills"
CODEX_HOOKS_DIR = Path(".codex") / "hooks"
CODEX_HOOKS_JSON = Path(".codex") / "hooks.json"


def _prune_skill_dir(skills_dir: Path, allowed: set[str]) -> None:
    if not skills_dir.exists():
        return
    for entry in skills_dir.iterdir():
        if entry.name not in allowed:
            _remove_path(entry)


def install_toolset(
    worktree: Path, toolset: Toolset, *, extra_allowed_prefixes: list[str] | None = None
) -> None:
    """Prune skills and banned sources, then register Codex tool hooks."""
    extra_allowed = [str(Path(path).resolve()) for path in (extra_allowed_prefixes or [])]
    _sanitize_banned_paths(worktree, toolset.banned_paths)

    allowed = set(toolset.effective_skills)
    _prune_skill_dir(worktree / CODEX_SKILLS_DIR, allowed)
    _prune_skill_dir(worktree / ".claude" / "skills", allowed)

    include_banned_paths = bool(toolset.banned_paths)
    _write_codex_hooks_config(worktree, include_banned_paths_hook=include_banned_paths)
    hooks_dir = worktree / CODEX_HOOKS_DIR
    hooks_dir.mkdir(parents=True, exist_ok=True)
    if include_banned_paths:
        _write_banned_paths_hook(
            hooks_dir,
            toolset.banned_paths,
            allowed_prefixes=[str(worktree.resolve()), *extra_allowed],
        )
    _write_process_guard_hook(hooks_dir)


def _write_codex_hooks_config(worktree: Path, *, include_banned_paths_hook: bool) -> None:
    hooks_json = worktree / CODEX_HOOKS_JSON
    hooks_json.parent.mkdir(parents=True, exist_ok=True)
    hook_commands = [
        {
            "type": "command",
            "command": (
                'python3 "$(git rev-parse --show-toplevel)/'
                f'{CODEX_HOOKS_DIR.as_posix()}/{PROCESS_GUARD_HOOK_NAME}"'
            ),
            "timeout": 30,
            "statusMessage": "Checking process guard",
        }
    ]
    if include_banned_paths_hook:
        hook_commands.append(
            {
                "type": "command",
                "command": (
                    'python3 "$(git rev-parse --show-toplevel)/'
                    f'{CODEX_HOOKS_DIR.as_posix()}/{BANNED_PATHS_HOOK_NAME}"'
                ),
                "timeout": 30,
                "statusMessage": "Checking banned paths",
            }
        )
    hooks_json.write_text(
        json.dumps({"hooks": {"PreToolUse": [{"matcher": "*", "hooks": hook_commands}]}}, indent=2)
        + "\n"
    )
