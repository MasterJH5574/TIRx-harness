"""Install KDA skills and guard hooks into a generated Claude worktree."""

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

CLAUDE_SKILLS_DIR = Path(".claude") / "skills"
CLAUDE_HOOKS_DIR = Path(".claude") / "hooks"
CLAUDE_SETTINGS = Path(".claude") / "settings.local.json"


def _prune_skill_dir(skills_dir: Path, allowed: set[str]) -> None:
    if not skills_dir.exists():
        return
    for entry in skills_dir.iterdir():
        if entry.name not in allowed:
            _remove_path(entry)


def install_toolset(
    worktree: Path, toolset: Toolset, *, extra_allowed_prefixes: list[str] | None = None
) -> None:
    """Prune skills and banned sources, then register Claude tool hooks."""
    extra_allowed = [str(Path(path).resolve()) for path in (extra_allowed_prefixes or [])]
    _sanitize_banned_paths(worktree, toolset.banned_paths)

    allowed = set(toolset.effective_skills)
    _prune_skill_dir(worktree / CLAUDE_SKILLS_DIR, allowed)
    _prune_skill_dir(worktree / ".agents" / "skills", allowed)

    hooks_dir = worktree / CLAUDE_HOOKS_DIR
    hooks_dir.mkdir(parents=True, exist_ok=True)
    include_banned_paths = bool(toolset.banned_paths)
    if include_banned_paths:
        _write_banned_paths_hook(
            hooks_dir,
            toolset.banned_paths,
            allowed_prefixes=[str(worktree.resolve()), *extra_allowed],
        )
    _write_process_guard_hook(hooks_dir)
    _write_claude_settings(worktree, include_banned_paths_hook=include_banned_paths)


def _write_claude_settings(worktree: Path, *, include_banned_paths_hook: bool) -> None:
    settings_path = worktree / CLAUDE_SETTINGS
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    except json.JSONDecodeError:
        settings = {}

    pre_tool_use = settings.setdefault("hooks", {}).setdefault("PreToolUse", [])
    if include_banned_paths_hook:
        _append_hook(
            pre_tool_use,
            matcher="Read|Grep|Glob|Bash|WebFetch|NotebookRead|NotebookEdit",
            command=f"{CLAUDE_HOOKS_DIR.as_posix()}/{BANNED_PATHS_HOOK_NAME}",
        )
    _append_hook(
        pre_tool_use,
        matcher="Bash",
        command=f"{CLAUDE_HOOKS_DIR.as_posix()}/{PROCESS_GUARD_HOOK_NAME}",
    )
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")


def _append_hook(pre_tool_use: list, *, matcher: str, command: str) -> None:
    for entry in pre_tool_use:
        for hook in entry.get("hooks", []) or []:
            if hook.get("type") == "command" and hook.get("command") == command:
                return
    pre_tool_use.append({"matcher": matcher, "hooks": [{"type": "command", "command": command}]})
