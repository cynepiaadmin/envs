"""browser_env — A2E environment host for browser / computer-use agents.

Public surface:

    BrowserEnvPlugin   env/*  capability (episodic RL surface, scored)
    BrowserToolPlugin  tools/* capability (inference surface, same browser)
    BrowserTask / score_episode   goal definition + deterministic scoring
    make_backend       'playwright' (real) | 'scripted' (offline, deterministic)
"""

from browser_env.plugin import BrowserEnvPlugin
from browser_env.browser_tool import BrowserToolPlugin
from browser_env.tasking import BrowserTask, TaskError, score_episode
from browser_env.backend import make_backend

__all__ = [
    "BrowserEnvPlugin",
    "BrowserToolPlugin",
    "BrowserTask",
    "TaskError",
    "score_episode",
    "make_backend",
]