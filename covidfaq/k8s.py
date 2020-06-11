import subprocess
import structlog

from typing import Any, Optional

log = structlog.get_logger(__name__)


class ShellError(Exception):
    pass


def rollout_restart():
    log.info("restarting app")
    shell("kubectl --namespace covidfaq rollout restart deployment app")
    log.info("restart app in progress")


def shell(cmd: str, cwd: Optional[str] = None) -> Any:
    try:
        return subprocess.check_output(
            cmd, cwd=cwd, shell=True, text=True, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        raise ShellError(e.output)
