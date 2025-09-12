# This file defines the tool for executing shell commands within the OpenHands runtime container.
# It uses the Docker SDK to connect to the Docker daemon and run commands in the specified container.

import docker
from typing import Tuple

def run_shell_command(command: str) -> Tuple[int, str]:
    """
    Executes a shell command inside the 'openhands_runtime' Docker container.

    Args:
        command: The shell command to execute.

    Returns:
        A tuple containing the exit code and the combined stdout/stderr of the command.
    """
    try:
        # Connect to the Docker daemon from within the orchestrator container.
        # The '/var/run/docker.sock' is mounted, allowing this connection.
        client = docker.from_env()
        
        # Get the specific container where the command needs to run.
        container = client.containers.get("openhands_runtime")
        
        # Execute the command. The `workdir` ensures it runs in the correct directory.
        # `demux=True` separates stdout and stderr, though we combine them here.
        exit_code, (stdout, stderr) = container.exec_run(
            cmd=command,
            workdir="/opt/workspace_base"
        )
        
        # Combine stdout and stderr for a complete log.
        output = (stdout.decode('utf-8') if stdout else '') + (stderr.decode('utf-8') if stderr else '')
        
        return exit_code, output

    except docker.errors.NotFound:
        return 1, "Error: The 'openhands_runtime' container was not found."
    except Exception as e:
        return 1, f"An unexpected error occurred: {e}"
