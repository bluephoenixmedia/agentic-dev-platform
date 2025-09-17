# This tool defines a function for executing shell commands inside a specified Docker container.
# It is used by the Coder agent to perform real development tasks.

import docker
import traceback
from docker.errors import DockerException

def run_shell_command(command: str, container_name: str = "openhands_runtime") -> tuple[int, str]:
    """
    Executes a shell command inside a running Docker container.

    Args:
        command: The shell command to execute.
        container_name: The name of the target container. Defaults to "openhands_runtime".

    Returns:
        A tuple containing the exit code and the combined stdout/stderr of the command.
    """
    try:
        # Connect to the Docker daemon using the exposed TCP socket.
        client = docker.DockerClient(base_url='tcp://host.docker.internal:2375')
        
        # Get the target container.
        container = client.containers.get(container_name)
        
        # Execute the command.
        exit_code, (stdout, stderr) = container.exec_run(command, demux=True)
        
        # Decode and combine the output.
        output = (stdout.decode('utf-8') if stdout else '') + (stderr.decode('utf-8') if stderr else '')
        
        return exit_code, output

    except DockerException as e:
        # NEW: Print the full traceback to the console in real-time when a Docker error occurs.
        print("\n--- REAL-TIME ERROR: DockerException ---")
        traceback.print_exc()
        print("------------------------------------------\n")
        return 1, f"An unexpected error occurred: {e}"
    except Exception as e:
        # NEW: Print the full traceback for any other general errors.
        print("\n--- REAL-TIME ERROR: General Exception ---")
        traceback.print_exc()
        print("------------------------------------------\n")
        return 1, f"A general error occurred: {e}"

