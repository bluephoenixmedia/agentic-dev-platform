Agentic Developer Platform
This project is an implementation of a multi-agent, design-doc-first developer platform based on the concepts outlined in the Agentic Developer Platform Design Document.

It uses LangGraph to orchestrate a team of specialized AI agents and OpenHands as a sandboxed execution environment for code generation, testing, and other development tasks.

Prerequisites
Before you begin, ensure you have the following installed on your local machine:

Docker Desktop: To build and run the containerized services.

Visual Studio Code: As the primary IDE.

VS Code Dev Containers Extension: For connecting the IDE to the running Docker containers. You can install it from the VS Code Marketplace.

Getting Started
1. Clone the Repository
Clone this repository to your local machine.

2. Launch the Development Environment
The entire development environment is containerized to ensure consistency. The easiest way to get started is to use the VS Code Dev Containers feature.

Open the root folder of this project in Visual Studio Code.

Wait for a notification to appear in the bottom-right corner prompting you to "Reopen in Container".

Click the "Reopen in Container" button.

VS Code will use the .devcontainer/devcontainer.json and docker-compose.yml files to build the necessary Docker images and connect your IDE session directly into the orchestrator service.

3. Run the Agent Workflow
Once your VS Code window has reloaded and is connected to the container (the bottom-left corner will be green), you can run the agent workflow.

Open a new terminal in VS Code (Terminal > New Terminal).

Run the main orchestrator script:

python /workspaces/agentic-dev-platform/orchestrator/graph_skeleton.py

You will see the output of each agent executing in sequence, confirming that the system is running correctly.
