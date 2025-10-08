Agentic Developer Platform
This project implements a multi-agent, design-doc-first developer platform that automates software development tasks. It uses a local Ollama LLM for intelligent planning and execution, orchestrated by LangGraph, within a secure Dockerized environment.

High-Level Architecture
The system is composed of three primary Docker services: an orchestrator that manages the agent workflow, an openhands runtime that provides a sandboxed environment for execution, and an ollama service that runs the local LLM. The agents within the orchestrator connect to the local Ollama service to generate roadmaps and execute development tasks.

graph TD
    subgraph "Development Environment (Local Machine)"
        A[User] -- Manages & Approves --> VSC[VS Code Dev Container]
        VSC -- Runs inside --> O[Orchestrator Container]
        O -- Executes --> GS[graph_skeleton.py]
    end

    subgraph "Docker Services (via docker-compose)"
        O -- Python Docker SDK --> D[Docker Socket via TCP]
        D -- Executes commands in --> OH[OpenHands Container]
        OH -- Mounts --> W[Workspace Volume]
        O -- HTTP Requests --> OLLAMA[Ollama Container]
    end
    
    subgraph "Agent Workflow (LangGraph in Orchestrator)"
        LG_Start[Start] --> DocAgent
        DocAgent --> Planner
        Planner -- design_doc --> OLLAMA
        OLLAMA -- roadmap --> Planner
        Planner --> Architect
        Architect --> Coder
        Coder -- task --> OLLAMA
        OLLAMA -- plan_and_commands --> Coder
        Coder -- run_shell --> OH
        OH -- output --> Coder
        Coder --> Tester
        Tester --> CICD
        CICD --> Loop{Continue?}
        Loop -- Yes --> Architect
        Loop -- No --> LG_End[End]
    end

    style VSC fill:#007ACC,color:#fff
    style OLLAMA fill:#2E2E2E,color:#fff
    style OH fill:#f6ae2d,color:#333

Prerequisites
Docker Desktop: Ensure the Docker engine is running.

Important: In Settings > General, you must enable the option to "Expose daemon on tcp://localhost:2375 without TLS".

VS Code: The recommended IDE for this project.

VS Code Dev Containers Extension: For a seamless, containerized development experience.

Getting Started
1. Configure Your Environment
Clone the Repository: Download this project to your local machine.

Create the .env file: Create a file named .env in the project root with the following content:

OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=agentic-llama3

Create the Modelfile: Create a file named Modelfile in the project root to define your custom Ollama model:

FROM llama3:8b
PARAMETER num_ctx 8192
PARAMETER temperature 0.6

2. Launch the Development Environment
The easiest way to get started is to use the VS Code Dev Container.

Open the Project in VS Code: Open the main agentic-dev-platform folder.

Reopen in Container: Wait for a pop-up in the bottom-right corner and click "Reopen in Container". This will build the Docker services and connect your VS Code instance directly into the orchestrator container.

3. Prepare the Ollama Model (First-Time Setup)
After the container starts, you need to download and create your custom Ollama model.

Open a Local Terminal: Use Terminal -> New Terminal in VS Code to open a new terminal. This will be a local terminal (e.g., PowerShell), not the one inside the container.

Pull the Base Model (This will take several minutes):

docker-compose exec ollama ollama pull llama3:8b

Create the Custom Model:

docker-compose exec ollama ollama create agentic-llama3 -f /Modelfile

4. Run the Agent Workflow
Once you are inside the Dev Container (the bottom-left corner of VS Code will be green), you can run the main agent script.

Open a Terminal: Use Terminal -> New Terminal in VS Code.

Execute the Script:

python /workspaces/agentic-dev-platform/orchestrator/graph_skeleton.py

5. Review the Output
Run Log: A high-level summary of the agents' actions will be written to run_log.txt.

Workspace: The implementation plans and any files created by the Coder agent will be saved in the workspace/ directory.

Roadmap: The persistent state of the project roadmap is saved in roadmap.json.