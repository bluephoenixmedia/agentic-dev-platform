Agentic Developer Platform — Design Document (v0.1, 2025-09-08)
Recommendation: Build on LangGraph for orchestration and OpenHands for the execution/runtime workspace.

This document specifies goals, requirements, architecture, state models, agent roles, prompt strategy, APIs, security, observability, roadmap, and test plan.

1) Executive Summary
We will implement a multi-agent, design-doc–first developer platform that:

Connects to an LLM via API key and maintains a canonical design document as a durable reference throughout the project lifecycle.

Plans a development roadmap (phases → tasks → subtasks), then executes work in a real filesystem using a sandboxed runtime.

Agents propose code changes, update files, run code/tests, stream logs, and push changes to git behind human approval gates.

Uses LangGraph for stateful orchestration (graph semantics, checkpointers, interrupts/HITL) and OpenHands for an execution workspace (mountable /workspace, terminal, file diffs, headless API).

Non-goals include building an IDE from scratch or an end-to-end CI service; we integrate with existing tools (VS Code UI in OpenHands, GitHub/GitLab CI).

2) Goals & Non-Goals
Goals
Design-doc–first workflow: The design document is a single source of truth (functional + technical spec) used by all agents.

Deterministic, resumable orchestration: Durable state, reproducible runs, human-in-the-loop checkpoints.

Sandboxed execution with shared mount: Agents edit files in /workspace, run code/tests, monitor logs, and propose/commit changes.

Role-specialized agents: Documentation, planning, architecture, coding, testing, CI/CD, and log analysis.

Prompt modularity: Role prompts and global conventions (microagents) are versioned in-repo.

Security, auditability, and governance: Human gates before destructive or externally visible actions; audit logs; minimal secrets exposure.

Measurable performance: Track cycle time, test pass rate, PR acceptance rate, incident rate, escape defects.

Non-Goals
Replacing CI systems (use GitHub/GitLab/Buildkite).

Replacing code review culture—humans retain final authority.

Building a general agent marketplace—focus on software engineering flows.