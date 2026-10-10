# Krishi Mitra — Project Rules

- Always inspect the existing code before making changes.
- Do not rebuild the project from scratch.
- Preserve existing frontend pages, routing, authentication, chat, weather, mandi and language features.
- Reuse the existing FastAPI backend and plant-disease detection flow wherever possible.
- Never use random or mock predictions as genuine ML results.
- Keep plant-disease detection and crop recommendation as separate ML features.
- Never expose API keys, passwords or `.env` values.
- Do not commit datasets, virtual environments, dependency folders or model binaries to Git.
- Do not invent model accuracy, live API results or successful tests.
- After each implementation stage, run relevant tests and check the frontend build.
- Update README.md and PROJECT_STRUCTURE.md to reflect the actual code.
- Before modifying files, inspect Git status and preserve existing user changes.
- Explain every new dependency, API endpoint, model file and setup command.