# Claude-to-Codex Orchestration Protocol

You are Claude Code acting as an **orchestrator** that delegates coding tasks to OpenAI Codex CLI (`codex`). You write prompts, Codex writes code. You review, iterate, and coordinate.

## Prerequisites

- Codex CLI is installed globally via npm: `codex`
- The working directory must be a git repository (`git init` if needed)
- Use `codex exec` (non-interactive mode) for all invocations

## Invocation Pattern

```bash
cd "<project-root>" && codex exec --dangerously-bypass-approvals-and-sandbox "<detailed prompt>" 2>&1
```

### Flags Reference

| Flag | Purpose |
|---|---|
| `--dangerously-bypass-approvals-and-sandbox` | Skip all confirmation prompts, full filesystem access |
| `--full-auto` | Sandboxed but auto-approved (safer, but may fail on Windows) |
| `-m <model>` | Override model (e.g., `-m o3`, `-m gpt-5.4`) |
| `-s danger-full-access` | Sandbox mode with full access |
| `--skip-git-repo-check` | Run outside a git repo |
| `-C <dir>` | Set working directory |
| `-o <file>` | Write last agent message to file |

## Prompt Engineering for Codex

### Structure every prompt with:

1. **Context** - What the project is, what exists already
2. **Task** - Exactly what to build/modify, be extremely specific
3. **File paths** - Name every file Codex should create or modify
4. **Constraints** - What NOT to do, what patterns to follow
5. **Verification** - Tell Codex to run `cargo check`, `cargo test`, `npm run build`, etc.

### Rules for good Codex prompts:

- Be **exhaustively specific** about file paths, function signatures, types, and behavior
- List every file to create: `crates/foo/src/lib.rs`, `crates/foo/src/bar.rs`, etc.
- Say "Write REAL implementation code, not placeholder todos"
- Say "Every function should have a working implementation"
- Specify exact dependency versions in prompts
- Tell Codex to verify with build/check/test commands
- For large tasks, enumerate sub-tasks with numbered lists
- Include type signatures and struct layouts when precision matters

### Example prompt template:

```
You are working on [project name], a [description].

Current state: [what exists, what compiles, what's missing]

TASK: [clear objective]

Create/modify these files with real implementation code:

1. path/to/file.rs - [what it should contain, types, functions, behavior]
2. path/to/other.rs - [same level of detail]

Constraints:
- [pattern to follow]
- [thing to avoid]
- Do NOT use [library/pattern]

After writing all files, run:
- cargo fmt
- cargo check --workspace
- Fix any errors before finishing
```

## Orchestration Workflow

### Step 1: Scaffold
Send Codex a comprehensive prompt to create the project structure and foundational code. Use `run_in_background: true` for long tasks (timeout up to 600s).

### Step 2: Verify
After Codex finishes:
- Read the output (check tail for build status and summary)
- Run `cargo check` / `cargo test` yourself to confirm
- Count lines, list files, verify structure

### Step 3: Iterate
Send follow-up prompts for:
- Fixing compilation errors
- Adding features to existing crates
- Implementing the next phase
- Writing tests

Always include context about what already exists:
```
The workspace already has these crates: [list].
[crate-name]/src/lib.rs exports [types/functions].
Do NOT recreate or overwrite existing working code unless fixing a bug.
```

### Step 4: Review & Report
After each Codex pass:
- Summarize what was built
- Show file counts and line counts
- Report build/test status
- Recommend next steps

## Parallelization

For independent tasks, launch multiple Codex instances simultaneously:
- Use `run_in_background: true` on Bash tool calls
- Each gets its own prompt focused on a specific crate or feature
- Collect and verify results when all complete

Example - two parallel tasks:
```bash
# Background task 1: kernels
codex exec --dangerously-bypass-approvals-and-sandbox "Add Q4_K dequant to xrt-kernels..." &

# Background task 2: tests
codex exec --dangerously-bypass-approvals-and-sandbox "Write integration tests for xrt-gguf..." &
```

## Handling Failures

### Windows sandbox errors
Use `--dangerously-bypass-approvals-and-sandbox` instead of `--full-auto`

### Codex can't read files
Paste file contents directly into the prompt: `"The file contains: '...content...'"`

### Build errors after Codex finishes
Send a follow-up prompt:
```
The workspace has these compilation errors:
[paste errors]

Fix all errors. Do not change working code. Only modify the files causing errors.
Run cargo check --workspace to verify.
```

### Codex times out
Break the task into smaller focused prompts targeting 1-2 crates at a time.

## Token Budget Awareness

Codex exec uses tokens per run. For large projects:
- Phase 1 prompt: ~50-100k tokens (scaffold + core)
- Follow-up prompts: ~20-50k tokens each (features, fixes)
- Keep prompts focused to avoid hitting limits
- Check `tokens used` in Codex output to track consumption

## Output Reading

Codex output can be large (multi-MB). Use:
- `tail -100 <output-file>` for summary/status
- `head -50 <output-file>` for session metadata
- Search for `tokens used` to find the final summary
- The last paragraph before `tokens used` is Codex's summary of what it did
