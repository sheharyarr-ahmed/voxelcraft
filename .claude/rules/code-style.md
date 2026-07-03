# Code style

Applies to `app.py`, `src/`, and `tests/`. Formatting and type checks gate every phase; a dirty
`black --check` or a failing `mypy --strict` blocks the commit, no exceptions.

## Formatting

- black, line length 100. isort with `profile=black` so the two never fight over import wrapping.
- Config lands in `pyproject.toml` under `[tool.black]` / `[tool.isort]` once that file is created.
  Until then, pass flags explicitly: `black --line-length 100 src tests app.py` and
  `isort --profile black src tests app.py` (see CLAUDE.md Commands).
- Never hand-format around black. If a line reads badly after formatting, restructure the code,
  not the formatter output.

## Types

- `mypy --strict src` must pass. Every public function is fully annotated — parameters and return
  type, including `-> None`.
- `# type: ignore` requires an error code and a trailing reason:
  `# type: ignore[attr-defined]  # diffusers ships no stubs for load_lora_weights`.
  A bare ignore is a review reject.
- Stub gaps in third-party packages (diffusers, gradio, controlnet_aux) get per-module overrides
  in mypy config, not blanket ignores scattered through source.

## Python 3.10+ idioms

- `X | None`, never `Optional[X]`. Builtin generics: `list[str]`, `dict[str, float]`.
- `pathlib.Path` over `os.path` for anything touching the filesystem — upload temp files,
  cache directories.
- f-strings for interpolation; no `%` formatting, no `.format()`.
- `match` only where it clarifies (dispatching on scheduler name is a fair case); a two-arm
  branch stays an `if`.

## Errors

- No bare `except:`. Catch the narrowest exception that can actually occur (`ValidationError`,
  `OSError`, `torch.cuda.OutOfMemoryError`) — `except Exception` only when re-raising.
- Never swallow silently. Handle and log why continuing is safe, or re-raise. An empty
  `except ...: pass` does not merge.
- `src/` raises typed project exceptions (`LoraLoadError`, `UploadValidationError`, and friends).
  Only the Gradio layer in `app.py` maps them to `gr.Error`; pipeline code never imports gradio.

## Logging

- The `logging` module in `src/`, never `print`. One `logger = logging.getLogger(__name__)` at
  module top; Spaces captures stdout/stderr, so structured log lines are what you get to debug with.
- Device selection, model/LoRA load timings, and per-generation inference time log at INFO.

## Structure

- Absolute imports rooted at `src`: `from src.pipelines.lora_manager import apply_loras`.
  No relative imports.
- No work at import time (decision D7): module level is constants, type definitions, and logger
  setup only. Model loading happens inside functions, on first call.
- Every module opens with a docstring stating its role in the pipeline, one or two sentences.
- Public functions get a one-line imperative docstring ("Validate an uploaded pose image against
  the format allowlist and 5MB cap."). Skip param-by-param blocks the signature already states.
- Functions stay under roughly 50 lines. Past that, extract a helper — validation and metadata
  bookkeeping usually want their own functions anyway.

## Tests

- pytest with plain functions, no test classes. `@pytest.mark.parametrize` for input matrices:
  LoRA weight bounds, upload format allowlist, prompt token limits.
- `tmp_path` fixture over `tempfile` for anything that writes to disk.
- Test names state the behavior under test: `test_generation_request_rejects_weight_above_1_5`,
  not `test_weights_2`.
