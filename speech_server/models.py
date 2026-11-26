"""Where the model files come from: a local directory, or a Hugging Face model repository downloaded on first start.

Two setups (README, "Model files"):

1. Automatic download: `--model-repo <org>/<repo>` (or env MODEL_REPO). The repository is downloaded once into
   `--model-cache` (env MODEL_CACHE, default ~/.cache/speech-server) and reused on the next start. A private or gated
   repository needs a read token in the environment variable named by `--hf-token-env` (default HF_TOKEN). The token is
   only read from the environment and passed to the Hugging Face client in-process; it is never logged.
2. Manual: download the files yourself and pass the directory with `--model` (env MODEL_DIR).

A model directory holds spec.json, the tokenizer, mel_stats.json and the ONNX graphs spec.json names.
"""
from __future__ import annotations

import json
import os
import re

REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
REV_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}")
DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "speech-server")


class ModelError(Exception):
    """A model directory or repository that cannot be used; the message is safe to print (no token)."""


def check_model_dir(path: str) -> str:
    """The directory as an absolute path, if it holds a readable spec.json whose graph files all exist and stay inside
    it; otherwise ModelError naming what is missing."""
    d = os.path.abspath(os.path.expanduser(path))
    spec = os.path.join(d, "spec.json")
    if not os.path.isfile(spec):
        raise ModelError(f"{path}: no spec.json (is this a model directory?)")
    try:
        with open(spec, encoding="utf-8") as f:
            sj = json.load(f)
    except (OSError, ValueError) as e:
        raise ModelError(f"{path}: spec.json is not readable JSON ({type(e).__name__})") from None
    if not isinstance(sj, dict):
        raise ModelError(f"{path}: spec.json is not a JSON object")
    graphs = sj.get("graphs")
    if not isinstance(graphs, dict) or not graphs:
        raise ModelError(f"{path}: spec.json lists no graphs")

    def inside(name, what) -> str:
        # a lexical check (no absolute paths, no "..": a spec.json from a downloaded repository cannot name files
        # elsewhere on the host); symlinks inside the directory are the operator's choice and are followed
        if not isinstance(name, str) or not name or os.path.isabs(name):
            raise ModelError(f"{path}: spec.json {what} must be a relative file name")
        rel = os.path.normpath(name)
        if rel == ".." or rel.startswith(".." + os.sep) or "\0" in name:
            raise ModelError(f"{path}: spec.json {what} {name!r} points outside the model directory")
        return os.path.join(d, rel)
    vocab, front = sj.get("vocab") or {}, sj.get("frontend") or {}
    if not isinstance(vocab, dict) or not isinstance(front, dict):
        raise ModelError(f"{path}: spec.json vocab / frontend must be JSON objects")
    for what, name in (("vocab.tokenizer", vocab.get("tokenizer", "tokenizer.model")),
                       ("frontend.stats", front.get("stats") or None)):
        if name is None:
            continue
        if not os.path.isfile(inside(name, what)):
            raise ModelError(f"{path}: {what} file {name!r} is missing")
    missing = []
    for name in graphs.values():
        if not os.path.isfile(inside(name, "graph")):
            missing.append(name)
    if missing:
        # graphs a deployment never opens may be left out on purpose (a trimmed package): warn, do not refuse; the
        # engine fails loudly on the first graph it needs that is not there
        print(f"model: {len(missing)} graph file(s) listed in spec.json are not in {path} (fine for a trimmed "
              "package; the server stops at start-up if a mode needs one)", flush=True)
    return d


def download(repo: str, revision: str | None, cache_dir: str, token_env: str = "HF_TOKEN") -> str:
    """Download (or reuse) a Hugging Face model repository; returns its local directory."""
    if not isinstance(repo, str) or not REPO_RE.fullmatch(repo) or ".." in repo:
        raise ModelError(f"--model-repo {repo!r}: expected <owner>/<name>")
    if revision is not None and (not REV_RE.fullmatch(revision) or ".." in revision):
        raise ModelError(f"--model-revision {revision!r}: not a valid branch, tag or commit")
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        raise ModelError("automatic download needs the huggingface_hub package (pip install huggingface_hub)") from None
    token = os.environ.get(token_env) or None
    # one directory per repository and revision ("/" cannot occur in repository names; "@" in neither part), so a
    # changed revision never mixes files with another one
    leaf = repo.replace("/", "@") + ("@" + revision.replace("/", "%") if revision else "")
    local = os.path.join(os.path.realpath(os.path.expanduser(cache_dir)), leaf)
    print(f"model: fetching {repo}{'@' + revision if revision else ''} into {local} "
          f"(${token_env} {'set' if token else 'not set: public repository, or a saved huggingface_hub login'}; "
          "files already there are reused)", flush=True)
    try:
        os.makedirs(local, mode=0o750, exist_ok=True)
    except OSError as e:
        raise ModelError(f"cannot create {local} ({type(e).__name__}; is the model cache writable by this user?)") \
            from None
    try:
        path = snapshot_download(repo_id=repo, revision=revision, local_dir=local, token=token,
                                 repo_type="model")
    except HfHubHTTPError as e:
        code = getattr(getattr(e, "response", None), "status_code", None)
        hint = (" (check that the repository exists and that the token in $" + token_env + " can read it)"
                if code in (401, 403, 404) else "")
        raise ModelError(f"download of {repo} failed: HTTP {code}{hint}") from None
    except Exception as e:  # noqa: BLE001 (network errors, disk full, ...): name the type, never the token
        raise ModelError(f"download of {repo} failed: {type(e).__name__}") from None
    return check_model_dir(path)


def resolve(model_dir: str | None, repo: str | None, revision: str | None, cache_dir: str | None,
            token_env: str = "HF_TOKEN", what: str = "model") -> str | None:
    """--model wins over --model-repo; neither: None."""
    if model_dir:
        return check_model_dir(model_dir)
    if repo:
        return download(repo, revision, cache_dir or DEFAULT_CACHE, token_env)
    return None
