# Hugging Face Hub operations for the Go server's "hf" remote kind.
# Reads one JSON request on stdin, writes one JSON result on stdout. The token
# arrives in the HF_TOKEN environment variable, never on the command line.
import base64
import json
import os
import shutil
import sys
import tempfile


def not_found(exc):
    names = {c.__name__ for c in type(exc).__mro__}
    return bool(names & {"EntryNotFoundError", "RemoteEntryNotFoundError", "RepositoryNotFoundError",
                         "RevisionNotFoundError", "FileNotFoundError"})


def modified(entry):
    commit = getattr(entry, "last_commit", None)
    date = getattr(commit, "date", None) if commit is not None else None
    return date.isoformat() if date is not None and hasattr(date, "isoformat") else None


def main():
    req = json.load(sys.stdin)
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    repo, rtype, rev = req["repo_id"], req.get("repo_type") or "model", req.get("revision") or "main"
    op = req["op"]

    def ensure_repo():
        api.create_repo(repo_id=repo, repo_type=rtype, private=req.get("private", True), exist_ok=True)

    def url(path):
        kind = "" if rtype == "model" else rtype + "s/"
        return f"{api.endpoint}/{kind}{repo}/resolve/{rev}/{path}"

    if op == "check":
        ensure_repo()
        probe = f".aitk-check-{os.getpid()}"
        api.upload_file(path_or_fileobj=b"ok", path_in_repo=probe, repo_id=repo, repo_type=rtype,
                        revision=rev, commit_message="aitk: remote check")
        api.delete_file(path_in_repo=probe, repo_id=repo, repo_type=rtype, revision=rev,
                        commit_message="aitk: remote check cleanup")
        return {"ok": True}

    if op in ("put", "put_bytes"):
        ensure_repo()
        data = req["local_path"] if op == "put" else base64.b64decode(req["data"])
        api.upload_file(path_or_fileobj=data, path_in_repo=req["key"], repo_id=repo, repo_type=rtype,
                        revision=rev, commit_message=req.get("message") or f"aitk: upload {req['key']}")
        return {"url": url(req["key"])}

    if op in ("get", "get_bytes"):
        tmp = tempfile.mkdtemp(prefix="aitk-hf-")
        try:
            got = api.hf_hub_download(repo_id=repo, filename=req["key"], repo_type=rtype, revision=rev, local_dir=tmp)
            if op == "get_bytes":
                with open(got, "rb") as f:
                    return {"data": base64.b64encode(f.read()).decode()}
            dest = req["local_path"]
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            part = dest + ".part"
            shutil.move(got, part)
            os.replace(part, dest)
            return {"ok": True}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    if op == "stat":
        infos = api.get_paths_info(repo_id=repo, paths=[req["key"]], repo_type=rtype, revision=rev, expand=True)
        files = [i for i in infos if getattr(i, "size", None) is not None and getattr(i, "path", None) == req["key"]]
        if not files:
            return {"error": f"{req['key']} not found", "not_found": True}
        return {"size": files[0].size, "modified": modified(files[0])}

    if op == "list":
        base = req.get("base") or ""
        try:
            entries = api.list_repo_tree(repo_id=repo, path_in_repo=base or None, recursive=True,
                                         repo_type=rtype, revision=rev, expand=True)
            items = []
            for e in entries:
                if getattr(e, "size", None) is None:  # folders
                    continue
                items.append({"path": e.path, "size": e.size, "modified": modified(e)})
        except Exception as exc:  # a missing folder or repo lists as empty
            if not_found(exc):
                return {"items": []}
            raise
        return {"items": items}

    if op == "delete":
        try:
            api.delete_file(path_in_repo=req["key"], repo_id=repo, repo_type=rtype, revision=rev,
                            commit_message=req.get("message") or f"aitk: delete {req['key']}")
        except Exception as exc:
            if not not_found(exc):
                raise
        return {"ok": True}

    return {"error": f"unknown op {op!r}"}


if __name__ == "__main__":
    try:
        out = main()
    except ImportError as exc:
        out = {"error": f"huggingface_hub is not installed in this Python ({exc})"}
    except Exception as exc:  # report, never a traceback on stdout
        out = {"error": f"{type(exc).__name__}: {exc}", "not_found": not_found(exc)}
    sys.stdout.write("\n" + json.dumps(out) + "\n")
