# GitHub setup and push workflow

Create an empty GitHub repository without adding a README/license/gitignore (this project already has them). Then from the project directory:

```bash
git init
git branch -M main
git add README.md .gitignore docs/PROJECT_PLAN.md docs/PROVENANCE.md
git commit -m "docs: define QPredRNN research scope and provenance"
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
git push -u origin main
```

Continue the commit plan in `COMMIT_PLAN.md`. Before each push:
```bash
git status
git diff --check
pytest -q
```

If the remote already has commits, clone it first or pull/reconcile rather than force-pushing. Never commit `.env`, API keys, dataset files, checkpoints, or private credentials. Use Git LFS only if you have a deliberate policy for model artifacts; usually store large checkpoints in cloud/object storage and link to release artifacts.
