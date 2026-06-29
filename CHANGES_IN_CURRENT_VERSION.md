# Audit: what the "current" website (Resumatch_OCI) changed vs. this version (Resumatch_OCI-main)

> **Context.** `Resumatch_OCI` is the live site, evolved by hand from this folder
> (`Resumatch_OCI-main`). This document records exactly what that evolution
> changed and where it regressed. It is the input to
> [`SECURITY_AND_FUNCTIONALITY_ROADMAP.md`](SECURITY_AND_FUNCTIONALITY_ROADMAP.md).
>
> **Important finding.** The live version was **not** built on top of this
> hardened code. It was built on an *earlier, pre-security* snapshot of the
> monolith (kept in the live repo as `app_old.py`, which contains no CSRF / nh3 /
> limiter). As a result the security hardening, the token economy, the employer
> approval gate, and several robustness fixes that exist here were **lost** in the
> live version.

Date of audit: 2026-06-29.

---

## 1. Structural change (the one genuine improvement)

| Aspect | This version (was) | Live version (now) |
|---|---|---|
| Code layout | One monolithic `app.py` (~2,180 lines) | Modular Flask package: `app/` with `routes/` blueprints, `services/ai_service.py`, `models.py`, `utils/helpers.py`, `extensions.py`, app factory |
| Background jobs | `threading.Thread` + filesystem progress markers | **Celery** tasks (`process_jit_task`, `run_upload_jit_full_task`) |

The package split and the move to Celery are real architectural upgrades — they
are the "good parts" worth keeping. **This refactor branch now adopts both** (see
the new `app/` package + `celery_worker.py` here), while preserving everything
the live version dropped.

---

## 2. Regressions in the live version

### 2.1 Broken features (runtime `NameError`)
The blueprint split dropped module-level imports that the monolith had in global
scope, so two core features crash on use:

- **`/api/supersearch`** — references `load_cached_embeddings()`, `util.cos_sim`,
  and `fcntl`, none of which are imported in `app/routes/api_routes.py`.
- **`/employer/match/<job_id>`** (candidate matching) — references `util.cos_sim`
  with `util` never imported in `app/routes/employer_routes.py`.

### 2.2 Deployment cannot start / background never runs
- **`celery` and `redis` missing from `requirements.txt`** — a clean
  `pip install -r requirements.txt` then boot crashes on `from celery import Celery`.
- **No Celery worker process** — the live `Procfile` defines only `web:`. Even
  with Celery installed, `.delay()` tasks queue but never execute (no worker, no
  eager mode). JIT match calculation silently never happens.

### 2.3 Security hardening removed
- **CSRF protection removed** — `flask-wtf` dropped; `csrf_token` went from 14
  templates to 0. Every POST form/AJAX call is now CSRF-able.
- **Stored-XSS protection removed** — `nh3` dropped; scraped/stored job HTML is
  rendered without sanitising (`clean_html` filter gone).
- **Rate limiting removed** — `flask-limiter` dropped; login / register /
  supersearch are no longer throttled (brute-force / abuse exposure).
- **State-changing actions downgraded GET ← POST** — `delete_job`,
  `toggle_status`, and `notifications/mark_read` lost `methods=['POST']`. In the
  live `employer_dashboard.html`, *delete* is now a plain `<a href>` GET link —
  triggerable by CSRF, link-prefetchers and crawlers.
- **Webhook hardening lost** — the live `/api/webhook/sync-jobs`:
  - uses a hardcoded fallback secret (`default-webhook-secret-123`) instead of
    failing closed when `WEBHOOK_SECRET` is unset;
  - compares tokens with `!=` instead of timing-safe `hmac.compare_digest`;
  - returns the full `traceback` in the 500 JSON response (information disclosure).
- **Session cookie hardening lost** — `SESSION_COOKIE_HTTPONLY / SAMESITE /
  SECURE` are not set in the live factory.
- **Fail-closed secrets weakened** — this version refuses to boot without
  `SECRET_KEY` / `DATABASE_URL`; that guard must be re-verified in the live one.

### 2.4 Features deleted
- **Token economy removed** — `token_balance`, `surveys_filled_count`,
  `TokenTransactions`, `SurveyCompletions` models; the `/tokens/spend`,
  `/profile/tokens`, `/survey/<id>/start`, `/survey/<id>/claim` routes; and
  `tokens.html` were all dropped.
- **Employer approval privacy gate removed** — `Employers.is_approved` and the
  check in candidate matching are gone, so any self-registered employer can view
  candidate matches immediately.

### 2.5 Robustness regressions
- **Atomic cache writes lost** — this version writes `job_embeddings.pt` via a
  temp file + `os.replace()` (atomic). The live version writes in place under an
  `fcntl` lock, so a reader can still see a half-written file / Windows dev can't
  run it.
- **Cross-worker calc state** — this version tracks "is this user calculating?"
  with shared marker files (correct across gunicorn workers). The live version
  uses a per-user `is_processing` DB boolean, which can get stuck `True` if a
  worker dies mid-task.
- **Matching algorithm changed** — this version keeps a hybrid score
  `0.7*semantic + 0.3*skill` and stores Top-3 **global / Hungarian / abroad**
  buckets so the location filter always has data. The live version replaced this
  with `semantic + small skill_boost − location_penalty`, which de-weights skills
  and drops the bucketing.

### 2.6 Project hygiene
- `.gitignore` shrank from a full list to 4 lines — `.env` is no longer ignored
  (secret-leak risk), and `__pycache__`, `*.pyc`, `*.db`, the big CSVs and
  `location_cache.json` became tracked.
- `.env.example` and the planning/GDPR docs were dropped.
- Maintenance scripts were dumped flat in the repo root instead of the organized
  `Deployment/ Migrations/ SampleData/ Tests/ Translating/` folders.

---

## 3. Net assessment

The live version is **more maintainable** (package + Celery) but **less secure
and partially broken**: two features crash, background processing doesn't run as
shipped, and the entire prior security layer plus the token economy and approval
gate were lost.

This refactor branch resolves that by taking the live version's *good parts*
(modular package + Celery) and re-applying them to **this** hardened codebase —
so we get the architecture upgrade **without** giving up security or features.
See the roadmap for the exact restore order and the verification checklist.
