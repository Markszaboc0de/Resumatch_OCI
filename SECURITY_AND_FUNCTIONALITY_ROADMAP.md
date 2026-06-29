# Roadmap: making the site secure & fully functional again

This is the ordered sequence to restore everything that was lost in the live
`Resumatch_OCI` update, on top of the modular + Celery refactor now living in
this folder. Phases are ordered by **risk and dependency** — do them top to
bottom. Items marked **[DONE in this branch]** are already implemented by the
refactor in this folder; the rest are deployment/operational steps you perform on
the OCI host.

Background: see [`CHANGES_IN_CURRENT_VERSION.md`](CHANGES_IN_CURRENT_VERSION.md)
for what each item is fixing.

---

## Phase 0 — Architecture port (prerequisite) **[DONE in this branch]**
The hardened monolith was split into a package without losing any logic:

- `app/__init__.py` — application factory (`create_app`) + `app = create_app()`.
- `app/extensions.py` — `db, login_manager, babel, csrf, limiter, celery`.
- `app/models.py` — all models, incl. token economy + `is_approved`.
- `app/utils/helpers.py` — text/file utils, quota deduction, nh3 `sanitize_html`,
  cross-worker calc markers.
- `app/services/ai_service.py` — model load, scoring, `refresh_all_matches`,
  Celery tasks.
- `app/routes/{main,user,employer,admin,api}_routes.py` — blueprints.
- Old monolith preserved as `app_legacy.py` for reference.
- All `templates/*.html` `url_for(...)` calls rewritten to blueprint endpoints
  (190 references); URLs are unchanged.
- Backward-compat re-exports in `app/__init__.py` keep `recalculate.py`,
  `import_scraped_jobs.py`, `geocoder.py`, `export_jobs_csv.py`,
  `gunicorn.conf.py`, `Migrations/`, `Tests/` working unchanged.

**Verify:** `python -m py_compile` passes for every module; a stubbed import
smoke test confirms blueprints register, endpoints resolve, CSRF + session
hardening + `clean_html` are active, and the two Celery tasks are registered.

---

## Phase 1 — Fix the broken features (correctness) **[DONE in this branch]**
These were `NameError` crashes in the live version:

1. **Supersearch** — `app/routes/api_routes.py` now imports `util`
   (`from sentence_transformers import util`) and, inside the view,
   `nlp_model, load_cached_embeddings, save_embeddings_cache` from
   `app.services.ai_service`.
2. **Employer candidate matching** — `app/routes/employer_routes.py` now imports
   `util` and pulls `nlp_model` / `extract_match_reasons` from the service.
3. **No `fcntl`** — replaced by the portable atomic `os.replace()` cache writer
   in `ai_service.save_embeddings_cache` (works on the Linux host *and* Windows).

**Acceptance:** POST `/api/supersearch` returns matches; `/employer/match/<id>`
renders `candidate_match.html` for an approved employer.

---

## Phase 2 — Make background processing actually run (Celery) 
Code is **[DONE in this branch]** (`celery_worker.py`, `Procfile` `worker:` line,
`celery` + `redis` in `requirements.txt`, `CELERY_*` in `.env.example`). The
remaining work is on the host:

1. **Install + run Redis** on the OCI box: `sudo apt-get install -y redis-server`
   and enable it (`systemctl enable --now redis-server`).
2. **Set env** in `.env`: `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND`
   (default `redis://localhost:6379/0`).
3. **Run the worker** as its own process:
   `celery -A celery_worker.celery worker --loglevel=info --concurrency=1`
   (the `worker:` Procfile entry; wire it into systemd/your process manager so it
   restarts on boot, same pattern as the web service).
4. **Point rate-limit + Celery at Redis** for multi-worker correctness
   (`RATELIMIT_STORAGE_URI=redis://localhost:6379/1`).

**Acceptance:** upload a CV → a task appears in the worker log → `parsed_tokens`
and `Precalc_Scores` get populated → `/api/match_status` flips to `false` when done.

---

## Phase 3 — Restore the security layer 
Code is **[DONE in this branch]**; this phase is mostly verification + deploy:

1. **Dependencies** — confirm `flask-wtf`, `nh3`, `flask-limiter` are installed
   (already in `requirements.txt`).
2. **CSRF** — `csrf.init_app(app)` is in the factory; every POST form template
   carries `{{ csrf_token() }}` (preserved here). The webhook is the only
   `@csrf.exempt` route (bearer-token auth).
3. **XSS** — the `clean_html` Jinja filter (`nh3.clean`) is registered; templates
   that render stored job HTML pipe through `| clean_html`.
4. **Rate limiting** — `@limiter.limit(...)` on login/register (user, employer,
   admin) and supersearch. Move storage to Redis (Phase 2.4) so limits are shared
   across the 2 gunicorn workers.
5. **State-changing routes are POST-only** — `delete_job`, `toggle_job_status`,
   `mark_notification_read` (verified POST-only by the smoke test).
6. **Webhook** — fails closed without `WEBHOOK_SECRET`, uses
   `hmac.compare_digest`, never returns tracebacks.
7. **Session cookies** — `HTTPONLY`, `SAMESITE=Lax`, `SECURE` (env-gated for
   local HTTP). Keep `SECURE=True` in production behind TLS.
8. **Fail-closed secrets** — factory raises if `SECRET_KEY` / `DATABASE_URL` are
   missing.

**Acceptance:** a POST without a CSRF token is rejected (400); a stored job
description containing `<script>` renders inert; >10 logins/min returns 429; the
webhook returns 503 when the secret is unset and 401 on a bad token.

---

## Phase 4 — Restore deleted features (data + UI) 
Models are **[DONE in this branch]**; the host steps are schema migration + seed:

1. **Run migrations** so the live DB regains the dropped columns/tables:
   `Migrations/add_token_system.py` (token_balance, surveys_filled_count,
   `token_transactions`, `survey_completions`) and the `employers.is_approved`
   column (`Migrations/alter_db.py` / `add_employer_approval.py`).
2. **Token economy** — `/profile/tokens`, `/tokens/spend`, `/survey/<id>/start`,
   `/survey/<id>/claim` and `tokens.html` are restored in `user_routes.py`.
3. **Employer approval gate** — new employers start `is_approved=False`; an admin
   approves them in the dashboard before candidate matching unlocks.

**Acceptance:** filling a survey credits tokens; spending tokens raises a limit;
an unapproved employer is blocked from `/employer/match/<id>`.

---

## Phase 5 — Robustness & data quality 
1. **Atomic cache** — `save_embeddings_cache` writes temp + `os.replace`
   **[DONE]**.
2. **Cross-worker calc state** — filesystem markers in `UPLOAD_FOLDER`
   (`mark/clear/is_user_calculating`) **[DONE]**; ensure web + worker share that
   folder (same host volume).
3. **Recompute embeddings** after deploy so matching has data:
   `python recalculate.py` (re-embeds CVs + jobs, clears the cache, rebuilds
   `Precalc_Scores` via `refresh_all_matches`).
4. **Confirm matching behaviour** — Top-3 global / Hungarian / abroad bucketing is
   restored; spot-check `/dream_job?location_filter=hungary|abroad|all`.

---

## Phase 6 — Project hygiene 
1. **`.gitignore`** — keep the full version here (ignores `.env`, `.env.*`,
   `uploads/`, `*.pt`, `*.db`, `__pycache__/`, `*.pyc`, `.DS_Store`).
2. **`.env.example`** — kept current (now includes `CELERY_*`).
3. **Untrack artifacts** that slipped into the live repo (`__pycache__`, `*.pyc`,
   `*.db`, big CSVs, `location_cache.json` if regenerated).
4. **Keep scripts organized** in `Deployment/ Migrations/ SampleData/ Tests/
   Translating/`.

---

## Phase 7 — Pre-prod verification checklist
Run on a staging copy before pointing DNS:

- [ ] `pip install -r requirements.txt` succeeds on a clean venv.
- [ ] `gunicorn ... app:app` boots; `celery -A celery_worker.celery worker` boots.
- [ ] Register → login → upload CV → worker processes it → `/dream_job` shows matches.
- [ ] Employer register (unapproved) blocked from matching; admin approves; matching works.
- [ ] Supersearch returns results; respects monthly limit + Redis-shared rate limit.
- [ ] CSRF rejection, XSS sanitisation, 429 throttling, webhook 401/503 all behave.
- [ ] Survey claim credits tokens; `/tokens/spend` raises a limit.
- [ ] `delete_job` / `toggle_status` / `mark_read` only work via POST.

---

### One-line summary of the order
**Port architecture → fix crashes → run Celery/Redis → re-arm security →
restore token+approval features → recompute matches → clean the repo → verify.**
