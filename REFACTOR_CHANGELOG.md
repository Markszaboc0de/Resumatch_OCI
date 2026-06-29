# Refactor changelog — modular package + Celery (security preserved)

Date: 2026-06-29. Scope: ported the live `Resumatch_OCI` version's good parts
(modular package + Celery) onto this hardened codebase, without losing any
security or features. See [`CHANGES_IN_CURRENT_VERSION.md`](CHANGES_IN_CURRENT_VERSION.md)
for the audit that motivated this and
[`SECURITY_AND_FUNCTIONALITY_ROADMAP.md`](SECURITY_AND_FUNCTIONALITY_ROADMAP.md)
for the deploy-side restore order.

---

## 1. New `app/` package (was: monolithic `app.py`)

| File | Contents |
|---|---|
| `app/__init__.py` | `create_app()` factory + `app = create_app()`; binds db/login/babel/csrf/limiter/celery; registers blueprints; `clean_html` filter; `inject_notifications`; backward-compat re-exports |
| `app/extensions.py` | `db, login_manager, babel, csrf, limiter, celery` (no app bound) |
| `app/models.py` | All models incl. token economy (`TokenTransactions`, `SurveyCompletions`, `token_balance`, `surveys_filled_count`) and `Employers.is_approved` |
| `app/utils/helpers.py` | text/file utils, quota deduction, nh3 `sanitize_html`, cross-worker calc markers (`mark/clear/is_user_calculating`) |
| `app/services/ai_service.py` | model load, hybrid scoring + Top-3 global/HU/abroad buckets, `refresh_all_matches`, atomic cache writer, Celery tasks |
| `app/routes/main_routes.py` | landing, set_language, job_detail, listings, map, `/employer` stub |
| `app/routes/user_routes.py` | auth, dashboard, profile/upload, dream_job, matches, surveys, tokens |
| `app/routes/employer_routes.py` | session auth, job CRUD, candidate matching, notify (url_prefix `/employer`) |
| `app/routes/admin_routes.py` | session auth, dashboard, edit user/employer (url_prefix `/admin`) |
| `app/routes/api_routes.py` | notifications, suggestions, supersearch, match_status, webhook |

- Original monolith preserved as `app_legacy.py` (renamed via `git mv`).

## 2. Celery (was: `threading.Thread`)

- `celery_worker.py` — worker entrypoint (`celery -A celery_worker.celery worker`).
- `Procfile` — added `worker:` process alongside `web:`.
- `requirements.txt` — added `celery`, `redis`.
- `.env.example` — added `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`.
- Background work runs as tasks `run_jit_task` / `run_upload_jit_full_task`;
  trigger points are `ensure_jit_matches()` and the CV upload route.

## 3. Bugs fixed (these crashed in the live version)

- **supersearch** (`api_routes.py`): now imports `util`, and
  `nlp_model / load_cached_embeddings / save_embeddings_cache` inside the view.
- **employer candidate matching** (`employer_routes.py`): now imports `util` and
  the service's `nlp_model` / `extract_match_reasons`.
- **`fcntl` removed**: portable atomic `os.replace()` cache writer instead
  (works on Linux host and Windows).

## 4. Security preserved (not dropped, unlike the live version)

- CSRF (`csrf.init_app`), webhook is the only `@csrf.exempt` route.
- nh3 `clean_html` Jinja filter for stored-HTML XSS.
- `flask-limiter` decorators on login/register (user/employer/admin) + supersearch.
- `delete_job`, `toggle_job_status`, `mark_notification_read` are **POST-only**.
- Hardened session cookies; fail-closed `SECRET_KEY` / `DATABASE_URL`.
- Webhook: fails closed without `WEBHOOK_SECRET`, `hmac.compare_digest`, no traceback leak.

## 5. Templates & scripts

- Rewrote **190** `url_for(...)` calls across 21 templates to blueprint endpoints
  (`login` → `user_routes.login`, etc.). Public URLs unchanged.
- Backward-compat re-exports in `app/__init__.py` keep `recalculate.py`,
  `import_scraped_jobs.py`, `geocoder.py`, `export_jobs_csv.py`,
  `gunicorn.conf.py`, `Migrations/`, `Tests/` working unchanged.

## 6. Verification

- `python -m py_compile` passes for every module + maintenance script.
- Stubbed-dependency smoke test (throwaway venv) — **all passed**:
  - package imports / `create_app()` succeeds (no circular imports, no NameErrors);
  - 5 blueprints register; all critical endpoints present;
  - `delete_job` / `toggle_status` / `mark_read` are POST-only;
  - URL paths preserved (`/employer/login`, `/admin/login`, `/api/supersearch`);
  - CSRF + `SESSION_COOKIE_HTTPONLY` + `clean_html` active;
  - Celery tasks registered; `url_for` resolves blueprint endpoints.

## 7. Still required on the OCI host (not doable from the dev box)

1. `pip install -r requirements.txt` on a clean venv.
2. Install + run Redis; run the Celery `worker` process (systemd).
3. Set `.env`: `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND` (and ideally
   `RATELIMIT_STORAGE_URI=redis://...`).
4. Run migrations (token system, `is_approved`) then `python recalculate.py`.

## 8. Known nit

- `.gitignore` pattern `.env.*` also ignores `.env.example`; narrow it to `.env`
  if you want the example file tracked.

## Files changed this session

- **Added:** `app/` package (12 modules), `celery_worker.py`,
  `CHANGES_IN_CURRENT_VERSION.md`, `SECURITY_AND_FUNCTIONALITY_ROADMAP.md`,
  `REFACTOR_CHANGELOG.md`.
- **Renamed:** `app.py` → `app_legacy.py`.
- **Modified:** `Procfile`, `requirements.txt`, `.env.example`, 21 `templates/*.html`.
