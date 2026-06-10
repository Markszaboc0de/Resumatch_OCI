# Resumatch — Project Memory

> Standalone notes for the Resumatch web platform. Kept separate from the scraper repo (MMm_repo) memory on purpose.

## What this is

`Resumatch_OCI-main/` is the **website half** of the internship platform — the consumer end of the scraper pipeline. It is a **separate codebase** from the scraper repo and is **not under git** (it was pulled from GitHub, so a remote copy exists, but local changes here are not version-controlled — deletions are permanent).

- **Stack:** Flask monolith (`app.py`, ~1900 lines: DB models + routes + matching), served by gunicorn (`Procfile` = `app:app`), PostgreSQL `job_match_db` (the same DB the scrapers' `sync_jobs.py` pushes to on 10.0.0.74).
- **AI models** loaded at startup: sentence-transformers **BAAI/bge-m3** (embeddings for matching) + **GLiNER** (skill NER). Also flask-login, flask-babel (en/hu), TinyMCE editor.
- **Hosts `recalculate.py`** — this is the "downstream recalculate.py (not in scraper repo)" referenced by the scraper CLAUDE.md.
- **The internship/EU/seniority keyword filter lives here**, in `import_scraped_jobs.py` (a SQL regex on title + country). That is *why the scrapers must not filter* — this stage does it.

## How it connects to the scrapers

```
scrapers → scraped_jobs table → import_scraped_jobs.py (upsert) → job_descriptions
        → recalculate.py (embed) → website serves matches
```
Triggered via the `POST /api/webhook/sync-jobs` endpoint in `app.py`.

## Root-pinned files — DO NOT move

Flask/gunicorn resolve these relative to `app.py`, so moving them breaks the site:
`app.py`, `templates/`, `static/`, `translations/`, `location_cache.json`, `requirements.txt`, `Procfile`, `gunicorn.conf.py`.

## Folder structure (reorganized 2026-06-04)

Functional folders created; the live website core stays in root.

| Folder | Contents |
|---|---|
| *(root)* | `app.py`, `templates/`, `static/`, `translations/`, `location_cache.json`, deploy files |
| *(root)* | runtime scripts left here on purpose (webhook/cron call them): `recalculate.py`, `import_scraped_jobs.py`, `geocoder.py`, `export_jobs_csv.py` |
| `Translating/` | `translator.py`, `translator_dash.py`, `update_po.py`, `babel.cfg`, `messages.pot` |
| `Migrations/` | 8 one-off schema scripts (incl. `create_scraped_jobs_table.py`) |
| `Tests/` | `test_*.py` (scratch debug scripts, not a real suite) + `check_db.py` |
| `Deployment/` | `setup_keepalive.sh` |
| `SampleData/` | `UpdatedResumeDataSet.csv`, `template_jobdesc.txt`, `template_resume.pdf` |

### Running moved scripts
Always run **from the project root** (cwd = `Resumatch_OCI-main`):
- Scripts that don't import `app` (the `Translating/` ones): `python Translating/translator.py`
- Scripts that import `app` (`Tests/check_db.py`, `Migrations/alter_db.py`, etc.): `python -m Tests.check_db`, `python -m Migrations.alter_db`

## Cleanup log (2026-06-04)
- Deleted: `__pycache__/`, empty `sys_out.txt`, stale `Elso.csv` (2 MB unreferenced job-board export). Kept `.DS_Store` and `UpdatedResumeDataSet.csv` per request.
- NOT done (deferred): splitting the runtime scripts into `Matching/` + `Pipeline/`. That needs 2 code edits (`import_scraped_jobs.py`'s `from recalculate import`, and `app.py`'s webhook subprocess) plus a VM crontab/systemd check.

## Security audit (2026-06-04)

Full audit of `app.py`, templates, scripts, config. No SQL injection found (ORM + static `text()` are parameterized). Issues split below into what can be fixed in code vs. what needs manual/server action.

### A. Fixes Claude can do in code  (status: ☐ todo / ☑ done)

| # | Severity | Issue | Where | Plan |
|---|---|---|---|---|
| 1 | 🔴 | Stored XSS via `\| safe` on unsanitized HTML | `templates/job_detail.html:135`, `employer_dashboard.html:366`, `map.html:274` | Sanitize `raw_text` with an allowlist (nh3/bleach) on render or store; use `tojson` for map_data | ☑ `clean_html` Jinja filter (nh3) on job_detail + employer_dashboard; map.html uses `tojson` |
| 2 | 🔴 | Hardcoded DB password fallback | `app.py:35`, `Migrations/create_scraped_jobs_table.py:9`, `Migrations/migrate_sync_columns.py:6` | Remove fallback, fail-fast on missing `DATABASE_URL` (rotation itself is server-side, see B) | ☑ app.py + all 3 Migrations scripts now fail-closed — zero hardcoded password left in source; rotate the live pw server-side (B1) |
| 3 | 🔴 | Default admin seeded `admin`/`admin123` | `Migrations/alter_db.py` | Stop seeding a known password; require env-provided strong pw (deleting the existing one is server-side, see B) | ☑ `Migrations/alter_db.py` now seeds admin only from `ADMIN_USERNAME`/`ADMIN_PASSWORD` env, else skips |
| 4 | 🔴 | Weak default `SECRET_KEY` + `WEBHOOK_SECRET` | `app.py:28`, `app.py:1868` | Fail hard if env var missing; no shipped default | ☑ |
| 5 | 🟠 | No CSRF protection | all POST forms + templates | Add Flask-WTF `CSRFProtect` + tokens (needs dep in requirements.txt) | ☑ CSRFProtect on; tokens in all 19 forms; webhook `@csrf.exempt`; supersearch fetch sends `X-CSRFToken` via meta tag |
| 6 | 🟠 | Destructive actions via GET | `app.py:807` toggle_status, `app.py:821` delete_job, mark_read | Convert to POST + CSRF; update templates | ☑ delete_job, toggle_status, mark_read now POST-only + POST forms in templates |
| 7 | 🟠 | Session cookies not hardened | `app.py` config | Set `SESSION_COOKIE_SECURE/SAMESITE/HTTPONLY` | ☑ (SECURE env-gated, default on) |
| 8 | 🟠 | `debug=True` in run entrypoint | `app.py:1885` | Default off, drive via env | ☑ (`FLASK_DEBUG`) |
| 9 | 🟡 | No rate limiting on auth/supersearch | login/register/admin/employer/supersearch | Add flask-limiter (needs dep) | ☑ limiter on login/register/admin-login/employer-login/employer-register (10/min or 10/hr) + supersearch (10/min); no global default so polling endpoints unaffected |
| 10 | 🟡 | User enumeration | `app.py:1181-1187` register; `app.py:1680` /api/suggestions unauth | Generic register messages; auth-gate suggestions | ◑ register msg done; `/api/suggestions` left public on purpose (public listings autocomplete) |
| 11 | 🟡 | Internal errors flashed to user | e.g. `app.py:760` | Log server-side, show generic message | ☑ (create_job, edit_job, upload) |
| 12 | 🟡 | Timing-unsafe webhook token compare | `app.py:1870` | `hmac.compare_digest` + fail-closed if `WEBHOOK_SECRET` unset | ☑ |
| 13 | 🟡 | `.gitignore` missing secrets/artifacts | `.gitignore` | Add `.env`, `uploads/`, `*.pt`, `__pycache__/`, `*.db`, `.DS_Store` | ☑ |
| 14 | 🟢 | `print()` instead of logging; unpinned deps; dead file-reextract path (`recalculate.py` vs `file_path=None`) | various | Optional hardening | ☑ app.py + recalculate.py now use `logging` (logger `resumatch`, `LOG_LEVEL` env); request-handler excepts use `logger.exception`; recalculate dead file path clarified+demoted to debug. Still open: dep pinning; `import_scraped_jobs.py`/`geocoder.py` still use print (CLI scripts) |

**#7-privacy (GDPR) — user chose "reduce PII in match results" (2026-06-04): ☑ DONE.** `/employer/match` (`app.py:843`) now passes only `{candidate_id, score, reasons, has_notified, short_description}` to the template instead of the full `Users`/`CVs` ORM objects — so username, email, and raw CV text no longer reach the rendering layer. Candidates stay anonymized as "Match #N" + skills + score + their own public blurb; notify still works via the opaque `candidate_id`. 
**Employer-approval gate — ☑ DONE (2026-06-04, user approved schema change).** Added `employers.is_approved` (model default False). New self-registered employers are pending; they can log in + post jobs but `/employer/match` (`app.py:913`) is gated until approved (banner + disabled "Find Matches" button on the dashboard). Admins approve via a toggle in the employer-edit modal (`admin_edit_employer`, `app.py:1188`) and see an Approved/Pending badge in the employer table. Migration `Migrations/add_employer_approval.py` adds the column and **grandfathers existing employers to approved** (idempotent — won't re-approve admin-revoked ones).
⚠️ **Deploy ordering:** run `python -m Migrations.add_employer_approval` BEFORE (or together with) deploying this code on an existing DB — the model now selects `is_approved`, so employer pages will error until the column exists. (`db.create_all()` does NOT add columns to existing tables.)

### B. ⭐ SERVER-SIDE DEPLOYMENT CHECKLIST (authoritative — these are the only steps left; Claude cannot do them)

> All code changes are done & static-verified. The app now **fails-closed**: it will not boot, and employer pages will error, until the steps below are completed on the VM. Do them in this order. Run python commands from the project root (`Resumatch_OCI-main/`).

**1. Rotate the leaked DB password** — on the Postgres server, change `app_user`'s password (old value `Mindenszarhoz` is in git history).

**2. Install new dependencies** on the VM:
```
pip install -r requirements.txt      # adds flask-wtf, nh3, flask-limiter
```

**3. Create `.env`** (copy from `.env.example`) with at minimum:
- `SECRET_KEY=` → `python -c "import secrets; print(secrets.token_hex(32))"`
- `DATABASE_URL=postgresql://app_user:<NEW_PASSWORD>@localhost:5432/job_match_db`
- `WEBHOOK_SECRET=` (if the scraper sync webhook is used)
- `ADMIN_USERNAME=` / `ADMIN_PASSWORD=` (for step 5)
- `SESSION_COOKIE_SECURE=True` (set `False` ONLY if testing over plain HTTP)
- optional: `RATELIMIT_STORAGE_URI`, `LOG_LEVEL`

**4. Run DB migrations** (before/with starting the new code — `db.create_all()` does NOT alter existing tables):
```
python -m Migrations.add_employer_approval   # adds employers.is_approved, grandfathers existing employers
python -m Migrations.alter_db                # creates the admin from ADMIN_USERNAME/ADMIN_PASSWORD (+ quota cols)
```

**5. Remove the old default admin** row (`admin` / `admin123`) from the live DB once your real admin (step 4) works.

**6. Restart the app** (gunicorn) so it picks up `.env` + new code. Changing `SECRET_KEY` logs out all existing sessions (expected).

**7. Apply the keepalive memory change** — re-run `Deployment/setup_keepalive.sh` (or `sudo systemctl restart oci-keepalive`) to pick up `--vm-bytes 15%`.

**8. Enforce HTTPS/TLS** at the reverse proxy (nginx/Caddy) so `SESSION_COOKIE_SECURE` cookies are sent.

**9. Scrub git history / rotate all leaked secrets**; make the GitHub repo private if it isn't (the old DB password is in history).

**10. (Optional) Redis** for `RATELIMIT_STORAGE_URI` so rate limits are accurate across both gunicorn workers.

**11. (Policy, not code) GDPR**: consent flow + employer-vetting process now that the approval gate exists.

### Post-deploy smoke test (manual, ~5 min)
Log in → upload a CV → see matches populate. Post/edit/delete a job. Open a job detail page (XSS-sanitized HTML renders). Run a supersearch (CSRF header works). Register a new employer → see "pending approval" banner + disabled "Find Matches" → approve in admin panel → matching unlocks. If using the webhook, fire it with the bearer token.

### Working order (code fixes)
Batch 1 (non-invasive, no template changes): #13 .gitignore, #2 remove pw fallbacks, #4 fail-closed secrets, #7 cookie flags, #8 debug off, #12 hmac compare, #11 generic errors. — ✅ DONE
Batch 2 (templates/deps): #1 XSS sanitize, #5 CSRF, #6 GET→POST. — ✅ DONE
Batch 3: #9 rate limit (flask-limiter dep), #3 admin seeding hardening, #14 logging cleanup, #7 reduce-PII. — ✅ DONE.
Batch 4: #7 employer-approval gate (schema change, user-approved) — ✅ DONE. Only dependency pinning (#14, needs VM `pip freeze`) remains.

### ⚠️ New runtime dependencies (added to requirements.txt 2026-06-04)
`flask-wtf`, `nh3`, and `flask-limiter` were added for CSRF + HTML sanitization + rate limiting. **The app will not import until these are `pip install`ed on the VM.** Also created `.env.example` documenting all env vars; the app now refuses to boot without `SECRET_KEY` + `DATABASE_URL` in `.env` (see section B).

## Operational / architecture audit (2026-06-04)

A second (non-security) audit was reviewed and fact-checked against the code. Verdicts:
- **Monolithic app.py** — accurate (1957 lines; models+helpers+AI+all routes). Roadmap: blueprint/factory split.
- **Hardcoded creds / weak SECRET_KEY** — already fixed in the security pass (#2/#4).
- **threading.Thread + global `active_calculations` set** — accurate; per-worker set doesn't sync across the 2 gunicorn workers, so `/api/match_status` is unreliable.
- **Models loaded per-worker → OOM** — partly outdated: Procfile already uses `--preload` + 2 workers (not 4), so models load once in master & share via CoW. RAM concern still real (small VM, CoW degrades).
- **Silent error swallowing / print** — print→logging was done in #14; silent `except: pass` swallows remained.
- **`job_embeddings.pt` concurrent-write corruption** — accurate (3 `torch.save` sites + reads/deletes, non-atomic).
- Audit *missed*: keepalive `stress-ng --vm-bytes 40%` deliberately holds 40% RAM (amplifies OOM); cache-deletion race in recalculate/import; no tests; no migration framework (Alembic).

### Quick wins — ✅ DONE (2026-06-04)
1. **Atomic embeddings cache write**: new `save_embeddings_cache()` helper (`app.py` ~369) writes a temp file then `os.replace()` (atomic on POSIX); all 3 `torch.save` sites now call it. Kills the corruption race.
2. **Killed silent swallows**: the 7 bare `except: pass` now log — `logger.debug(..., exc_info=True)` for expected skip-bad-vector cases, `logger.warning(..., exc_info=True)` for real failures (location_cache parse, supersearch cache save). (Remaining `pass` are intentional `except ValueError` input validation + one `except OSError` temp cleanup.)
3. **Keepalive memory** lowered `stress-ng --vm-bytes 40% → 15%` (`Deployment/setup_keepalive.sh`) to free RAM headroom for the ML workers.

### Cross-worker calc tracking — ✅ DONE (2026-06-04, no server-side change)
Replaced the per-worker global `active_calculations` set with **shared-filesystem marker files** (`.calc_<user_id>.lock` in UPLOAD_FOLDER) via helpers `mark_calculating` / `clear_calculating` / `is_user_calculating` (`app.py` ~126). Correct across the 2 gunicorn workers (FS is shared) and `/api/match_status` now reliable. 600s TTL auto-clears stale markers from crashed workers. Chosen over the DB-column approach specifically because that would have needed an ALTER TABLE migration (server-side); this needs only a normal code deploy.

### Operational roadmap (NOT done — needs decisions/infra)
- **Large (infra):** Celery/RQ + Redis for ML inference & matching — fixes the thread model AND model-per-worker OOM (web workers drop the models); also gives a real embeddings cache store. Needs Redis + a worker process on the VM.
- **Large (refactor):** blueprint/factory split of app.py — do incrementally (models → services → route blueprints) and add a smoke test first (there are currently no tests).
- Consider Alembic for migrations (currently hand-rolled SQL scripts in `Migrations/`).
