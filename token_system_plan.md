# Token System Implementation Plan

## Overview

Students earn tokens by filling surveys. Tokens can be spent on platform rewards. The system hooks into the existing `match_limit`/`supersearch_limit` mechanics and the existing `Surveys` model.

**Key constraint:** Surveys are external URLs (Google Forms, Typeform, etc.), so completion cannot be verified programmatically. The plan uses a time-gated self-claim flow with anti-abuse guards.

---

## 1. Database Changes

### 1a. Add to `Users` model

```python
token_balance = db.Column(db.Integer, default=0, nullable=False)
surveys_filled_count = db.Column(db.Integer, default=0, nullable=False)
```

`surveys_filled_count` is a denormalized counter used for survey ranking — faster than a COUNT query on every page load.

### 1b. New table: `SurveyCompletions`

```python
class SurveyCompletions(db.Model):
    __tablename__ = 'survey_completions'
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    survey_id   = db.Column(db.Integer, db.ForeignKey('surveys.survey_id'), nullable=False)
    started_at  = db.Column(db.DateTime, nullable=False)
    claimed_at  = db.Column(db.DateTime, nullable=True)   # None = started but not claimed
    tokens_awarded = db.Column(db.Integer, nullable=True)

    __table_args__ = (db.UniqueConstraint('user_id', 'survey_id'),)
```

### 1c. New table: `TokenTransactions`

Audit log — never deleted, only appended.

```python
class TokenTransactions(db.Model):
    __tablename__ = 'token_transactions'
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    amount      = db.Column(db.Integer, nullable=False)   # positive = earn, negative = spend
    reason      = db.Column(db.String(255), nullable=False)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)
```

### 1d. Migration script

Add `Migrations/add_token_system.py` — same pattern as existing migration scripts.

---

## 2. Token Economy

### Earning tokens

| Action | Tokens earned |
|---|---|
| Fill a survey | `floor(estimated_minutes / 5)`, minimum 1 |

Examples: 5-min survey → 1 token, 15-min → 3 tokens, 30-min → 6 tokens.

This scales reward with effort and incentivises filling longer/harder surveys.

### Spending tokens

| Reward | Cost |
|---|---|
| 1 extra job match | 2 tokens |
| 1 extra supersearch | 3 tokens |
| Publish/update a survey | 1 token |

No monthly cap on token spending. Survey ranking is also determined by `surveys_filled_count` of the creator — the more surveys you fill, the higher your survey appears on the board.

### Token balance rules
- Balance can never go below 0 (enforced in spend route).
- No expiry in v1.

---

## 3. Anti-Abuse

### 3a. Minimum time gate
When user clicks a survey link, record `started_at` in `SurveyCompletions`. The claim endpoint rejects the request if:

```
(now - started_at).total_seconds() < survey.estimated_minutes * 60 * 0.5
```

(Must spend at least 50% of estimated time before claiming.)

### 3b. One completion per user per survey
Enforced by the `UniqueConstraint('user_id', 'survey_id')` on `SurveyCompletions`. The claim route returns a 400 if the row already has a `claimed_at`.

### 3c. No self-completion
Check `survey.user_id != current_user.user_id` before awarding tokens.

### 3d. Future (v2)
- Attention check question injection (requires surveys to be hosted on-platform)
- Flagging system for survey creators to dispute completions

---

## 4. New Routes

### `GET /survey/<int:survey_id>/start`
- Creates or updates a `SurveyCompletions` row with `started_at = now` (only if not already claimed).
- Redirects to `survey.survey_url`.

### `POST /survey/<int:survey_id>/claim`
- Validates: logged in, not own survey, time gate passed, not already claimed.
- Awards tokens: updates `token_balance`, increments `surveys_filled_count`, writes `TokenTransactions` row, sets `claimed_at`.
- Returns JSON `{tokens_awarded: N, new_balance: M}` (called via fetch from the survey board).

### `POST /tokens/spend`
- Body: `{"reward": "match"}` or `{"reward": "supersearch"}`
- Costs: match = 2 tokens, supersearch = 3 tokens. No cap on how many times per month.
- Validates balance, deducts tokens, increments `match_limit` or `supersearch_limit` by 1.
- Returns JSON.

### `GET /profile/tokens`
- Renders token balance + last 20 `TokenTransactions` for the current user.

---

## 5. Survey Board Changes (`/survey` route + template)

**Current:** Surveys ordered by `created_at desc`.

**New:** Ordered by creator's `surveys_filled_count desc`, then `created_at desc`.

```python
surveys = (
    db.session.query(Surveys)
    .join(Users, Surveys.user_id == Users.user_id)
    .order_by(Users.surveys_filled_count.desc(), Surveys.created_at.desc())
    .all()
)
```

**Per-survey card UI changes:**
- "Go to survey" button → triggers `GET /survey/<id>/start` then opens URL in new tab
- After clicking, button becomes "Mark as completed" (enabled after timer)
- Token badge showing how many tokens the survey is worth
- "Already filled" badge if the user has a `claimed_at` record for this survey

**Timer UX:** JavaScript countdown (`estimated_minutes * 0.5` minutes) that enables the "Mark as completed" button client-side. Server still validates server-side — this is just UX.

---

## 6. Profile Page Changes

- Add token balance display with short explanation
- "Spend tokens" section with the two rewards (matches, supersearches), showing current balance vs. cost
- Link to `/profile/tokens` for full history
- Survey upload section: show the user's current rank position on the survey board (based on `surveys_filled_count`)

---

## 7. Implementation Order

1. **DB + migration** — Add columns/tables, run migration, verify with existing data
2. **Backend: earn flow** — `/survey/<id>/start` and `/survey/<id>/claim` routes + helpers
3. **Backend: spend flow** — `/tokens/spend` route
4. **Survey board UI** — Reorder query + new button states + token badge
5. **Profile UI** — Token balance widget + spend section
6. **Token history page** — `/profile/tokens`
7. **Manual QA** — Test full earn/spend loop, time gate, self-completion block, double-claim block

---

## 8. Files to Change

| File | Change |
|---|---|
| `app.py` | New models, new routes, modified survey board query |
| `Migrations/add_token_system.py` | New migration script |
| `templates/survey.html` | New button states, token badge, JS timer |
| `templates/profile.html` | Token balance widget, spend section |
| `templates/tokens.html` | New page — token transaction history |

---

## Confirmed Token Economy

- 1 job match = **2 tokens**
- 1 supersearch = **3 tokens**
- Publish/update a survey = **1 token** (deducted on each save/update)
- Survey ranking = ordered by creator's `surveys_filled_count` descending
- No monthly cap on spending
