# GDPR – Technikai megvalósítási terv (Resumatch)

> Cél: az **Adatvédelmi Tájékoztató v1** minden állítását igazzá tenni a kódban.
> Stack: Flask + SQLAlchemy + PostgreSQL, `sentence-transformers` (helyi vektorizálás), flask-login, flask-wtf (CSRF), nh3 (HTML-sanitizálás), flask-limiter.
> Jelölések: 🔴 kötelező (jogi kockázat) · 🟡 fontos · 🟢 finomítás · ⏱️ becsült méret.

---

## 0. Két eltérés, amit először tisztázni kell

**(A) A szabályzat megerősítő e-mailt ígér (3.1), de nincs e-mail küldés a kódban.**
A `requirements.txt`-ben nincs SMTP/Flask-Mail, a `users` táblában nincs `email_verified`. Két lehetőség:
- **Megvalósítjuk** (ajánlott) – lásd 1. feladat. Ez egyben feltétele a hírlevélnek, inaktivitási figyelmeztetésnek és a jogérvényesítési visszaigazolásnak.
- **Töröljük az ígéretet** a 3.1-ből (gyengébb megoldás).

**(B) A 3.5 (céges megkeresés) a kódban biztonságosabban működik, mint ahogy a szabályzat leírja.**
A `employer_match_candidate` route a munkáltatónak **csak anonim adatot** ad át (pontszám, skill-indokok, `short_description`, opák `candidate_id`) – nevet, e-mailt, CV-szöveget **nem**. A `notify_candidate` route nem a jelölt adatát adja a cégnek, hanem a **cég elérhetőségét küldi el a jelöltnek**, aki maga dönt a kapcsolatfelvételről. Tehát **mi sosem továbbítunk jelölti PII-t a munkáltatónak.**
➡️ **Teendő:** a szabályzat 3.5 pontját igazítsuk ehhez a valós, kedvezőbb folyamathoz (lásd 9. feladat). Kódváltás nem kell, csak ellenőrzés, hogy a `candidate_match.html` tényleg nem szivárogtat PII-t.

---

## Fázis 1 – Alapok (jogi minimum) 🔴

### 1. E-mail infrastruktúra + e-mail megerősítés 🔴 ⏱️ M
**Hivatkozás:** Tájékoztató 3.1.
**Lépések:**
1. Adj a stackhez transzakciós e-mail-küldést: `Flask-Mail` SMTP-vel, vagy egy EGT-s/DPF-es szolgáltató (a szabályzat 5.1 `[E-mail szolgáltató]` helykitöltőjét töltsd ki a választott szolgáltatóval). Konfiguráció `.env`-ben (`MAIL_SERVER`, `MAIL_USERNAME`, stb.).
2. `users` séma bővítése (migráció a `Migrations/` mintájára):
   - `email_verified BOOLEAN DEFAULT FALSE`
   - `email_verify_token VARCHAR(255)` (vagy külön `email_tokens` tábla lejárati idővel)
   - `created_at TIMESTAMP DEFAULT now()`
3. `register()` route (app.py ~1306): regisztráció után **ne** logináljon be automatikusan; generálj tokent (`secrets.token_urlsafe`), küldj megerősítő linket (`/verify-email/<token>`).
4. Új route: `/verify-email/<token>` → beállítja `email_verified=True`, törli a tokent.
5. `login()` (app.py ~1343): ha nincs megerősítve, jelezd és kínálj újraküldést (vagy korlátozd a funkciókat megerősítésig).
**Kész-kritérium:** új regisztrációnál érkezik megerősítő e-mail; megerősítés nélkül a fiók korlátozott.

### 2. A Tájékoztató publikálása + kötelező elfogadás 🔴 ⏱️ S
**Hivatkozás:** GDPR 13. cikk (átláthatóság), 7. cikk (hozzájárulás bizonyíthatósága).
**Lépések:**
1. Publikáld a szabályzatot: `/privacy` route + `templates/privacy.html` (a v1 docx tartalmából HTML), és `/cookie-policy` (lásd 6. feladat).
2. Tedd be a **láblécbe** minden oldalon (a közös base templ.-ben), és a `register.html` + `employer_register.html` űrlapokra.
3. Regisztrációs űrlap: **kötelező checkbox** „Elolvastam és elfogadom az Adatvédelmi Tájékoztatót" (link nélkül nem mehet tovább). Tárold: `users.privacy_accepted_at TIMESTAMP` + a `version` (pl. `'1.0'`).
**Kész-kritérium:** elfogadás nélkül nincs regisztráció; az elfogadás ténye + verzió + időbélyeg eltárolva.

### 3. Hozzájárulás-kezelés + napló (accountability) 🔴 ⏱️ M
**Hivatkozás:** 3.2, 3.6, GDPR 7. cikk (5) – elszámoltathatóság.
**Lépések:**
1. `users` bővítés: `marketing_consent BOOLEAN DEFAULT FALSE`, `marketing_consent_at TIMESTAMP NULL`.
2. Új tábla `consent_log` (audit): `id, user_id, consent_type, granted BOOL, version, source, created_at`. Minden hozzájárulás-változás ide is kerüljön (regisztráció, marketing be/ki, CV-feltöltési különleges-adat tudomásul vétel).
3. Külön, **alapból kikapcsolt** marketing checkbox a regisztrációnál és a profilban (nem összevonva a kötelező elfogadással – a GDPR tiltja a csomagolt hozzájárulást).
**Kész-kritérium:** minden hozzájárulás visszakereshető (ki, mit, mikor, melyik verzióra).

### 4. Önkiszolgáló adatexport (hordozhatóság + hozzáférés) 🔴 ⏱️ S
**Hivatkozás:** 6.3 (Hozzáférés, Adathordozhatóság), GDPR 15. és 20. cikk.
**Lépések:**
1. Új route `/profile/export` (login szükséges) → összegyűjti a user adatait: `Users` profil, `CVs` (raw_text, extracted_skills, upload_date), `Surveys`, `SurveyCompletions`, `TokenTransactions`, `Notifications`.
2. Add vissza **gép által olvasható** formátumban: `Response(json.dumps(...), mimetype='application/json', headers={'Content-Disposition': 'attachment; filename=resumatch_export.json'})`.
3. Profil oldalra gomb: „Adataim letöltése".
**Kész-kritérium:** a user egy kattintással letölti az összes saját adatát JSON-ban.

### 5. Önkiszolgáló fióktörlés (elfeledtetés) 🔴 ⏱️ M
**Hivatkozás:** 6.3 (Törlés), GDPR 17. cikk.
**Jelenlegi állapot:** van `delete_cv` (app.py ~1634), de **teljes fióktörlés nincs**.
**Lépések:**
1. Új route `/profile/delete_account` (POST, megerősítő modal + jelszó/újrabejelentkezés).
2. **Hard delete** kaszkád (egy tranzakcióban): `PrecalcScores` (cv-khez), a CV-fájlok a lemezről (`UPLOAD_FOLDER`, `CVs.file_path`), `CVs`, `Surveys`, `SurveyCompletions`, `Notifications`, `TokenTransactions`, végül `Users`.
3. **Kivétel:** a már kiállított számviteli bizonylatok jogszabály miatt maradnak (8 év, 3.9/9. sor) – ezeket ne a user-rekordból, hanem a könyvelésből kezeld.
4. Ellenőrizd, hogy az embedding-cache (`job_embeddings.pt` / parsed_tokens) nem tartalmaz-e árva jelölti vektort.
**Kész-kritérium:** törlés után a user egyetlen táblában és a lemezen sem szerepel (a számviteli kivételtől eltekintve).

---

## Fázis 2 – Cookie, marketing, automatizmusok 🟡

### 6. Sütibanner + sütitájékoztató (GA előtt!) 🔴 ⏱️ M
**Hivatkozás:** 6.5, 3.8, 5.3. Eker. tv. / ePrivacy.
**Fontos:** a Google Analytics csak ezután kapcsolható be, különben jogsértő.
**Lépések:**
1. `/cookie-policy` oldal: sütik felsorolása (jelenleg: csak a session-süti – feltétlenül szükséges). Frissítsd, amikor jön a GA.
2. Sütibanner (base templ.): „Csak szükséges" / „Elfogadom" választás; a döntést tárold (pl. `cookie_consent` süti vagy localStorage).
3. A GA `<script>` **csak** „Elfogadom" után töltődjön be (feltételes betöltés). Visszavonási lehetőség (beállítások link).
**Kész-kritérium:** alapból nem fut nem-szükséges süti; GA csak hozzájárulás után aktív.

### 7. Marketing hírlevél + leiratkozás 🟡 ⏱️ M
**Hivatkozás:** 3.6, 6.3 (tiltakozás közvetlen üzletszerzés ellen), Grt.
**Függ:** 1. (e-mail) és 3. (marketing_consent).
**Lépések:**
1. Hírlevél küldése csak `marketing_consent=True` userekre.
2. Minden marketing e-mail alján **egykattintásos leiratkozó link**: `/unsubscribe/<token>` → `marketing_consent=False`, `consent_log` bejegyzés. Token legyen user-specifikus, login nélkül is működjön.
3. A profilban is kapcsolható.
**Kész-kritérium:** leiratkozás után nem megy több marketing e-mail; a tiltakozás naplózva.

### 8. 24 hónapos inaktivitási törlés 🟡 ⏱️ M
**Hivatkozás:** Tájékoztató 2. sor (megőrzés).
**Jelenlegi állapot:** `users.last_active_date` már létezik ✅.
**Lépések:**
1. Ütemezett feladat (cron a szerveren, vagy `APScheduler`, vagy külön szkript a `Migrations`/util mintájára, amit a deploy ütemez):
   - `last_active_date < now() - 23 hónap` → **figyelmeztető e-mail** (a törlés előtt, „30 nap múlva töröljük").
   - `last_active_date < now() - 24 hónap` → automatikus fióktörlés (az 5. feladat törlési logikáját hívja).
2. Logold a futást (audit).
**Kész-kritérium:** inaktív fiókok figyelmeztetés után 24 hónapnál törlődnek.

### 9. CV-feltöltési különleges-adat tájékoztatás + a 3.5 szöveg igazítása 🟡 ⏱️ S
**Hivatkozás:** 3.2 (különleges adat), 3.5 (céges megkeresés).
**Lépések:**
1. A `profile.html` CV-feltöltő gombja mellé **rövid figyelmeztetés**: „Kérjük, ne tölts fel különleges adatot (egészség, vallás, stb.). Ha mégis, a feltöltéssel hozzájárulsz a kezeléséhez." + (opcionális) tudomásulvételi checkbox → `consent_log`.
2. **Pontosítás a kódról:** a `CVs.raw_text` a teljes CV-szöveget tárolja (a kulcsszavas szűréshez kell), tehát az „álnevesítés = eredeti törlése" megfogalmazás a szabályzatban pontatlan. Két út:
   - (a) **Ajánlott:** a szabályzat „Álnevesítés" definícióját igazítsd a valósághoz (a vektor álnevesített, de a raw_text megmarad a szolgáltatáshoz) — apró szövegmódosítás.
   - (b) Ha tényleg törölni akarjátok: a feldolgozás után töröld a feltöltött **fájlt** a lemezről (`CVs.file_path`), és csak a `raw_text`-et + vektort tartsd. (A nyers fájl törlése amúgy is jó adatminimalizálás.)
3. **3.5 igazítás:** a szabályzat 3.5 szövegét írd át a valós folyamatra (a munkáltató anonim találatot lát; a saját elérhetőségét küldi; a jelölt dönt; mi nem továbbítunk jelölti PII-t). Ellenőrizd a `candidate_match.html`-t, hogy tényleg csak a `match_list` minimumát rendereli.
**Kész-kritérium:** a feltöltőnél megjelenik a figyelmeztetés; a szabályzat szövege egyezik a kód viselkedésével.

---

## Fázis 3 – Folyamatok és dokumentáció 🟢

### 10. Adatvédelmi incidens napló + 72 órás folyamat 🟡 ⏱️ S
**Hivatkozás:** 6.2, GDPR 33–34. cikk.
**Lépések:**
1. Egyszerű belső nyilvántartás: vagy admin-only `incidents` tábla (`id, detected_at, description, affected_scope, risk_level, actions, reported_to_naih_at`), vagy egy belső táblázat/dokumentum.
2. Írott folyamat: ki észleli → ki értékeli → 72 órán belül NAIH-bejelentés, ha kockázatos → magas kockázat esetén érintettek értesítése.
**Kész-kritérium:** létezik incidensnyilvántartás és egy féloldalas belső eljárásrend.

### 11. Érdekmérlegelési teszt (LIA) – 4.2 🟡 ⏱️ S
**Hivatkozás:** 4.2 (a szöveg hivatkozik rá, ezért léteznie kell).
**Lépések:** féloldalas belső dokumentum az üzletfejlesztési kapcsolatgyűjtéshez: cél, szükségesség, érintetti érdekek, egyensúly, garanciák (tiltakozási jog, forrás). Iktasd. (Külön doksiban elkészíthetem.)
**Kész-kritérium:** iktatott LIA a 4.2-höz.

### 12. Jogérvényesítési csatorna + belső SLA 🟢 ⏱️ S
**Hivatkozás:** 6.3.
**Lépések:** a `[adatvédelmi e-mail]` legyen valós és figyelt; a beérkező kérelmekre **1 hónapos** belső határidő; a legtöbb kérés önkiszolgáló (export/törlés/módosítás), így csak a maradékot kell kézzel kezelni.
**Kész-kritérium:** működő e-mail-cím + dokumentált kezelési folyamat.

### 13. Biztonsági higiénia – ellenőrzés (6.1) 🟢 ⏱️ S
**Jelenlegi állapot (jó):** jelszó-hash (werkzeug), CSRF (flask-wtf), nh3-sanitizálás, rate-limit (flask-limiter), `SESSION_COOKIE_SECURE=True`, TLS a proxynál.
**Teendő (ellenőrzés):** rendszeres mentés megléte; hozzáférési naplózás; `FLASK_DEBUG=False` prod; a `WEBHOOK_SECRET` rotálva; a feltöltési mappa nem publikusan elérhető.
**Kész-kritérium:** a fenti pontok bizonyítottan rendben.

---

## Megvalósítási sorrend (összefoglaló)

| # | Feladat | Prioritás | Függ | Méret |
|---|---|---|---|---|
| 1 | E-mail infrastruktúra + megerősítés | 🔴 | – | M |
| 2 | Tájékoztató publikálása + elfogadás | 🔴 | – | S |
| 3 | Hozzájárulás-kezelés + napló | 🔴 | 2 | M |
| 4 | Önkiszolgáló adatexport | 🔴 | – | S |
| 5 | Önkiszolgáló fióktörlés | 🔴 | – | M |
| 6 | Sütibanner + sütitájékoztató | 🔴 | – | M |
| 7 | Hírlevél + leiratkozás | 🟡 | 1, 3 | M |
| 8 | 24 hónapos inaktivitási törlés | 🟡 | 1, 5 | M |
| 9 | CV különleges-adat + 3.5 igazítás | 🟡 | 3 | S |
| 10 | Incidens napló + 72h folyamat | 🟡 | – | S |
| 11 | LIA (4.2) | 🟡 | – | S |
| 12 | Jogérvényesítési csatorna + SLA | 🟢 | 1 | S |
| 13 | Biztonsági higiénia ellenőrzés | 🟢 | – | S |

**Javasolt első sprint (jogi minimum élesítéshez):** 1, 2, 3, 4, 5, 6.
A 6. (sütibanner) **kötelező feltétele** a Google Analytics bevezetésének.
