import os

po_path = 'translations/hu/LC_MESSAGES/messages.po'

translations = {
    "Candidate Dashboard": "Jelöltek Irányítópultja",
    "Find My Dream Job": "Keresd Meg Az Álommunkám",
    "Upload new primary CV": "Új elsődleges önéletrajz feltöltése",
    "Log Out": "Kijelentkezés",
    "Upload PDF/TXT": "PDF/TXT feltöltése",
    "Dashboard": "Irányítópult",
    "Profile": "Profil",
    "Candidate Profile": "Jelölt Profilja",
    "Edit Profile": "Profil Szerkesztése",
    "Save Changes": "Változtatások Mentése",
    "Short Description": "Rövid Leírás"
}

with open(po_path, 'r') as f:
    content = f.read()

changed = False
for key, value in translations.items():
    if f'msgid "{key}"' not in content:
        content += f'\nmsgid "{key}"\nmsgstr "{value}"\n'
        changed = True

if changed:
    with open(po_path, 'w') as f:
        f.write(content)
