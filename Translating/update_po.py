import os
import re

po_path = 'translations/hu/LC_MESSAGES/messages.po'

translations = {
    "Search Role or Description": "Keresés (szerepkör vagy leírás)",
    "Country": "Ország",
    "Company": "Cég",
    "Filter": "Szűrés",
    "View Job": "Állás megtekintése",
    "Visit": "Látogatás",
    "No Link": "Nincs link",
    "Job Listings": "Állásajánlatok",
    "No jobs found": "Nem található állás",
    "Try adjusting your search filters to find more results.": "Próbálja módosítani a keresési szűrőket több eredményért.",
    "Previous": "Előző",
    "Next": "Következő",
    "Welcome back!": "Üdvözöljük újra!",
    "Email": "Email",
    "Password": "Jelszó",
    "Don't have an account?": "Még nincs fiókja?",
    "Create your account": "Hozza létre fiókját",
    "Email address": "Email cím",
    "Username": "Felhasználónév",
    "Confirm Password": "Jelszó megerősítése",
    "Already have an account?": "Már van fiókja?"
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
    print("translations updated.")
else:
    print("no new translations needed.")
