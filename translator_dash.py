import re

files = [
    'templates/dashboard.html',
    'templates/profile.html'
]

replacements = {
    r'>Candidate Dashboard<': r'>{{ _(\'Candidate Dashboard\') }}<',
    r'>Find My Dream Job<': r'>{{ _(\'Find My Dream Job\') }}<',
    r'>Upload new primary CV<': r'>{{ _(\'Upload new primary CV\') }}<',
    r'>Log Out<': r'>{{ _(\'Log Out\') }}<',
    r'>Upload PDF/TXT<': r'>{{ _(\'Upload PDF/TXT\') }}<',
    r'>Dashboard<': r'>{{ _(\'Dashboard\') }}<',
    r'>Logout<': r'>{{ _(\'Log Out\') }}<',
    r'>Profile<': r'>{{ _(\'Profile\') }}<',
    r'>Candidate Profile<': r'>{{ _(\'Candidate Profile\') }}<',
    r'>Edit Profile<': r'>{{ _(\'Edit Profile\') }}<',
    r'>Save Changes<': r'>{{ _(\'Save Changes\') }}<',
    r'>Short Description<': r'>{{ _(\'Short Description\') }}<'
}

for f in files:
    try:
        with open(f, 'r') as file:
            content = file.read()
            
        new_content = content
        for pattern, repl in replacements.items():
            new_content = re.sub(pattern, repl, new_content)
            
        with open(f, 'w') as file:
            file.write(new_content)
    except Exception as e:
        pass
