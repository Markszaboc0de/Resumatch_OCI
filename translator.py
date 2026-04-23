import glob
import re

files = [
    'templates/listings.html',
    'templates/login.html',
    'templates/register.html',
    'templates/dashboard.html',
    'templates/profile.html',
    'templates/employer_login.html',
    'templates/map.html'
]

# Add a dictionary of exact block strings to be replaced across those files
replacements = {
    r'Search Role or Description': r"{{ _('Search Role or Description') }}",
    r'placeholder="Country"': r'placeholder="{{ _(\'Country\') }}"',
    r'placeholder="Company"': r'placeholder="{{ _(\'Company\') }}"',
    r'>Filter<': r'>{{ _(\'Filter\') }}<',
    r'>View Job<': r'>{{ _(\'View Job\') }}<',
    r'>Visit<': r'>{{ _(\'Visit\') }}<',
    r'>No Link<': r'>{{ _(\'No Link\') }}<',
    r'<title>Resumatch - Job Listings</title>': r'<title>Resumatch - {{ _(\'Job Listings\') }}</title>',
    r'>No jobs found<': r'>{{ _(\'No jobs found\') }}<',
    r'>Try adjusting your search filters to find more results.<': r'>{{ _(\'Try adjusting your search filters to find more results.\') }}<',
    r'>Previous<': r'>{{ _(\'Previous\') }}<',
    r'>Next<': r'>{{ _(\'Next\') }}<'
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
    except FileNotFoundError:
        pass
print("Listings replacements done.")
