import glob
import os

files = glob.glob('templates/*.html')

for f in files:
    try:
        with open(f, 'r') as file:
            content = file.read()
            
        # Target exact broken syntax
        new_content = content.replace(r"{{ _(\'", "{{ _('")
        new_content = new_content.replace(r"\') }}", "') }}")
        
        # also check for inside attribute strings like placeholder="{{ _(\'Company\') }}"
        new_content = new_content.replace(r"{{ _(\"", "{{ _(\"")
        new_content = new_content.replace(r"\") }}", "\") }}")
        
        if new_content != content:
            with open(f, 'w') as file:
                file.write(new_content)
            print(f"Fixed backslashes in {f}")
    except Exception as e:
        print(f"Error reading {f}: {e}")
