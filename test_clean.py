import re
import html

def clean_text(text):
    if not text: return ""
    text = html.unescape(text) # convert &amp; -> &, &nbsp; -> space
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

print(clean_text("Hello &amp; welcome &lt;b&gt;fejlesztő&lt;/b&gt; &nbsp;&#39;hello&#39; !"))
