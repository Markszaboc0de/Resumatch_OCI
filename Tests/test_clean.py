import re
text = "<style>body{color:red;}</style><h1>Title</h1><p>Description</p>"
t1 = re.sub(r'<[^>]+>', ' ', text)
print("Old clean:", t1)

t2 = re.sub(r'<(script|style)[^>]*>.*?</\1>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
t2 = re.sub(r'<[^>]+>', ' ', t2)
print("New clean:", t2)

