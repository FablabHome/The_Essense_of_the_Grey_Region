

import requests

resp = requests.get('https://std.puiching.edu.mo/~0763236-3/cgi-bin/WRS_color_write.py?drink=black')
data = resp.json()
print(data)
