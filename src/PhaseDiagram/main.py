import pandas as pd
import requests

data = pd.read_excel('./PhaseDiagram.xlsx')
temperature = data['T'] + 273.15
pressure = data['p'] * 10 ** 5
density = 1 / data['V']

for i in range(len(temperature)):
    requests.post(
        'http://api.localhost/database/phase_diagram/',
        {
            'temperature': temperature[i],
            'pressure': pressure[i],
            'density': density[i],
            'element': 1
        },
        headers={
            'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNjk4Njk4MDIwLCJpYXQiOjE2OTgwOTMyMjAsImp0aSI6ImQ5NWQ0MmUyODc4YTRkOWQ5OTE0ZmI4NGIwMWUyYjViIiwidXNlcl9pZCI6MX0.FxoRcwqDqsJEvPBrpX-i4bB-lBG40diNXkyznEZWd6Q'
        }
    )
