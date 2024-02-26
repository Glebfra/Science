import pandas as pd
import matplotlib.pyplot as plt
import requests

data = pd.read_excel('./Saturation1.xlsx')
temperature = data['T1']
pressure = data['p1'] * 10 ** 6
density = 1 / data['V1']

for i in range(len(temperature)):
    requests.post(
        'http://api.localhost/database/saturation/',
        {
            'temperature': temperature[i],
            'pressure': pressure[i],
            'density': density[i],
            'element': 1,
            'state': 1
        },
        headers={
            'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNjk4Njk4MDIwLCJpYXQiOjE2OTgwOTMyMjAsImp0aSI6ImQ5NWQ0MmUyODc4YTRkOWQ5OTE0ZmI4NGIwMWUyYjViIiwidXNlcl9pZCI6MX0.FxoRcwqDqsJEvPBrpX-i4bB-lBG40diNXkyznEZWd6Q'
        }
    )
