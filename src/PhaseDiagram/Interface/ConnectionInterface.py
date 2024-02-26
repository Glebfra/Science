import requests


class ConnectionInterface:
    TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNjk5ODkzNTA5LCJpYXQiOjE2OTkyODg3MDksImp0aSI6ImJkYzMwYWVhMDkwZDQxNGY4MWRlOGNjNzUzMzVmMGIxIiwidXNlcl9pZCI6MX0.bPo0AKvC8JSpK7t1US6YZTAv1KviJ4Wws-TPcMGlHVU'
    HEADERS = {
        'Authorization': f'Bearer {TOKEN}'
    }

    def secure_post(self, *args, **kwargs):
        return requests.post(*args, **kwargs, headers=self.HEADERS)

    @staticmethod
    def unsecure_post(*args, **kwargs):
        return requests.post(*args, **kwargs)

    def secure_get(self, *args, **kwargs):
        return requests.get(*args, **kwargs, headers=self.HEADERS)

    @staticmethod
    def unsecure_get(*args, **kwargs):
        return requests.get(*args, **kwargs)

    def update_token(self):
        data = requests.post(
            'http://api.localhost/token/',
            {
                'username': 'root',
                'password': 'root'
            }
        ).json()

        self.TOKEN = data['access']
        self.HEADERS = data['Authorization'] = f'Bearer {self.TOKEN}'
