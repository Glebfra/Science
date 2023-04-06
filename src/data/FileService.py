import json
import os


class FileService(object):
    def __init__(self):
        self.data_dir = f'{os.getenv("PROJECT_DIR")}{os.getenv("DATA_FOLDER")}'

    def load_file(self, filename: str, type: str = 'json', **kwargs) -> dict:
        filepath = self.data_dir
        if 'filepath' in kwargs:
            filepath = kwargs['filepath']

        if type == 'json':
            with open(f'{filepath}/{filename}') as file:
                data = json.load(file, **kwargs)
        else:
            raise TypeError(f'The type {type} is not compatible')

        return data

    def save_file(self, data: list | dict, filename: str, type: str = 'json', **kwargs) -> None:
        filepath = self.data_dir
        if 'filepath' in kwargs:
            filepath = kwargs['filepath']

        if type == 'json':
            with open(f'{filepath}/{filename}') as file:
                json.dump(data, file, **kwargs)
        else:
            raise TypeError(f'The type {type} is not compatible')
