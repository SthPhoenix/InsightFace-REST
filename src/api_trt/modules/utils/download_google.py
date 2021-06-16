import requests
from tqdm import tqdm

# Script taken from https://stackoverflow.com/a/39225039
def download_from_gdrive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 1024

        total_length = response.headers.get('content-length')
        if total_length is None:
            range = response.headers.get('content-range')
            if range:
                total_length = int(range.partition('/')[-1])

        with open(destination, 'wb') as f:
            if total_length is None:  # no content length header
                for chunk in tqdm(response.iter_content(chunk_size=CHUNK_SIZE), unit='KB',
                                  unit_scale=False, dynamic_ncols=True):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(response.iter_content(chunk_size=CHUNK_SIZE),
                                  total=int(total_length / 1024. + 0.5),
                                  unit='KB', unit_scale=False, dynamic_ncols=True):
                    f.write(chunk)
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        headers = {'Range': 'bytes=0-'}
        response = session.get(URL, params=params, headers=headers, stream=True)

    if response.status_code not in (200, 206):
        raise RuntimeError(f"Failed downloading file {id}")

    save_response_content(response, destination)
