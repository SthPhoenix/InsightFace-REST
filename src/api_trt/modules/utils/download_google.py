import requests
import hashlib
from tqdm import tqdm
import re


def check_hash(filename, hash, algo='md5'):
    """Check whether hash of the file content matches the expected hash.
    Parameters
    ----------
    filename : str
        Path to the file.
    hash : str
        Expected hash in hexadecimal digits.
    algo: str
        Hashing algorithm (md5, sha1, sha256, sha512)

    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    algos = {
        'md5': hashlib.md5,
        'sha1': hashlib.sha1,
        'sha256': hashlib.sha256,
        'sha512': hashlib.sha512,
    }
    hasher = algos[algo]()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            hasher.update(data)

    file_hash = hasher.hexdigest()
    l = min(len(file_hash), len(hash))
    return hasher.hexdigest()[0:l] == hash[0:l]



# Script taken from https://stackoverflow.com/a/39225039
def download_from_gdrive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        if 'text/html' in response.headers['Content-Type']:
            m = re.search('.*confirm=([^\"]*)', response.text, re.M)
            if m and m.groups():
                return m.groups()[0]

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
