import app as test_app
import unittest
import os
import json
import base64

dir_path = os.path.dirname(os.path.realpath(__file__))
test_cat = os.path.join(dir_path, 'test_images')


def file2base64(path):
    with open(path, mode='rb') as fl:
        encoded = base64.b64encode(fl.read()).decode('ascii')
        return encoded


def extract_vecs(ims, app):
    target = [file2base64(im) for im in ims]
    req = {'images': {"data": target}}
    resp = app.post('/extract', json=req)
    content = resp.data
    values = json.loads(content.decode('utf-8'))
    return values


def compare(im1, ims, app):
    source = file2base64(im1)
    target = [file2base64(im) for im in ims]
    req = {'source': {"data": source}, 'target': {"data": target}}
    resp = app.post('/ver', json=req)
    content = resp.data
    values = json.loads(content.decode('utf-8'))
    return values


class FlaskTestCase(unittest.TestCase):

    def setUp(self):
        test_app.app.config['TESTING'] = True
        self.app = test_app.app.test_client()

    def tearDown(self):
        pass

    def test_hello(self):
        rv = self.app.get('/')
        assert b'Hello, This is InsightFace!' in rv.data

    def test_broken_embed(self):
        image = os.path.join(test_cat, 'broken.jpg')
        embs = extract_vecs([image], self.app)
        assert len(embs[0]) == 0

    def test_single_embed(self):
        image = os.path.join(test_cat, 'TH.png')
        embs = extract_vecs([image], self.app)
        assert len(embs[0]) == 1

    def test_group_embed(self):
        image = os.path.join(test_cat, 'group.jpg')
        embs = extract_vecs([image], self.app)
        assert len(embs[0]) > 40

    def test_mixed(self):
        images = ['broken.jpg', 'TH.png', 'group.jpg']
        images = [os.path.join(test_cat, im) for im in images]
        embs = extract_vecs(images, self.app)
        assert len(embs[0]) == 0
        assert len(embs[1]) == 1
        assert len(embs[2]) > 40

    def test_verfication(self):
        source = os.path.join(test_cat, 'TH.png')
        images = ['TH1.jpg', 'Stallone.jpg']
        target = [os.path.join(test_cat, im) for im in images]
        res = compare(source, target, self.app)
        assert res[0] > 0.7
        assert res[1] < 0.60


if __name__ == '__main__':
    unittest.main()
