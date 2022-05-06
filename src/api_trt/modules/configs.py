import os
from collections import namedtuple

# Net outputs in correct order expected by postprocessing code.
# TensorRT might change output order for some reasons.
# Also Triton Inference Server may change output order for both
# ONNX and TensorRT backends if automatic configuration is used.

retina_outputs = ['face_rpn_cls_prob_reshape_stride32',
                  'face_rpn_bbox_pred_stride32',
                  'face_rpn_landmark_pred_stride32',
                  'face_rpn_cls_prob_reshape_stride16',
                  'face_rpn_bbox_pred_stride16',
                  'face_rpn_landmark_pred_stride16',
                  'face_rpn_cls_prob_reshape_stride8',
                  'face_rpn_bbox_pred_stride8',
                  'face_rpn_landmark_pred_stride8']

anticov_outputs = [
    'face_rpn_cls_prob_reshape_stride32',
    'face_rpn_bbox_pred_stride32',
    'face_rpn_landmark_pred_stride32',
    'face_rpn_type_prob_reshape_stride32',
    'face_rpn_cls_prob_reshape_stride16',
    'face_rpn_bbox_pred_stride16',
    'face_rpn_landmark_pred_stride16',
    'face_rpn_type_prob_reshape_stride16',
    'face_rpn_cls_prob_reshape_stride8',
    'face_rpn_bbox_pred_stride8',
    'face_rpn_landmark_pred_stride8',
    'face_rpn_type_prob_reshape_stride8'
]

centerface_outputs = ['537', '538', '539', '540']
dbface_outputs = ["hm", "tlrb", "landmark"]
scrfd_outputs = ['score_8', 'score_16', 'score_32', 'bbox_8', 'bbox_16', 'bbox_32', 'kps_8', 'kps_16', 'kps_32']
yolo_outputs = ['output']

models = {
    'retinaface_mnet025_v0': {
        'shape': (1, 3, 480, 640),
        'outputs': retina_outputs,
        'reshape': True,
        'in_package': False,
        'function': 'retinaface_mnet025_v0',
        'link': '1PFDlZF8CYEr7TVnjGo_qVKVVlLP-7_du',
        'dl_type': 'google'
    },
    'retinaface_mnet025_v1': {
        'shape': (1, 3, 480, 640),
        'outputs': retina_outputs,
        'reshape': True,
        'in_package': False,
        'function': 'retinaface_mnet025_v1',
        'link': '12H4TXtGlAr1boEGtUukteolpQ9wfUTWe',
        'dl_type': 'google'
    },
    'retinaface_mnet025_v2': {
        'shape': (1, 3, 480, 640),
        'outputs': retina_outputs,
        'reshape': True,
        'in_package': False,
        'function': 'retinaface_mnet025_v2',
        'link': '1hzgOejAfCAB8WyfF24UkfiHD2FJbaCPi',
        'dl_type': 'google'
    },
    'retinaface_r50_v1': {
        'shape': (1, 3, 480, 640),
        'outputs': retina_outputs,
        'reshape': True,
        'in_package': False,
        'function': 'retinaface_r50_v1',
        'link': '1peUaq0TtNBhoXUbMqsCyQdL7t5JuhHMH',
        'dl_type': 'google'
    },
    'mnet_cov2': {
        'shape': (1, 3, 480, 640),
        'outputs': anticov_outputs,
        'reshape': True,
        'in_package': False,
        'function': 'mnet_cov2',
        'link': '1xPc3n_Y0jKyBONRx71UqCfcHjOGOLc2g',
        'dl_type': 'google'
    },
    'arcface_r100_v1': {
        'shape': (1, 3, 112, 112),
        'allow_batching': True,
        'reshape': False,
        'in_package': False,
        'function': 'arcface_mxnet',
        'link': '1sj170K3rbo5iOdjvjHw-hKWvXgH4dld3',
        'dl_type': 'google'
    },
    'genderage_v1': {
        'shape': (1, 3, 112, 112),
        'allow_batching': True,
        'reshape': False,
        'in_package': False,
        'function': 'genderage_v1',
        'link': '1MnkqBzQHLlIaI7gEoa9dd6CeknXMCyZH',
        'dl_type': 'google'
    },
    'centerface': {
        'in_package': False,
        'shape': (1, 3, 480, 640),
        'reshape': True,
        'function': 'centerface',
        'outputs': centerface_outputs,
        'link': 'https://raw.githubusercontent.com/Star-Clouds/CenterFace/master/models/onnx/centerface_bnmerged.onnx'
    },
    'dbface': {
        'in_package': False,
        'shape': (1, 3, 640, 640),
        'reshape': True,
        'function': 'dbface',
        'outputs': dbface_outputs,
        'dl_type': 'google'
    },
    'scrfd_500m_bnkps': {
        'in_package': False,
        'shape': (1, 3, 640, 640),
        'reshape': True,
        'function': 'scrfd',
        'outputs': scrfd_outputs,
        'allow_batching': True,
        'link': '1ccagW08CyKJoeFAmL2ao7pjwr3_m_Zm1',
        'md5': '03fd8fe67902798b7584b1e7ff3e0f6f',
        'dl_type': 'google'
    },
    'scrfd_2.5g_bnkps': {
        'in_package': False,
        'shape': (1, 3, 640, 640),
        'reshape': True,
        'function': 'scrfd',
        'outputs': scrfd_outputs,
        'allow_batching': True,
        'link': '1yObTSsOYCJFn38uf6O6Hku-XEhK8-q0n',
        'md5': '02710671a1af640b610b501383153868',
        'dl_type': 'google'
    },
    'scrfd_10g_bnkps': {
        'in_package': False,
        'shape': (1, 3, 640, 640),
        'reshape': True,
        'function': 'scrfd',
        'outputs': scrfd_outputs,
        'allow_batching': True,
        'link': '1llbQa8HFle-45wscZulzGcNncPFgYruJ',
        'md5': '2740f9259d46355ca5fa0d9b54943524',
        'dl_type': 'google'
    },

    'scrfd_10g_bnkps_v2': {
        'in_package': False,
        'shape': (1, 3, 640, 640),
        'reshape': True,
        'allow_batching': True,
        'function': 'scrfd',
        'outputs': scrfd_outputs,
    },

    'scrfd_10g_bnkps_e0': {
        'in_package': False,
        'shape': (1, 3, 640, 640),
        'reshape': True,
        'allow_batching': True,
        'function': 'scrfd',
        'outputs': scrfd_outputs,
    },

    'scrfd_500m_gnkps': {
        'in_package': False,
        'shape': (1, 3, 640, 640),
        'reshape': True,
        'function': 'scrfd',
        'outputs': scrfd_outputs,
        'allow_batching': True,
        'link': '19CeBV03a3DEhZeas4olZn7GgiUESDu0L',
        'md5': 'ef4bcb0606d6e5ebf325f8322e507a7b',
        'dl_type': 'google'
    },
    'scrfd_2.5g_gnkps': {
        'in_package': False,
        'shape': (1, 3, 640, 640),
        'reshape': True,
        'function': 'scrfd',
        'outputs': scrfd_outputs,
        'allow_batching': True,
        'link': '1_LeETpKhWL4sRPvLZEvka-bGNMN4tMOU',
        'md5': '50febd32caa699ef7a47cf7422c56bbd',
        'dl_type': 'google'
    },
    'scrfd_10g_gnkps': {
        'in_package': False,
        'shape': (1, 3, 640, 640),
        'reshape': True,
        'function': 'scrfd',
        'outputs': scrfd_outputs,
        'allow_batching': True,
        'link': '14BuXR6L73w1mwKXHPIcZlc9LYaid4Evl',
        'md5': '1d9b64bb0e6e18d4838872c9e7efd709',
        'dl_type': 'google'
    },
    'yolov5l-face': {
        'in_package': False,
        'shape': (1, 3, 640, 640),
        'reshape': True,
        'function': 'yolov5_face',
        'outputs': yolo_outputs,
        'allow_batching': True,
        'link': '1PL52lvybe1nJU5k09twbfKNRWw904HgS',
        'dl_type': 'google'
    },
    'yolov5m-face': {
        'in_package': False,
        'shape': (1, 3, 640, 640),
        'reshape': True,
        'function': 'yolov5_face',
        'outputs': yolo_outputs,
        'allow_batching': True,
        'link': '1degIq0DEFML97PFvfpi-mMN8mfzRzy5z',
        'dl_type': 'google'
    },
    'yolov5s-face': {
        'in_package': False,
        'shape': (1, 3, 640, 640),
        'reshape': True,
        'function': 'yolov5_face',
        'outputs': yolo_outputs,
        'allow_batching': True,
        'link': '14Ah6jfXJ5QuzaN2OsKE-g61x3-_hBnQV',
        'dl_type': 'google'
    },
    'yolov5n-face': {
        'in_package': False,
        'shape': (1, 3, 640, 640),
        'reshape': True,
        'function': 'yolov5_face',
        'outputs': yolo_outputs,
        'allow_batching': True,
        'link': '1P9u9vEGA9J2v1riAe3jIqZ-Zu5OrUCQN',
        'dl_type': 'google'
    },
    'yolov5n-0.5': {
        'in_package': False,
        'shape': (1, 3, 640, 640),
        'reshape': True,
        'function': 'yolov5_face',
        'outputs': yolo_outputs,
        'allow_batching': True,
        'link': '16gNzkxSByVtyYOMxRDRDn_qKfVxco1p0',
        'dl_type': 'google'
    },
    'scrfd_10g_gnkps_norm': {
        'in_package': False,
        'shape': (1, 3, 640, 640),
        'reshape': True,
        'allow_batching': True,
        'function': 'scrfd_v2',
        'outputs': scrfd_outputs,
        'link': '1Ks7kAHxSVnE-Zh0p99mVyQ9CLas1Amem',
        'dl_type': 'google'
    },
    'coordinateReg': {
        'in_package': False,
        'shape': (1, 3, 192, 192),
        'reshape': False,
    },
    'r100-arcface-msfdrop75': {
        'in_package': False,
        'shape': (1, 3, 112, 112),
        'allow_batching': True,
        'function': 'arcface_mxnet',
        'reshape': False,
    },
    'r50-arcface-msfdrop75': {
        'in_package': False,
        'shape': (1, 3, 112, 112),
        'allow_batching': True,
        'function': 'arcface_mxnet',
        'reshape': False,
    },
    'glint360k_r100FC_1.0': {
        'in_package': False,
        'shape': (1, 3, 112, 112),
        'allow_batching': True,
        'function': 'arcface_mxnet',
        'reshape': False
    },
    'glint360k_r100FC_0.1': {
        'in_package': False,
        'shape': (1, 3, 112, 112),
        'allow_batching': True,
        'function': 'arcface_mxnet',
        'reshape': False
    },
    'glintr100': {
        'in_package': False,
        'shape': (1, 3, 112, 112),
        'allow_batching': True,
        'reshape': False,
        'function': 'arcface_torch',
        'link': '1TR_ImGvuY7Dt22a9BOAUAlHasFfkrJp-',
        'dl_type': 'google'
    },
    'w600k_r50': {
        'in_package': False,
        'shape': (1, 3, 112, 112),
        'allow_batching': True,
        'function': 'arcface_torch',
        'reshape': False,
        'link': '1_3WcTE64Mlt_12PZHNWdhVCRpoPiblwq',
        'dl_type': 'google'
    },

    'w600k_mbf': {
        'in_package': False,
        'shape': (1, 3, 112, 112),
        'allow_batching': True,
        'function': 'arcface_torch',
        'reshape': False,
        'link': '1GtBKfGucgJDRLHvGWR3jOQovHYXY-Lpe',
        'dl_type': 'google'
    },

    'mask_detector': {
        'shape': (1, 224, 224, 3),
        'allow_batching': True,
        'reshape': False,
        'in_package': False,
        'function': 'mask_detector',
        'link': '1RsQonthhpJDwwdcB0sYsVGMTqPgGdMGV',
        'dl_type': 'google'
    },

    'mask_detector112': {
        'shape': (1, 112, 112, 3),
        'allow_batching': True,
        'reshape': False,
        'in_package': False,
        'function': 'mask_detector',
        'link': '1ghS0LEGV70Jdb5un5fVdDO-vmonVIe6Z',
        'dl_type': 'google'
    },

    # You can put your own pretrained ArcFace model to /models/onnx/custom_rec_model
    'custom_rec_model': {
        'in_package': False,
        'shape': (1, 3, 112, 112),
        'allow_batching': True,
        'function': 'arcface_torch',
        'reshape': False
    }
}


class Configs(object):
    def __init__(self, models_dir: str = '/models'):
        self.models_dir = self.__get_param('MODELS_DIR', models_dir)
        self.onnx_models_dir = os.path.join(self.models_dir, 'onnx')
        self.trt_engines_dir = os.path.join(self.models_dir, 'trt-engines')
        self.models = models
        self.type2path = dict(
            onnx=self.onnx_models_dir,
            engine=self.trt_engines_dir,
            plan=self.trt_engines_dir
        )

    def __get_param(self, ENV, default=None):
        return os.environ.get(ENV, default)

    def build_model_paths(self, model_name: str, ext: str):
        base = self.type2path[ext]
        parent = os.path.join(base, model_name)
        file = os.path.join(parent, f"{model_name}.{ext}")
        return parent, file

    def get_outputs_order(self, model_name):
        return self.models.get(model_name, {}).get('outputs')

    def get_shape(self, model_name):
        return self.models.get(model_name, {}).get('shape')

    def get_dl_link(self, model_name):
        return self.models.get(model_name, {}).get('link')

    def get_dl_type(self, model_name):
        return self.models.get(model_name, {}).get('dl_type')
