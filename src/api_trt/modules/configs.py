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
scrfd_500m_bnkps_outputs = ['443', '468', '493', '446', '471', '496', '449', '474', '499']
scrfd_2_5g_bnkps_outputs = ['446', '466', '486', '449', '469', '489', '452', '472', '492']
scrfd_10g_bnkps_outputs = ['448', '471', '494', '451', '474', '497', '454', '477', '500']
scrfd_500m_gnkps_outputs = ['447', '512', '577', '450', '515', '580', '453', '518', '583']
scrfd_2_5g_gnkps_outputs = ['448', '488', '528', '451', '491', '531', '454', '494', '534']
scrfd_10g_gnkps_outputs = ['451', '504', '557', '454', '507', '560', '457', '510', '563']


models = {
    'retinaface_mnet025_v0': {
        'shape': (1, 3, 480, 640),
        'outputs': retina_outputs,
        'reshape': True,
        'in_package': False,
        'link': '1PFDlZF8CYEr7TVnjGo_qVKVVlLP-7_du',
        'dl_type': 'google'
    },
    'retinaface_mnet025_v1': {
        'shape': (1, 3, 480, 640),
        'outputs': retina_outputs,
        'reshape': True,
        'in_package': False,
        'link': '12H4TXtGlAr1boEGtUukteolpQ9wfUTWe',
        'dl_type': 'google'
    },
    'retinaface_mnet025_v2': {
        'shape': (1, 3, 480, 640),
        'outputs': retina_outputs,
        'reshape': True,
        'in_package': False,
        'link': '1hzgOejAfCAB8WyfF24UkfiHD2FJbaCPi',
        'dl_type': 'google'
    },
    'retinaface_r50_v1': {
        'shape': (1, 3, 480, 640),
        'outputs': retina_outputs,
        'reshape': True,
        'in_package': False,
        'link': '1peUaq0TtNBhoXUbMqsCyQdL7t5JuhHMH',
        'dl_type': 'google'
    },
    'mnet_cov2': {
        'shape': (1, 3, 480, 640),
        'outputs': anticov_outputs,
        'reshape': True,
        'in_package': False,
        'link': '1xPc3n_Y0jKyBONRx71UqCfcHjOGOLc2g',
        'dl_type': 'google'
    },
    'arcface_r100_v1': {
        'shape': (1, 3, 112, 112),
        'allow_batching': True,
        'reshape': False,
        'in_package': False,
        'link': '1sj170K3rbo5iOdjvjHw-hKWvXgH4dld3',
        'dl_type': 'google'
    },
    'genderage_v1': {
        'shape': (1, 3, 112, 112),
        'allow_batching': True,
        'reshape': False,
        'in_package': False,
        'link': '1MnkqBzQHLlIaI7gEoa9dd6CeknXMCyZH',
        'dl_type': 'google'
    },
    'centerface': {
        'in_package': False,
        'shape': (1, 3, 480, 640),
        'reshape': True,
        'outputs': centerface_outputs,
        'link': 'https://raw.githubusercontent.com/Star-Clouds/CenterFace/master/models/onnx/centerface_bnmerged.onnx'
    },

    'dbface': {
        'in_package': False,
        'shape': (1, 3, 640, 640),
        'reshape': True,
        'outputs': dbface_outputs,
        'dl_type': 'google'
    },

    'scrfd_500m_bnkps': {
        'in_package': False,
        'shape': (1, 3, 640, 640),
        'reshape': True,
        'outputs': scrfd_500m_bnkps_outputs,
        'link': '13mY-c6NIShu_-4AdCo3Z3YIYja4HfNaA',
        'dl_type': 'google'
    },

    'scrfd_2.5g_bnkps': {
        'in_package': False,
        'shape': (1, 3, 640, 640),
        'reshape': True,
        'outputs': scrfd_2_5g_bnkps_outputs,
        'link': '1qnKTHMkuoWsCJ6iJeiFExGy5PSi8JKPL',
        'dl_type': 'google'
    },

    'scrfd_10g_bnkps': {
        'in_package': False,
        'shape': (1, 3, 640, 640),
        'reshape': True,
        'outputs': scrfd_10g_bnkps_outputs,
        'link': '1OAXx8U8SIsBhmYYGKmD-CLXrYz_YIV-3',
        'dl_type': 'google'
    },

    'scrfd_500m_gnkps': {
        'in_package': False,
        'shape': (1, 3, 640, 640),
        'reshape': True,
        'outputs': scrfd_500m_gnkps_outputs,
        'link': '13OoTQlyDI2BkuA5oJUtuuvMlxvkM_-h7',
        'dl_type': 'google'
    },

    'scrfd_2.5g_gnkps': {
        'in_package': False,
        'shape': (1, 3, 640, 640),
        'reshape': True,
        'outputs': scrfd_2_5g_gnkps_outputs,
        'link': '1F__ILEeCTzeR71BAV-vInuyBezYmNMsB',
        'dl_type': 'google'
    },

    'scrfd_10g_gnkps': {
        'in_package': False,
        'shape': (1, 3, 640, 640),
        'reshape': True,
        'outputs': scrfd_10g_gnkps_outputs,
        'link': '1v9nhtPWMLSedueeL6c3nJEoIFlSNSCvh',
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
        'reshape': False,
    },
    'r50-arcface-msfdrop75': {
        'in_package': False,
        'shape': (1, 3, 112, 112),
        'allow_batching': True,
        'reshape': False,
    },
    'glint360k_r100FC_1.0': {
        'in_package': False,
        'shape': (1, 3, 112, 112),
        'allow_batching': True,
        'reshape': False
    },
    'glint360k_r100FC_0.1': {
        'in_package': False,
        'shape': (1, 3, 112, 112),
        'allow_batching': True,
        'reshape': False
    },

    'glintr100': {
        'in_package': False,
        'shape': (1, 3, 112, 112),
        'allow_batching': True,
        'reshape': False,
        'link': '1TR_ImGvuY7Dt22a9BOAUAlHasFfkrJp-',
        'dl_type': 'google'
    },

    # You can put your own pretrained ArcFace model to /models/onnx/custom_rec_model
    'custom_rec_model': {
        'in_package': False,
        'shape': (1, 3, 112, 112),
        'allow_batching': True,
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
