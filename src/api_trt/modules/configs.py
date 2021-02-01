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

anticonv_outputs = [
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

mxnet_models = {
    'retinaface_mnet025_v0': {
        'symbol': 'mnet.25-symbol.json',
        'params': 'mnet.25-0000.params',
        'shape': (1, 3, 480, 640),
        'outputs': retina_outputs,
        'reshape': True,
        'in_package': False
    },
    'retinaface_mnet025_v1': {
        'symbol': 'mnet10-symbol.json',
        'params': 'mnet10-0000.params',
        'shape': (1, 3, 480, 640),
        'outputs': retina_outputs,
        'reshape': True,
        'in_package': True,
    },
    'retinaface_mnet025_v2': {
        'symbol': 'mnet12-symbol.json',
        'params': 'mnet12-0000.params',
        'shape': (1, 3, 480, 640),
        'outputs': retina_outputs,
        'reshape': True,
        'in_package': True,
    },
    'retinaface_r50_v1': {
        'symbol': 'R50-symbol.json',
        'params': 'R50-0000.params',
        'shape': (1, 3, 480, 640),
        'outputs': retina_outputs,
        'reshape': True,
        'in_package': True
    },
    'mnet_cov2': {
        'symbol': 'mnet_cov2-symbol.json',
        'params': 'mnet_cov2-0000.params',
        'shape': (1, 3, 480, 640),
        'outputs': anticonv_outputs,
        'reshape': True,
        'in_package': False
    },
    'arcface_r100_v1': {
        'symbol': 'model-symbol.json',
        'params': 'model-0000.params',
        'shape': (1, 3, 112, 112),
        'reshape': False,
        'in_package': True
    },
    'genderage_v1': {
        'symbol': 'model-symbol.json',
        'params': 'model-0000.params',
        'shape': (1, 3, 112, 112),
        'reshape': False,
        'in_package': True
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
        'outputs': dbface_outputs
    },

    'coordinateReg': {
        'symbol': '2d106det-symbol.json',
        'params': '2d106det-0000.params',
        'in_package': False,
        'shape': (1, 3, 192, 192),
        'reshape': False
    },
    'r100-arcface-msfdrop75': {
        'symbol': 'model-symbol.json',
        'params': 'model-0000.params',
        'in_package': False,
        'shape': (1, 3, 112, 112),
        'reshape': False
    },
    'r50-arcface-msfdrop75': {
        'symbol': 'model-symbol.json',
        'params': 'model-0000.params',
        'in_package': False,
        'shape': (1, 3, 112, 112),
        'reshape': False
    },
    'glint360k_r100FC_1.0': {
        'symbol': 'model-symbol.json',
        'params': 'model-0000.params',
        'in_package': False,
        'shape': (1, 3, 112, 112),
        'reshape': False
    },
    'glint360k_r100FC_0.1': {
        'symbol': 'model-symbol.json',
        'params': 'model-0000.params',
        'in_package': False,
        'shape': (1, 3, 112, 112),
        'reshape': False
    }
}

models_repo = 'https://drive.google.com/drive/folders/109D__GLXHPmiW9tIgsCadTdjcXB0u0vK'


class Configs(object):
    def __init__(self, models_dir: str = '/models'):
        self.models_dir = self.__get_param('MODELS_DIR', models_dir)
        self.mxnet_models_dir = os.path.join(self.models_dir, 'mxnet')
        self.onnx_models_dir = os.path.join(self.models_dir, 'onnx')
        self.trt_engines_dir = os.path.join(self.models_dir, 'trt-engines')
        self.mxnet_models = mxnet_models
        self.type2path = dict(
            mxnet=self.mxnet_models_dir,
            onnx=self.onnx_models_dir,
            engine=self.trt_engines_dir,
            plan=self.trt_engines_dir
        )

    def __get_param(self, ENV, default=None):
        return os.environ.get(ENV, default)

    def get_mxnet_model_paths(self, model_name):
        symbol_path = os.path.join(self.mxnet_models_dir, model_name, self.mxnet_models[model_name].get('symbol', ''))
        param_path = os.path.join(self.mxnet_models_dir, model_name, self.mxnet_models[model_name].get('params', ''))
        return symbol_path, param_path

    def in_official_package(self, model_name):
        return mxnet_models[model_name]['in_package']

    def build_model_paths(self, model_name: str, ext: str):
        base = self.type2path[ext]
        parent = os.path.join(base, model_name)
        file = os.path.join(parent, f"{model_name}.{ext}")
        return parent, file

    def get_outputs_order(self, model_name):
        return self.mxnet_models.get(model_name, {}).get('outputs')

    def get_shape(self, model_name):
        return self.mxnet_models.get(model_name, {}).get('shape')

    def get_dl_link(self, model_name):
        return self.mxnet_models.get(model_name, {}).get('link')
