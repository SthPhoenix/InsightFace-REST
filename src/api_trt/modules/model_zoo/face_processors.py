# Backend wrapper for older models trained with MXNet (doesn't require image normalization)
def arcface_mxnet(model_path, backend, **kwargs):
    model = backend.Arcface(rec_name=model_path, **kwargs)
    return model


# Backend wrapper for PyTorch trained models, which requires image normalization
def arcface_torch(model_path, backend, **kwargs):
    model = backend.Arcface(rec_name=model_path, input_mean=127.5, input_std=127.5, **kwargs)
    return model


# Backend wrapper for Gender/Age estimation model.
def genderage_v1(model_path, backend, **kwargs):
    model = backend.FaceGenderage(rec_name=model_path, **kwargs)
    return model

# Backend wrapper for mask detection model.
def mask_detector(model_path, backend, **kwargs):
    model = backend.MaskDetection(rec_name=model_path, **kwargs)
    return model
