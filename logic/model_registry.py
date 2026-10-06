"""Model identities are independent of execution device and GUI selection."""
from config.model import Model


def model_spec(model_id=None):
    model_id = model_id or Model.DEFAULT_MODEL
    if model_id not in Model.DETECTORS:
        raise ValueError(f"Unknown detector: {model_id}")
    spec = dict(Model.DETECTORS[model_id], model_id=model_id)
    spec['path'] = Model.MODEL_DIR / spec['filename']
    spec['hashes'] = {name: value for name, value in Model.SHA256.items()
                      if name in (Model.SHAPE_PATH.name, Model.CLASSIFIER_PATH.name)}
    spec['hashes'][spec['filename']] = spec['sha256']
    return spec


def model_for_pipeline(pipeline):
    return next((name for name, spec in Model.DETECTORS.items() if spec['pipeline'] == pipeline), None)


def compatible_sidecar(metadata):
    """Allow genuine legacy S metadata, but never conflicting supplied identity."""
    name = model_for_pipeline(metadata.get('pipeline'))
    if name is None:
        return False
    spec = model_spec(name)
    if metadata.get('model_id', name) != name:
        return False
    if metadata.get('nice_threshold', Model.NICE_THRESHOLD) != Model.NICE_THRESHOLD:
        return False
    provenance = metadata.get('provenance', {})
    if not isinstance(provenance, dict):
        return False
    expected = dict(model_id=name, models=spec['hashes'], confidence=spec['confidence'], nice_threshold=Model.NICE_THRESHOLD)
    if any(key in provenance and provenance[key] != value for key, value in expected.items()):
        return False
    return name == 's' or all(key in provenance for key in ('models', 'confidence', 'model_id'))
