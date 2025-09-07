# Detectron Framework Integration

The Detectron framework integration in BentoML provides methods for saving, loading, and managing Detectron2 models.

## Functions

### `save_model()`

```python
bentoml.detectron.save_model(
    name: str,
    checkpointables: Engine.DefaultPredictor | nn.Module,
    config: Config.CfgNode | None = None,
    signatures: ModelSignaturesType | None = None,
    labels: dict[str, str] | None = None,
    custom_objects: dict[str, Any] | None = None,
    external_modules: List[ModuleType] | None = None,
    metadata: dict[str, Any] | None = None
) -> bentoml.Model
```

**Description**: Saves a Detectron model instance to the BentoML model store.

**Parameters**:
- `name`: Name for the model instance
- `checkpointables`: Model instance (DefaultPredictor or nn.Module)
- `config`: Optional configuration for the model
- `signatures`: Methods for model inference
- `labels`: User-defined labels for model management
- `custom_objects`: Custom objects to be saved with the model
- `external_modules`: List of external modules
- `metadata`: Additional metadata

**Examples**:

```python
# Saving a model from ModelZoo
import bentoml
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
bento_model = bentoml.detectron.save_model('mask_rcnn', build_model(cfg), config=cfg)

# Saving a Predictor
from detectron2.engine import DefaultPredictor
predictor = DefaultPredictor(cfg)
bento_model = bentoml.detectron.save_model('mask_rcnn_predictor', predictor)
```

### `load_model()`

```python
bentoml.detectron.load_model(
    bento_model: str | Tag | Model
) -> Engine.DefaultPredictor | nn.Module
```

**Description**: Loads a Detectron model from the BentoML local model store.

**Example**:
```python
import bentoml
model = bentoml.detectron.load_model('mask_rcnn:latest')
```

### `get()`

```python
bentoml.detectron.get(
    tag_like: str | Tag
) -> Model
```

**Description**: Retrieves a BentoML model with the given tag from the model store.

**Example**:
```python
import bentoml
model = bentoml.detectron.get("mask_rcnn:latest")
```

## Usage in BentoML Services

```python
import bentoml
from bentoml.io import Image, JSON
import cv2
import numpy as np

# Load the model
detectron_runner = bentoml.detectron.get("mask_rcnn:latest").to_runner()

svc = bentoml.Service("detectron_service", runners=[detectron_runner])

@svc.api(input=Image(), output=JSON())
def predict(input_img):
    # Convert PIL image to OpenCV format
    img_array = np.array(input_img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Run inference
    outputs = detectron_runner.run(img_bgr)
    
    # Process results
    instances = outputs["instances"]
    return {
        "boxes": instances.pred_boxes.tensor.tolist(),
        "scores": instances.scores.tolist(),
        "classes": instances.pred_classes.tolist()
    }
```

## Model Management

Detectron models can be managed through BentoML's standard model management commands:

- List models: `bentoml models list`
- Delete model: `bentoml models delete mask_rcnn:latest`
- Get model info: `bentoml models get mask_rcnn:latest`

## Configuration

When saving Detectron2 models, it's recommended to include the configuration object to ensure proper model reconstruction:

```python
import bentoml
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

cfg = get_cfg()
cfg.merge_from_file("path/to/config.yaml")
predictor = DefaultPredictor(cfg)

# Save with configuration
bento_model = bentoml.detectron.save_model(
    'my_detectron_model',
    predictor,
    config=cfg
)
```