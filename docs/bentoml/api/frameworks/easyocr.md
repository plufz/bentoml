# EasyOCR Framework Integration

BentoML provides integration for EasyOCR, allowing easy saving, loading, and managing of EasyOCR models within the BentoML ecosystem.

## Functions

### `save_model()`

```python
bentoml.easyocr.save_model(
    name: str,
    reader: easyocr.Reader,
    signatures: ModelSignaturesType | None = None,
    labels: dict[str, str] | None = None,
    custom_objects: dict[str, Any] | None = None,
    external_modules: List[ModuleType] | None = None,
    metadata: dict[str, Any] | None = None
) -> bentoml.Model
```

**Description**: Saves an EasyOCR model instance to the BentoML model store.

**Key Parameters**:
- `name`: Name for the model instance
- `reader`: The EasyOCR model to be saved
- `signatures`: Methods for model inference
- `labels`: User-defined labels for model management

**Example**:
```python
import bentoml
import easyocr

reader = easyocr.Reader(['en'])
bento_model = bentoml.easyocr.save_model('en_reader', reader)
```

### `load_model()`

```python
bentoml.easyocr.load_model(
    bento_model: str | Tag | Model
) -> easyocr.Reader
```

**Description**: Loads an EasyOCR model from the BentoML local model store.

**Example**:
```python
import bentoml
reader = bentoml.easyocr.load_model('en_reader:latest')
```

### `get()`

```python
bentoml.easyocr.get(
    tag_like: str | Tag
) -> Model
```

**Description**: Retrieves a BentoML model with the given tag from the model store.

**Example**:
```python
import bentoml
model = bentoml.easyocr.get("en_reader:latest")
```

## Usage in BentoML Services

```python
import bentoml
from bentoml.io import Image, JSON

# Load the model
reader_runner = bentoml.easyocr.get("en_reader:latest").to_runner()

svc = bentoml.Service("easyocr_service", runners=[reader_runner])

@svc.api(input=Image(), output=JSON())
def predict(input_img):
    result = reader_runner.readtext.run(input_img)
    return {"text": result}
```

## Model Management

EasyOCR models can be managed through BentoML's standard model management commands:

- List models: `bentoml models list`
- Delete model: `bentoml models delete en_reader:latest`
- Get model info: `bentoml models get en_reader:latest`