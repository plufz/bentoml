# BentoML CLI

The BentoML CLI provides a comprehensive set of commands for managing machine learning model services, including building, deploying, and managing Bento packages and models.

## Main Commands

### `bentoml build`

Builds a new Bento from the current directory.

**Syntax:**
```
bentoml build [OPTIONS] [BUILD_CTX]
```

**Key Options:**
- `-f, --bentofile`: Path to bentofile (default: 'bentofile.yaml')
- `--name`: Specify Bento name
- `--version`: Specify Bento version
- `--containerize`: Automatically containerize the Bento after building
- `--push`: Push the Bento to BentoCloud

**Example:**
```bash
bentoml build --name my_service --version v1.0
bentoml build -f custom_bentofile.yaml
bentoml build --containerize --push
```

### `bentoml models`

Manages machine learning models in the local model store.

**Subcommands:**
- `list`: List saved models
- `delete`: Remove models from local store
- `export`: Export a model to an archive
- `import`: Import a model from an archive
- `pull`: Retrieve models from a remote store
- `push`: Upload models to a remote store

**Examples:**
```bash
bentoml models list
bentoml models delete iris_classifier:v1
bentoml models export my_model:latest ./model_archive.tar
bentoml models import ./model_archive.tar
```

### `bentoml serve`

Starts a HTTP server for a Bento service.

**Syntax:**
```
bentoml serve [OPTIONS] [BENTO]
```

**Key Options:**
- `--port`: Specify server port
- `--host`: Set binding host
- `--reload`: Enable auto-reload during development
- `--development`: Run in development mode
- `--production`: Run in production mode

**Examples:**
```bash
bentoml serve fraud_detector:latest --port 3000 --reload
bentoml serve service.py:MyService --development
bentoml serve my_bento:latest --host 0.0.0.0 --port 8080
```

### `bentoml containerize`

Creates an OCI-compliant container image for a Bento service.

**Syntax:**
```
bentoml containerize [OPTIONS] BENTO:TAG
```

**Key Options:**
- `-t, --image-tag`: Set container image name and tag
- `--platform`: Target platform for multi-arch builds
- `--progress`: Set build progress output type

**Examples:**
```bash
bentoml containerize my_service:latest -t my_service:v1.0
bentoml containerize my_service:latest --platform linux/amd64,linux/arm64
```

### `bentoml list`

Lists all Bentos in the local Bento store.

**Examples:**
```bash
bentoml list
bentoml list --output json
```

### `bentoml delete`

Removes Bentos from the local store.

**Examples:**
```bash
bentoml delete my_service:v1.0
bentoml delete my_service  # Deletes all versions
```

### `bentoml export`

Exports a Bento to an archive file.

**Examples:**
```bash
bentoml export my_service:latest ./my_service.tar
bentoml export my_service:v1.0 ./exports/
```

### `bentoml import`

Imports a Bento from an archive file.

**Examples:**
```bash
bentoml import ./my_service.tar
bentoml import ./exports/my_service_v1.0.tar
```

## Development Workflow Commands

### Development Mode
For active development with auto-reload:
```bash
bentoml serve service.py:MyService --development --reload
```

### Production Mode
For production deployment:
```bash
bentoml serve my_bento:latest --production --host 0.0.0.0
```

### Build and Containerize Pipeline
```bash
# Build the bento
bentoml build

# Containerize the built bento
bentoml containerize my_service:latest -t my_service:production

# Or build and containerize in one step
bentoml build --containerize -t my_service:latest
```

## Global Options

Most commands support these global options:
- `--help`: Show help information
- `--version`: Show BentoML version
- `--verbose`: Enable verbose logging
- `--quiet`: Suppress output

## Configuration

CLI behavior can be configured through:
- Environment variables
- Configuration files
- Command-line options

See the configurations documentation for detailed setup options.