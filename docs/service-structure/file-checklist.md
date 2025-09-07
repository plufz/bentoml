# Service File Checklist

This checklist ensures you create and update all necessary files when adding or modifying BentoML services.

## New Service Creation Checklist

### ✅ Required Files

#### Core Implementation
- [ ] **`services/your_service.py`** - Main service implementation
  - Service class with `@bentoml.service()` decorator
  - API endpoints with `@bentoml.api` decorator
  - Pydantic request/response models
  - Proper error handling and logging

- [ ] **`config/bentofiles/your-service.yaml`** - Bento build configuration
  - Service path and name
  - Dependencies and requirements
  - Docker configuration (if needed)
  - Include/exclude file patterns

- [ ] **`tests/test_your_service.py`** - Unit and integration tests
  - Unit test class (`TestYourServiceUnit`)
  - Integration test class (`TestYourServiceIntegration`) 
  - Test fixtures and mocked dependencies
  - HTTP endpoint testing

#### Optional Files (Create as Needed)
- [ ] **`utils/your_service_utils.py`** - Utility functions (if complex logic needed)
- [ ] **`config/your_service_config.yaml`** - Service-specific configuration (if needed)
- [ ] **`docs/services/your-service.md`** - Detailed service documentation (recommended)

### 📝 Files to Update

#### Core Documentation
- [ ] **`README.md`** - Add service to main project documentation
  - Add to services list
  - Update endpoint examples
  - Add to quick start guide

- [ ] **`CLAUDE.md`** - Update AI assistant instructions
  - Add service to key files list
  - Add endpoint testing examples
  - Update multi-service endpoint count
  - Add service to building/running sections

#### Scripts and Automation
- [ ] **`scripts/build_services.sh`** - Add service build command
  ```bash
  echo "Building Your Service..."
  BENTOFILE=config/bentofiles/your-service.yaml ./scripts/run_bentoml.sh build services/your_service.py
  ```

- [ ] **`scripts/test.sh`** - Add service-specific test option
  ```bash
  your_service)
      echo "Running Your Service tests..."
      uv run pytest tests/test_your_service.py -v
      ;;
  ```

#### Dependencies
- [ ] **`pyproject.toml`** - Add new dependencies (if not in Bentofile)
  - Update `dependencies` section with new packages
  - Ensure version compatibility

#### Multi-Service Integration (if applicable)
- [ ] **`services/multi_service.py`** - Integrate with unified service
  - Import your service classes
  - Initialize service in `__init__`
  - Add endpoint methods
  - Update health/info endpoints

- [ ] **`config/bentofiles/multi-service.yaml`** - Update multi-service config
  - Add your service dependencies
  - Update system packages if needed

#### Documentation Structure
- [ ] **`docs/README.md`** - Update documentation index
  - Add link to your service documentation
  - Update service list in overview

### 🧪 Testing Checklist

#### Test Implementation
- [ ] **Unit Tests** - Test service logic with mocked dependencies
- [ ] **Integration Tests** - Test actual service startup and HTTP endpoints
- [ ] **Error Handling Tests** - Test error scenarios and edge cases
- [ ] **Parameterized Tests** - Test multiple input variations
- [ ] **Mock Setup** - Properly mock heavy dependencies (models, external APIs)

#### Test Execution
- [ ] **Local Testing** - Run tests locally and ensure they pass
  ```bash
  ./scripts/test.sh --service your_service
  ```
- [ ] **Coverage Testing** - Ensure adequate test coverage
  ```bash
  ./scripts/test.sh --coverage
  ```
- [ ] **Integration Testing** - Test service startup and HTTP endpoints
  ```bash
  ./scripts/test.sh --all
  ```

### 🔧 Development Integration

#### Build and Deployment
- [ ] **Service Builds Successfully** - Verify Bento build works
  ```bash
  BENTOFILE=config/bentofiles/your-service.yaml ./scripts/run_bentoml.sh build services/your_service.py
  ```

- [ ] **Service Starts Successfully** - Verify service can start
  ```bash
  ./scripts/run_bentoml.sh serve services.your_service:YourService
  ```

- [ ] **Endpoints Respond** - Test endpoints with sample data
  ```bash
  ./scripts/endpoint.sh your_endpoint '{"input_field": "test"}'
  ```

#### Health and Monitoring
- [ ] **Health Check Implementation** - Add health endpoint to your service
- [ ] **Health Check Integration** - Verify health check works
  ```bash
  ./scripts/health.sh
  ```

- [ ] **Error Logging** - Ensure proper error logging and handling

## Service Update Checklist

### 🔄 When Modifying Existing Services

#### Code Changes
- [ ] **Service Implementation** - Update `services/your_service.py`
- [ ] **Utility Functions** - Update `utils/your_service_utils.py` (if exists)
- [ ] **Request/Response Models** - Update Pydantic models
- [ ] **Dependencies** - Update `config/bentofiles/your-service.yaml` if new packages needed

#### Testing Updates
- [ ] **Update Unit Tests** - Modify `tests/test_your_service.py`
- [ ] **Add New Test Cases** - Test new functionality
- [ ] **Update Test Fixtures** - Modify test data if needed
- [ ] **Verify All Tests Pass** - Run full test suite

#### Documentation Updates
- [ ] **Service Documentation** - Update `docs/services/your-service.md`
- [ ] **README Updates** - Update examples and usage in `README.md`
- [ ] **Claude Instructions** - Update `CLAUDE.md` with new endpoints/usage
- [ ] **API Documentation** - Update endpoint examples and schemas

#### Configuration Updates
- [ ] **Bentofile Updates** - Update dependencies and configuration
- [ ] **Environment Variables** - Add new environment variables if needed
- [ ] **Resource Requirements** - Update resource specifications if needed

#### Script Updates
- [ ] **Endpoint Examples** - Update `scripts/endpoint.sh` examples
- [ ] **Health Checks** - Update health check scripts if new endpoints
- [ ] **Test Scripts** - Update test automation if new test types

### 🚀 Deployment Verification

#### Pre-Deployment
- [ ] **All Tests Pass** - Verify comprehensive test suite passes
- [ ] **Build Succeeds** - Verify clean Bento build
- [ ] **Dependencies Resolved** - Ensure all dependencies install correctly
- [ ] **Configuration Valid** - Validate all configuration files

#### Post-Deployment
- [ ] **Service Starts** - Verify service starts without errors
- [ ] **Endpoints Accessible** - Test all endpoints respond correctly
- [ ] **Health Checks Pass** - Verify health monitoring works
- [ ] **Performance Acceptable** - Check response times and resource usage

## Quick Reference Commands

### Testing
```bash
# Test specific service
./scripts/test.sh --service your_service

# Test with coverage
./scripts/test.sh --coverage

# Run all tests including integration
./scripts/test.sh --all
```

### Building
```bash
# Build specific service
BENTOFILE=config/bentofiles/your-service.yaml ./scripts/run_bentoml.sh build services/your_service.py

# Build all services
./scripts/build_services.sh
```

### Running
```bash
# Run specific service
./scripts/run_bentoml.sh serve services.your_service:YourService

# Run multi-service
./scripts/start.sh
```

### Testing Endpoints
```bash
# Test endpoint
./scripts/endpoint.sh your_endpoint '{"input_field": "test"}'

# Health check
./scripts/health.sh

# Service info
./scripts/endpoint.sh info '{}'
```

## Common Mistakes to Avoid

### ❌ File Creation Mistakes
- Forgetting to update `build_services.sh`
- Missing test file creation
- Not updating `CLAUDE.md` with new endpoints
- Forgetting multi-service integration
- Missing Bentofile configuration

### ❌ Testing Mistakes  
- Not mocking heavy dependencies in unit tests
- Missing integration test for HTTP endpoints
- Not testing error scenarios
- Inadequate test coverage
- Slow tests not marked with `@pytest.mark.slow`

### ❌ Configuration Mistakes
- Missing dependencies in Bentofile
- Incorrect service path in configuration
- Resource requirements too low/high
- Missing environment variable documentation
- Incorrect Docker configuration

### ❌ Integration Mistakes
- Not updating multi-service endpoint count
- Missing endpoint examples in documentation
- Not updating health check information
- Missing script integration
- Inconsistent API patterns

Use this checklist to ensure complete and proper service integration with the project structure.