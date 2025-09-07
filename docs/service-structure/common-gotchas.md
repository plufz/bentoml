# Common Gotchas - Quick Reference for Claude

This is a quick checklist of the most critical things to remember when creating or editing BentoML services.

## Essential Updates Required

When creating or modifying a service, **always** update these files:

### Scripts (Critical)
- `scripts/endpoint.sh` - Add endpoint examples
- `scripts/test.sh` - Add service test option
- `scripts/build_services.sh` - Add service build command
- `scripts/start.sh` - Update if multi-service integration

### Documentation (Critical)
- `README.md` - Add service to main documentation
- `CLAUDE.md` - Update endpoint examples and service list
- `docs/services/` - Create or update service-specific documentation

### Multi-Service Integration
- `services/multi_service.py` - Add service integration
- `config/bentofiles/multi-service.yaml` - Add dependencies

## Quick Verification

After service changes, verify:
- [ ] Service builds: `./scripts/build_services.sh`
- [ ] Tests pass: `./scripts/test.sh --service your_service`  
- [ ] Endpoints work: `./scripts/endpoint.sh your_endpoint '{}'`
- [ ] Documentation updated with new service/endpoints

**Missing any of these updates will break the development workflow or leave documentation inconsistent.**