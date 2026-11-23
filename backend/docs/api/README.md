# API Documentation

Status: **NOT YET IMPLEMENTED.** There is no HTTP server, no route, and no
schema in this project yet.
[backend/interfaces/api/](../../backend/interfaces/api/) contains empty packages
only.

## What this directory will contain

| Document | Purpose |
| --- | --- |
| `overview.md` | Purpose and scope of the service, versioning policy |
| `endpoints.md` | Endpoint reference: method, path, request/response schema, status codes |
| `schemas.md` | Request/response models and validation rules |
| `authentication.md` | Auth scheme and how credentials are issued and rotated |
| `errors.md` | Error taxonomy and response envelope |
| `openapi.json` | Generated specification — produced by the app, never hand-edited |

## Planned surface

**PLANNED — none of these exist.** Sketched here so the eventual design is
reviewable before it is built:

| Area | Intent |
| --- | --- |
| Health | Liveness and readiness |
| Dataset | Report registry state and validation status |
| Inference | Score a transaction (or batch) against a trained model |
| Explainability | Return SHAP attributions for a scored transaction |
| Experiments | List runs and retrieve their metrics |

## Rules

- Documentation is written after an endpoint works, not before.
- `openapi.json` is generated output; it is never edited by hand.
- No endpoint is documented here until it is reachable and covered by a test.
