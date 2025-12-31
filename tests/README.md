# Testing Configuration

## Running Tests

### Install Test Dependencies

```bash
pip install pytest pytest-cov pytest-mock
```

### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_context.py

# Run specific test class
pytest tests/test_context.py::TestConversationMemory

# Run specific test
pytest tests/test_context.py::TestConversationMemory::test_create_conversation
```

### Run Tests by Marker

```bash
# Run only unit tests (skip integration)
pytest -m "not integration"

# Run only integration tests
pytest -m integration
```

## Test Structure

```
tests/
├── test_context.py              # Context management tests
├── test_query_enhancement.py    # Query enhancement tests
├── test_integration.py          # Integration component tests
└── test_integration_e2e.py      # End-to-end tests
```

## Test Coverage

### Unit Tests (Ready to Run)
- ✅ `test_context.py`: Conversation memory, buffer, window manager
- ✅ `test_query_enhancement.py`: Multi-query, HyDE, reranking, expansion
- ✅ `test_integration.py`: Cache, config, streaming events

### Integration Tests (Require Setup)
- ⏳ `test_integration_e2e.py`: Full pipeline tests (require vector store & LLM)

## Configuration

Create `pytest.ini` in project root:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    integration: Integration tests requiring external dependencies
    slow: Slow tests
addopts = -v --strict-markers
```

## Mock Data

For integration tests, mock vector stores and LLM clients:

```python
from unittest.mock import Mock

# Mock vector store
mock_vector_store = Mock()
mock_vector_store.similarity_search_with_score.return_value = [
    (Mock(page_content="RAG is..."), 0.9)
]

# Mock LLM client
mock_llm = Mock()
mock_llm.generate.return_value = "Generated response"
```

## Continuous Integration

Add to `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest --cov=src
```

## Test Best Practices

1. **Isolation**: Each test should be independent
2. **Mocking**: Mock external dependencies (LLM, vector store)
3. **Fixtures**: Use pytest fixtures for common setup
4. **Coverage**: Aim for >80% code coverage
5. **Speed**: Keep unit tests fast (<1s each)
6. **Integration**: Mark integration tests appropriately
