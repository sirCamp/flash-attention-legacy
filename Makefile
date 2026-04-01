.PHONY: build install dev test bench clean

build:
	python setup.py build_ext --inplace

install:
	pip install .

dev:
	pip install -e ".[test,bench]"

test:
	pytest tests/ -v

bench:
	python benchmarks/benchmark.py

clean:
	rm -rf build/ dist/ *.egg-info flash_attn_legacy_cuda*.so
	find . -name "*.o" -delete
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
