.PHONY: test smoke smoke-lite doctor smoke-synthid bootstrap-synthid docker-synthid-build docker-synthid-help \
	smoke-ctrlregen bootstrap-ctrlregen docker-ctrlregen-build docker-ctrlregen-help install-skill clean

SCRIPTS := skills/remove-ai-marks/scripts
PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

test:
	$(PYTHON) -m pytest

doctor:
	$(PYTHON) $(SCRIPTS)/doctor.py

smoke:
	$(PYTHON) $(SCRIPTS)/doctor.py --json >/dev/null
	-python3 $(SCRIPTS)/inspect_text.py tests/fixtures/sample_watermarked.txt
	python3 $(SCRIPTS)/clean_text.py tests/fixtures/sample_watermarked.txt -o /tmp/wm.cleaned.txt --stats
	python3 $(SCRIPTS)/rewrite_text.py tests/fixtures/sample_watermarked.txt --backend print-prompt >/dev/null
	-python3 $(SCRIPTS)/inspect_file.py tests/fixtures/sample_ai.md
	python3 $(SCRIPTS)/clean_file.py tests/fixtures/sample_ai.md -o /tmp/sample_ai.cleaned.md
	python3 $(SCRIPTS)/clean_file.py tests/fixtures/sample_ai.html -o /tmp/sample_ai.cleaned.html
	python3 $(SCRIPTS)/clean_file.py tests/fixtures/sample_meta.svg -o /tmp/sample_meta.cleaned.svg
	$(PYTHON) $(SCRIPTS)/clean_pixel_lite.py --help >/dev/null
	$(PYTHON) $(SCRIPTS)/detect_text_watermark.py --help >/dev/null
	@echo "smoke ok"

smoke-lite:
	$(PYTHON) -c "import numpy, PIL" 2>/dev/null && \
	  $(PYTHON) -c "import sys; sys.path.insert(0,'$(SCRIPTS)'); \
	    from PIL import Image; Image.new('RGB',(64,64),(120,90,60)).save('/tmp/wm_lite.png')" && \
	  $(PYTHON) $(SCRIPTS)/clean_pixel_lite.py /tmp/wm_lite.png -o /tmp/wm_lite.out.png \
	    --keep-psnr 0 --json >/dev/null && echo "smoke-lite ok" || \
	  echo "smoke-lite skipped (needs Pillow + numpy)"

smoke-synthid:
	@if [ -z "$(REVERSE_SYNTHID_DIR)" ]; then \
	  echo "smoke-synthid skipped (set REVERSE_SYNTHID_DIR)"; \
	else \
	  $(PYTHON) $(SCRIPTS)/score_synthid.py --help >/dev/null && echo "score_synthid adapter present"; \
	fi

bootstrap-synthid:
	./skills/remove-ai-marks/scripts/setup_synthid.sh

docker-synthid-build:
	docker build -f Dockerfile.synthid -t watermarks-remover-synthid-scorer .

docker-synthid-help:
	docker run --rm watermarks-remover-synthid-scorer --help

smoke-ctrlregen:
	@if [ -z "$(NOAI_WATERMARK_DIR)" ]; then \
	  echo "smoke-ctrlregen skipped (set NOAI_WATERMARK_DIR)"; \
	else \
	  $(PYTHON) $(SCRIPTS)/clean_ctrlregen.py --help >/dev/null && echo "clean_ctrlregen adapter present"; \
	fi

bootstrap-ctrlregen:
	./skills/remove-ai-marks/scripts/setup_ctrlregen.sh

docker-ctrlregen-build:
	docker build -f Dockerfile.ctrlregen -t watermarks-remover-ctrlregen .

docker-ctrlregen-help:
	docker run --rm watermarks-remover-ctrlregen --help

install-skill:
	mkdir -p $(HOME)/.grok/skills
	ln -sfn $(CURDIR)/skills/remove-ai-marks $(HOME)/.grok/skills/remove-ai-marks
	@echo "linked -> $(HOME)/.grok/skills/remove-ai-marks"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .venv
