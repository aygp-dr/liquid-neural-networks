# Liquid Neural Networks - Makefile
# Org-mode tangle/detangle operations

.PHONY: help tangle detangle clean-tangled setup-tangle all

# Default target
all: help

# Help target
help:
	@echo "Liquid Neural Networks - Makefile Commands"
	@echo "========================================="
	@echo "  make tangle        - Tangle all code blocks from SETUP.org"
	@echo "  make detangle      - Extract code back to SETUP.org (updates)"
	@echo "  make clean-tangled - Remove all tangled files"
	@echo "  make setup-tangle  - Tangle and create directory structure"
	@echo "  make help          - Show this help message"

# Tangle SETUP.org to extract all code blocks
tangle:
	@echo "Tangling SETUP.org..."
	@emacs --batch --eval "(require 'org)" \
		--eval "(setq org-babel-tangle-comment-format-beg \"\")" \
		--eval "(setq org-babel-tangle-comment-format-end \"\")" \
		--eval "(org-babel-tangle-file \"SETUP.org\")"
	@echo "✓ Tangling complete"

# Tangle README.org separately
tangle-readme:
	@echo "Tangling README.org..."
	@emacs --batch --eval "(require 'org)" \
		--eval "(setq org-babel-tangle-comment-format-beg \"\")" \
		--eval "(setq org-babel-tangle-comment-format-end \"\")" \
		--eval "(org-babel-tangle-file \"README.org\")"
	@echo "✓ README tangling complete"

# Detangle - extract code changes back to SETUP.org
detangle:
	@echo "Detangling code back to SETUP.org..."
	@emacs --batch --eval "(require 'org)" \
		--eval "(setq org-src-preserve-indentation t)" \
		--eval "(org-babel-detangle \"SETUP.org\")"
	@echo "✓ Detangling complete"

# Clean all tangled files
clean-tangled:
	@echo "Cleaning tangled files..."
	@rm -rf src/liquid_neural_networks/core.clj
	@rm -rf src/python/lnn_research.py
	@rm -rf scripts/setup.sh
	@rm -rf docker/Dockerfile
	@rm -rf scripts/create_structure.sh
	@rm -rf scripts/freebsd_setup.sh
	@echo "✓ Cleaned tangled files"

# Setup and tangle
setup-tangle: tangle
	@echo "Running setup scripts..."
	@[ -f scripts/create_structure.sh ] && sh scripts/create_structure.sh || true
	@echo "✓ Setup complete"

# Watch for changes and auto-tangle (requires inotify-tools or fswatch)
watch:
	@echo "Watching SETUP.org for changes..."
	@if command -v fswatch >/dev/null 2>&1; then \
		fswatch -o SETUP.org | xargs -n1 -I{} make tangle; \
	else \
		echo "Please install fswatch: pkg install fswatch"; \
		exit 1; \
	fi

# Verify org-mode is available
check-org:
	@emacs --batch --eval "(require 'org)" --eval "(princ (org-version))" || \
		(echo "Error: Emacs org-mode not found" && exit 1)

# Generate README.md from README.org
README.md: README.org
	@echo "Generating README.md from README.org..."
	@emacs --batch --eval "(require 'org)" \
		--eval "(require 'ox-md)" \
		--eval "(with-current-buffer (find-file-noselect \"$<\") (org-md-export-to-markdown))"
	@echo "✓ README.md generated"