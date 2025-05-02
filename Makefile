# Makefile

.PHONY: setup install

setup:
	@echo "Setting PYTHONPATH..."
	@echo "export PYTHONPATH=$(PWD):\$$PYTHONPATH" >> ~/.bashrc
	@echo "export PYTHONPATH=$(PWD):\$$PYTHONPATH" >> ~/.zshrc
	@echo "PYTHONPATH updated. Run 'source ~/.bashrc' or 'source ~/.zshrc' to apply changes."

install:
	@echo "Installing ICESEE as a package..."
	pip install -e .
	@echo "ICESEE installed. You can now use 'from ICESEE.core.utilities' anywhere."
