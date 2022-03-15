VENV = venv-sepia
ACTIVATE = . $(VENV)/bin/activate

serve-notebook:
	$(ACTIVATE) && jupyter-lab --port=8004 --no-browser
