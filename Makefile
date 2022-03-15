VENV = venv-sepia
ACTIVATE = . $(VENV)/bin/activate

serve-notebook:
	$(ACTIVATE) && jupyter-lab --port=8004 --no-browser

get-img:
	source .config && scp -r $$(loc)/examples/img/* examples/img
