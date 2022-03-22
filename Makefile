VENV = venv-sepia
ACTIVATE = . $(VENV)/bin/activate

serve-notebook:
	$(ACTIVATE) && jupyter-lab --port=8007 --no-browser

get-img:
	source .config && scp -r $${REMOTE_DIR}/examples/img/* examples/img