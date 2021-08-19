import os
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter

def run_notebook(notebook_filename='',execute_path='.',html=0):
    # open notebook
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
    # set up execution
    ep = ExecutePreprocessor(timeout=-1, kernel_name='python3')
    # run execution
    print('executing:',notebook_filename)
    ep.preprocess(nb, {'metadata': {'path': execute_path}})
    if html:
        # export to html
        html_exporter = HTMLExporter()
        html_exporter.exclude_input = True
        html_data, resources = html_exporter.from_notebook_node(nb)
        # write to output file
        outfile = os.path.splitext(notebook_filename)[0]+'.html'
        print('exporting:',outfile)
        with open(outfile, 'w') as f:
            f.write(html_data)

if __name__ == "__main__":
    if len(sys.argv)>1:
        notebook_filename = sys.argv[1]
    else:
        notebook_filename = ''
    if len(sys.argv)>2:
        execute_path = sys.argv[2]
    else:
        execute_path = '.'
    if len(sys.argv)>3:
        html = sys.argv[3]
    else:
        html = 0
    run_notebook(notebook_filename,execute_path,html)
