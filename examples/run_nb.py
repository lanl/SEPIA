import os
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter
import glob

def export_to_html(nb,file):
        html_exporter = HTMLExporter()
        html_exporter.exclude_input = True
        html_data, resources = html_exporter.from_notebook_node(nb)
        # write to output file
        outfile = os.path.splitext(file)[0]+'.html'
        print('exporting:',outfile)
        with open(outfile, 'w') as f:
            f.write(html_data)
            
# search basedir for .ipynb files recursively and run all
def run_all_notebooks(basedir='.',html=0):
    # open notebook
    nb_filenames = glob.glob(basedir + '/**/*.ipynb', recursive=True)
    print('notebooks found:')
    for file_name in nb_filenames:
        print(file_name)
    
    print('executing notebooks:')
    for file_name in nb_filenames:
        execute_path = os.path.split(file_name)[0] # get execute directory
                
        with open(file_name) as f:
            nb = nbformat.read(f, as_version=4)
        # set up execution
        ep = ExecutePreprocessor(timeout=-1, kernel_name='python3')
        # run execution
        print(file_name)
        ep.preprocess(nb, {'metadata': {'path': execute_path}})
        if html:
            export_to_html(nb,file_name)
     
def run_notebook(file_name='', basedir='.', html=0):
    """
    Execute a notebook, or all notebooks in a directory, and optionally export to html

    :var string file_name: file name of notebook to be executed
    :var string basedir: base directory, or location to execute notebook. Imports may fail if execute path is improperly set
    :var bool html: flag for html export
    """
    if file_name == 'all':
        run_all_notebooks(basedir,html)
    else:
        with open(file_name) as f:
                nb = nbformat.read(f, as_version=4)
        # set up execution
        ep = ExecutePreprocessor(timeout=-1, kernel_name='python3')
        # run execution
        print('executing:',file_name)
        ep.preprocess(nb, {'metadata': {'path': basedir}})
        if html:
            export_to_html(nb,file_name)

if __name__ == "__main__":
    if len(sys.argv)>1:
       	notebook_filename = sys.argv[1]
    else:
        notebook_filename = 'all'
    if len(sys.argv)>2:
        basedir = sys.argv[2]
    else:
        basedir = '.'
    if len(sys.argv)>3:
        html = sys.argv[3]
    else:
        html = 0
    run_notebook(notebook_filename,basedir,html)
