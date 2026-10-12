"""Execute selected notebooks in the current Python environment without socket startup."""
from pathlib import Path
import nbformat
import json, importlib.metadata
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display

def execute(path):
    from .io import settings,destination
    cfg=settings();status_path=destination(cfg)/'run_status.json'
    state=json.loads(status_path.read_text()) if status_path.exists() else {'mode':cfg['mode'],'software':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','lifelines','matplotlib','h5py']},'tasks':{}}
    state['tasks'][Path(path).name]={'status':'RUNNING'};status_path.write_text(json.dumps(state,indent=2))
    notebook=nbformat.read(Path(path),as_version=4)
    namespace={'__name__':'__main__','display':display}
    shell=InteractiveShell.instance()
    for index,cell in enumerate(notebook.cells):
        if cell.cell_type!='code':continue
        try:exec(compile(shell.transform_cell(cell.source),f'{Path(path).name}:cell{index}','exec'),namespace)
        except Exception as exc:
            state['tasks'][Path(path).name]={'status':'FAILED','cell':index,'reason':f'{type(exc).__name__}: {exc}'}
            status_path.write_text(json.dumps(state,indent=2))
            raise RuntimeError(f'{Path(path).name}, cell {index}: {type(exc).__name__}: {exc}') from exc
    state['tasks'][Path(path).name]={'status':'COMPLETE'};status_path.write_text(json.dumps(state,indent=2))
    return namespace
