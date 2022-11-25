from cvgutils.experimentManager import execute_experiments
import os
from arguments import parse_arguments_deepfnf
local = False
experiments_fn = os.path.join(os.getcwd(),"experiments_params.json")
if local:
    testsettings_fn = os.path.join(os.getcwd(),"testsettings_local.json")    
    method_py = os.path.join(os.getcwd(),'train.py')
else:
    testsettings_fn = os.path.join(os.getcwd(),"testsettings_server.json")
    method_py = '/mshvol2/users/mohammad/optimization/deepfnf_fork/train.py'

execute_experiments(parse_arguments_deepfnf, testsettings_fn, experiments_fn, method_py, local,max_count=2)