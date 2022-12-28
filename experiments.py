from cvgutils.experimentManager import execute_experiments
import os
from arguments_deepfnf import parse_arguments_deepfnf
local = False
experiments_fn = os.path.join(os.getcwd(),"experiments_params.json")
if local:
    testsettings_fn = os.path.join(os.getcwd(),"testsettings_local.json")    
    method_py = os.path.join(os.getcwd(),'train.py')
else:
    testsettings_fn = os.path.join(os.getcwd(),"testsettings_server.json")
    method_py = '/mshvol2/users/mohammad/optimization/deepfnf_fork/train.py'
# test_keys=["0","1","2","3","4"]
test_keys=["0"]
exp_keys=["0"]
methods={"deepfnf":{"arg_parser":parse_arguments_deepfnf}}
execute_experiments(methods, testsettings_fn, experiments_fn, method_py, local=local, exp_keys=exp_keys,test_keys=test_keys)