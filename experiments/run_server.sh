#!/bin/bash
ngpus=1
ncpus=1
meml=30G
memr=26G

server_path=/mshvol2/users/mohammad/optimization/deepfnf_fork
cmd1="python deploy --image docker.io/mohammadsh/deepfnf:latest --priority $3 --key kub --name $2 --ngpus $ngpus --cpu $ncpus --meml $meml --memr $memr --type deeplearning"
cmd2="$server_path/experiments/run.sh $1"
echo $cmd1 "$cmd2"

cd /home/mohammad/cvgutils/cluster_control/deployments/
$cmd1 "$cmd2"