#/bin/bash

post=$(date +%s)

nohup wq sub -r "mode:bycore;N:${1};group:[gen6]; hostfile: auto;job_name: Nls;priority:med" -c "source ~/.bash_profile ; source ~/.bashrc  ; cd ~/repos/alhazen ; mpirun -hostfile %hostfile% python tests/testRecon.py $1" > output_$post.log &

