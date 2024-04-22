from neuralhydrology.nh_run import start_run
from pathlib import Path
for basin in ['01333000']:
    with open("examples/07-Superflex/1_basin_list.txt",'w') as f:
        f.write(basin)
    f1 =  open("examples/07-Superflex/experiment.yml", 'w')
    f2 =  open("examples/07-Superflex/example4.yml", 'r')
    l = f2.readline()
    while l[:7] != 'run_dir':
        f1.write(l)
        l = f2.readline()
    f1.write('run_dir: runs/' + basin + '\n')
    l = f2.readline()
    while l:
        f1.write(l)
        l = f2.readline()
    f1.close()
    f2.close()
    for _ in range(5):
        start_run(config_file=Path("examples/07-Superflex/experiment.yml", gpu=-1))