"""
from pathlib import Path

from neuralhydrology.nh_run import start_run

start_run(config_file=Path("examples/07-Superflex/fabrizio.yml", gpu=-1))
"""
from pathlib import Path
import matplotlib.pyplot as plt

run_dir = "../../runs/dev/run_size_1"
#eval_run(run_dir=run_dir, period="test")

import pickle

qsim = []
for i in range(10,51,10):
    with open(run_dir + "/validation/model_epoch0"+str(i)+"/validation_results.p", "rb") as fp:
        results = pickle.load(fp)
        qsim.append(results['01510000']['1D']['xr']['QObs(mm/d)_sim'])
        if i == 10:
            qobs = results['01510000']['1D']['xr']['QObs(mm/d)_obs']

plt.plot(qobs, label='observed')
for i in range(1,6,2):
    plt.plot(qsim[i-1],label="epoch "+str(i*10), alpha = 0.15*i)
plt.legend()
plt.show()

import sys
sys.path.append('..')  # Add the parent folder to the sys.path
sys.path.append('../..')  # Add the parent of the parent folder to the sys.path
from neuralhydrology.evaluation import metrics

models = {}
for j in [1,4,10]:
    metric_values = []
    run_dir = "../../runs/dev/run_size_"+str(j)
    qsim = []
    for i in range(10,51,10):
        with open(run_dir + "/validation/model_epoch0"+str(i)+"/validation_results.p", "rb") as fp:
            results = pickle.load(fp)
            qsim.append(results['01510000']['1D']['xr']['QObs(mm/d)_sim'])

    for i in range(1,6):
        metric_values.append(metrics.calculate_all_metrics(qobs.isel(time_step=-1),
                                                        qsim[i-1].isel(time_step=-1)))
    
    models[j] = metric_values

fig, axs = plt.subplots(3, 1+len(metric_values[0])//3, sharex=True)
colors = [None,'r',None,None,'g',None,None,None,None,None,'b']
for j in [1,4,10]:
    metric_values = models[j]
    for ik, k in enumerate(list(metric_values[0].keys())):
        y = [i[k] for i in metric_values]
        axs[ik%3][ik//3].plot(list(range(10,51,10)),y,c = colors[j],label=str(j)+' routing reservoirs')
        if j == 10:
            axs[ik%3][ik//3].set_title(k)
            axs[ik%3][ik//3].legend()
plt.show()