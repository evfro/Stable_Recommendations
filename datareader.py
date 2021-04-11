import re
from collections import defaultdict
import numpy as np
import pandas as pd


userid = 'userId'
algs = ['PureSVD', 'PSI']
alg_files = {'PureSVD': 'SVD', 'PSI': 'PSI', 'MP': 'MPRec', 'RND': 'RRec'}

metrics = ['HR', 'MRR', 'Coverage']
metric_files = {'HR': 'StepHirate', 'MRR': 'StepMRR', 'Coverage': 'COVRatio'}
display_ranks = range(10, 100, 10)


def read_metric(alg, metric, folder, label):
    file = f'{folder}/{label}_{alg_files[alg]}_{metric_files[metric]}.csv'
    return pd.read_csv(file).set_index('Rank').filter(regex=r'Step_\d+$')

def read_stability(folder, label, alg):
    file = f'{folder}/{label}_{alg_files[alg]}_ALLUsersCORR'
    try:
        return np.load(f'{file}.npy')
    except FileNotFoundError:
        return np.load(f'{file}.npz')['arr_0']
        

def read_data(folder, label):
    data = defaultdict(dict)
    scores = defaultdict(dict)
    
    try:
        test_users = pd.read_csv(f'{folder}/{label}_Userlist.gz', index_col=[0, 1])
    except FileNotFoundError:
        test_users = pd.read_csv(f'{folder}/{label}_User_list.gz', index_col=[0, 1])
    
    for alg in algs:
        for metric in metrics:
            data[alg][metric] = read_metric(alg, metric, folder, label)
            
    for metric in metrics:
        scores[metric]['wide'] = (
            pd.concat({alg:data[alg][metric] for alg in algs}, axis=1)
            .stack(level=1)
            .rename_axis(index=['rank', 'step'], columns=['model'])
        )
        scores[metric]['long'] = scores[metric]['wide'].stack().to_frame(metric).reset_index()
        
        
    for alg in algs:
        data[alg]['Stability'] = read_stability(folder, label, alg)
        
    n_steps = data[algs[0]]['Stability'].shape[2]
    step_stab = defaultdict(list)
    for i in range(n_steps):
        step_users = test_users.loc[f'step_{i+1}', userid].values
        for alg in algs:
            step_stab[alg].append(
                pd.DataFrame(
                    data[alg]['Stability'][:, step_users, i],
                    index=data[alg][metrics[0]].index,
                    columns=step_users
                ).stack()
            )

    for alg in algs:
        data[alg]['Stability_df'] = (
            pd.concat(step_stab[alg], keys=[f'step_{i+1}' for i in range(n_steps)])
            .sort_index()
            .reset_index()
            .rename(columns={'level_0': 'step', 'Rank': 'rank', 'level_2': userid, 0: 'Stability'})
        )
        data[alg]['Stability_avg_df'] = (
            data[alg]['Stability_df'].groupby(['step', 'rank'])[['Stability']].mean().reset_index()
        )
    return data, scores, test_users


def increase_step(key):
    '''Function to align index in standard metrics data with stability measurments'''
    step = re.match(re.compile('(Step)_(\d+)$', re.IGNORECASE), key)
    if step:
        key, num = step.groups()
    else:
        raise ValueError('"Step" key is expected')
    return f'{key.capitalize()}_{int(num)+1}'

def combine_all_metrics(scores, data):
    # average metrics across all test users
    _metrics = [
        scores[metric]['wide']
        .sort_index()
        .loc[pd.IndexSlice[:, 'Step_2':], :] # no stability measures at step 1
        .stack('model')
        for metric in metrics
    ]
        
    # average stability across all test users
    stb = pd.concat(
        {
            alg: data[alg]['Stability_df'].groupby(['rank', 'step'])['Stability'].mean()
            for alg in algs
        },
        axis=1
    ).rename(mapper=increase_step, axis='index', level='step').rename_axis('model', axis=1).stack('model')

    return pd.concat([stb]+_metrics, keys=['Stability']+metrics, axis=1).reset_index()