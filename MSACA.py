# The Implementation of Multi-strategy Adaptive Cultural Algorithm
import heapq
import json
import os
import random
import time
import copy
from tqdm import tqdm
from fast_domination_sort import *
from crowded_distance import calculate_crowding_distance
from global_search import global_search2
from json_util import NoIndentEncoder, NoIndent
from local_search import local_search2, roulette
from read_instance import read_instance
from decode1 import INEH, cal_obj, decode
from concurrent.futures import ProcessPoolExecutor
from init1 import NEH_deduct
from collections import deque
from NSGA3 import reference_points, environmental_selection

# A Kernel Function for Prallel Population Initialization
def init2(input_table):
    [i, dno, filter_no, X_par1, X_par2, X, Y, process, mach_ava, job_release, layer_stage, job_num, mach_lists, prd_pt, weight, prd, release, sum_op] = input_table
    if dno == 1:
        jobs = list(range(job_num))
        if filter_no == 0:
            jobs.remove(X_par1[0])
            jobs.remove(X_par1[1])
            random.shuffle(jobs)
            X[:] = X_par1 + jobs
        elif filter_no == 1:
            jobs.remove(X_par2[0])
            jobs.remove(X_par2[1])
            random.shuffle(jobs)
            X[:] = X_par2 + jobs
        else:
            if i % 2 == 0:
                jobs.remove(X_par1[0])
                jobs.remove(X_par1[1])
                random.shuffle(jobs)
                X[:] = X_par1 + jobs
            else:
                jobs.remove(X_par2[0])
                jobs.remove(X_par2[1])
                random.shuffle(jobs)
                X[:] = X_par2 + jobs
    else:
        X[:] = np.random.permutation(job_num).tolist()

    # Decoding
    decode(X, dno, process, mach_ava, job_release, layer_stage, job_num, mach_lists, prd_pt, prd, release, weight)

    # Calculate Objectives
    cal_obj(Y, process, mach_ava, job_release, weight, sum_op, mach_lists)

    return [X, Y, process, mach_ava, job_release, dno]


if __name__ == '__main__':
    Np = 200  # Population Size Np
    LP = 20  # Memory Size Lp
    for e, code in enumerate(['B']):  # ['A', 'B', 'C', 'D'], range(1,6)
        inst_d = 'data.json'
        inst_k = f'2-30-{code}'
        rr = 10
        # inst_d = 'small_data.json'  # Small-Scale
        # inst_k = f'2-20-A-{code}'
        print(inst_k)
        run_doc = {}
        for run in tqdm(range(rr)):
            run_doc[run] = {}
            V = reference_points(Np, 5)  # reference vectors
            decodes = [j for j in range(4) for k in range(Np//4)]

            '''Read the Instance'''
            layer_stage, job_num, mach_lists, prd_pt, weight, prd, release, sum_op = read_instance(inst_d, inst_k)
            terminal = job_num * 15

            '''Initialize the Empty List Structure Required for Recording Machine Processing Sequences'''
            process = [[] for i in range(sum(mach_lists))]
            mach_ava = np.zeros(sum(mach_lists)).tolist()
            job_release = np.zeros(job_num).tolist()
            obj_list = np.zeros(5, int).tolist()
            job_codes = np.zeros(job_num, int).tolist()

            probs = [0.25] * 4  # Success Failure Memory Probability
            succ_queue = deque(maxlen=LP)
            fail_queue = deque(maxlen=LP)
            succ_ele = [0] * 4
            fail_ele = [0] * 4

            begin = time.time()
            '''Multi-Strategy Initialization Population(init1)'''
            X1_par, X2_par = INEH(layer_stage, job_num, prd_pt, prd, weight)
            input_pattern1 = [X1_par, obj_list, process, mach_ava, job_release, layer_stage, 2, mach_lists, prd_pt, weight, prd, release, (sum_op*2)/job_num]
            input_pattern2 = copy.deepcopy(input_pattern1)
            input_pattern2[0] = X2_par
            input_patterns = [input_pattern1, input_pattern2]

            '''Parallel Deduction of Two NEH Operation'''
            with ProcessPoolExecutor() as neh_pool:
                output_patterns = neh_pool.map(NEH_deduct, input_patterns)
            output_patterns = list(output_patterns)
            filter_res = filter_domination_set(output_patterns[0], output_patterns[1])
            if filter_res == [1]:
                filter_no = 0
            elif filter_res == [2]:
                filter_no = 1
            else:
                filter_no = 2
            input_table = [[i, decodes[i], filter_no, X1_par, X2_par, job_codes, obj_list, process, mach_ava, job_release, layer_stage, job_num, mach_lists, prd_pt, weight, prd, release, sum_op] for i in range(Np)]
            with ProcessPoolExecutor() as pool:
                output_table = pool.map(init2, input_table)
            output_table = copy.deepcopy(list(output_table))
            del input_table
            Y = np.array([output_table[i][1] for i in range(Np)])
            z_ = Y.min(axis=0)
            a_ = Y.max(axis=0)

            '''Fast Non Dominated Sorting'''
            fronts, ranks = fast_non_domination_sort(Y.tolist())

            '''Establish Elite Archive'''
            archive = []
            archive.extend(output_table)

            iter1 = 0
            while time.time()-begin <= terminal:
                X = np.array([archive[i][0] for i in range(Np)])
                Y = np.array([archive[i][1] for i in range(Np)])
                if iter1 > 0:
                    decodes = [archive[i][-1] for i in range(Np)]

                '''Normalize Objective Values'''
                norm_objs = (Y - z_) / (a_ - z_)

                '''Calculate Crowding Distance'''
                crowded_distance = calculate_crowding_distance([norm_objs.tolist(), fronts[0]])
                border_list = heapq.nlargest(5, crowded_distance, key=crowded_distance.get)

                '''Global Search based on Multi-Level Group Interaction'''
                input_global = [[i, archive[i][-1], decodes, border_list, X, fronts, ranks, process, mach_ava, job_release, layer_stage, job_num, mach_lists, prd_pt, weight, prd, release, sum_op] for i in range(Np)]
                with ProcessPoolExecutor() as g_pool:
                    output_global = g_pool.map(global_search2, input_global)
                output_global = list(output_global)
                archive.extend(output_global)
                # print(mach_ava)

                '''Local Search Based on Success Failure Memory'''
                LSO = [roulette(probs) for i in range(Np)]  # Roulette Wheel Selection
                input_local = [[archive[i][-1], output_global[i][0], LSO[i], process, mach_ava, job_release, layer_stage, job_num, mach_lists, prd_pt, weight, prd, release, sum_op] for i in range(Np)]
                with ProcessPoolExecutor() as l_pool:
                    output_local = l_pool.map(local_search2, input_local)
                output_local = list(output_local)
                archive.extend(output_local)

                '''Count the Number of Successes and Failures, Update Memory'''
                count_res = [count_filter(output_local[i][1], output_global[i][1]) for i in range(Np)]
                succ_ele1 = copy.deepcopy(succ_ele)
                fail_ele1 = copy.deepcopy(fail_ele)
                for i in range(Np):
                    if count_res[i]:
                        succ_ele1[LSO[i]] += 1
                    else:
                        fail_ele1[LSO[i]] += 1
                succ_queue.append(succ_ele1)
                fail_queue.append(fail_ele1)
                SR = np.array([sum([j[i] for j in succ_queue]) for i in range(4)])
                FR = np.array([sum([j[i] for j in fail_queue]) for i in range(4)])
                new_probs = SR/(SR+FR)
                # Update VNS Operator Selection Probability
                probs = (new_probs / new_probs.sum()).tolist()
                # print(probs)

                '''Niche Selection'''
                X = np.array([archive[i][0] for i in range(len(archive))])
                Y = np.array([archive[i][1] for i in range(len(archive))])
                z_ = Y.min(axis=0)  # 参考点
                a = Y.max(axis=0)
                a_ = np.vstack([a_, a]).max(axis=0)
                norm_Y = (Y - z_) / (a_ - z_)  # 归一化
                z_min = np.zeros(5)
                selected, new_ranks = environmental_selection(3*Np, norm_Y, z_min, Np, V)
                new_archive = []
                ranks = {}
                fronts = {}
                be = 0
                for f in range(max(list(new_ranks.values()))):
                    fronts[f] = []
                for i in range(len(archive)):
                    if selected[i]:
                        new_archive.append(archive[i])
                        fronts[new_ranks[i]].append(be)
                        ranks[be] = new_ranks[i]
                        be += 1
                iter1 += 1
                # print(iter1)
                archive = copy.deepcopy(new_archive)
                # P = get_ps(np.array([archive[i][1] for i in range(Np)]))
                # v_max = P.max(axis=0)
                # v_min = P.min(axis=0)
                # P = ((P - np.tile(v_min, (P.shape[0], 1))) / (np.tile(v_max, (P.shape[0], 1)) - np.tile(v_min, (P.shape[0], 1))))
                # print('Noc', v_min, 'HV', hv.compute(P))  # Strategy Number
                # print([archive[i][-1] for i in range(len(archive))])
                del new_archive
            res_X = [archive[i][0] for i in range(Np) if ranks[i] == 0]
            # Verify the job code
            for res in res_X:
                if sum(res) != sum(list(range(job_num))):
                    raise Exception('ERROR')
            objs = [archive[i][1] for i in range(Np) if ranks[i] == 0]
            run_doc[run]['pareto'] = NoIndent(np.unique(np.array(objs), axis=0).tolist())
            dir1 = f'Outputs/MSACA/'
            file = f'{dir1}/{inst_k}_16p.json'
            if not os.path.exists(dir1):
                os.makedirs(dir1)
            with open(file, 'w') as f:
                obj = json.dumps(run_doc, indent=1, cls=NoIndentEncoder)
                f.write(obj)
            print(f"Stop_checkpoint: {time.time()}")
