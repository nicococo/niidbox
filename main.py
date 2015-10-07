# import matplotlib
# import matplotlib.rcsetup as rcsetup
# print(rcsetup.all_backends)
# #
# matplotlib.use('MacOSX')
# # change to type 1 fonts!
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

import argparse, sys
from experiment_toy_seq import *
from gridmap import Job, process_jobs
import logging

def generate_param_set(set_name = 'full'):
    param_flx = [[1000, 0.001], [1000, 0.0001]]
    param_rr = [[0.1], [0.01], [0.001], [0.0001], [0.00001], [0.000001]]
    param_svr = [[0.1, 0.01], [0.1, 0.1], [1.0, .01], [1.0, 0.1], [10., .01], [10., .1], [100., .1], [100., .01]]
    param_krr = param_rr
    param_tr = list()
    for i in range(len(param_rr)):
        for j in range(len(param_rr)):
            for k in range(len(param_rr)):
                param_tr.append([param_rr[i][0], 100.*param_rr[j][0], 100.*param_rr[k][0]])

    param_tcrfr_indep = list()
    param_tcrfr = list()
    param_tcrfr_qp = list()
    param_tcrfr_pl = list()

    tcrfr_theta = [0.85]
    tcrfr_lambda = [0.000001]
    tcrfr_gamma = [100.0]
    tcrfr_k1 = [8, 20, 30]

    tcrfr_k1 = [30] # 5%
    tcrfr_k1 = [30] # 10%
    tcrfr_k1 = [15, 30]
    tcrfr_k1 = [4, 8] # 20%

    tcrfr_k2 = [4]  # 20%
    tcrfr_k2 = [18] # 5%
    tcrfr_k2 = [4] # 5%
    tcrfr_k2 = [4, 8] # 20%

    tcrfr_neighb = [10.]

    for i in range(len(tcrfr_theta)):
        for j in range(len(tcrfr_lambda)):
            for k in range(len(tcrfr_gamma)):
                param_tcrfr.append([tcrfr_theta[i], tcrfr_lambda[j], tcrfr_gamma[k]])
                for l in range(len(tcrfr_neighb)):
                    param_tcrfr_indep.append([tcrfr_theta[i], tcrfr_lambda[j], tcrfr_gamma[k], tcrfr_neighb[l]])
                for l in range(len(tcrfr_k1)):
                    param_tcrfr_pl.append([tcrfr_theta[i], tcrfr_lambda[j], tcrfr_gamma[k], tcrfr_k1[l]])
                for l in range(len(tcrfr_k2)):
                    param_tcrfr_qp.append([tcrfr_theta[i], tcrfr_lambda[j], tcrfr_gamma[k], tcrfr_k2[l]])

    params = []
    methods = []
    if set_name == 'full':
        params = [param_rr, param_svr, param_krr, param_tr, param_flx, param_tcrfr_indep, param_tcrfr]
        methods = [method_rr, method_svr, method_krr,
                   method_transductive_regression, method_flexmix,
                   method_tcrfr_indep, method_tcrfr]
    if 'tcrfr_qp' in set_name:
        methods.append(method_tcrfr_qp)
        params.append(param_tcrfr_qp)
    if 'tcrfr_pl' in set_name:
        methods.append(method_tcrfr_pl)
        params.append(param_tcrfr_pl)
    if 'tcrfr_indep' in set_name:
        methods.append(method_tcrfr_indep)
        params.append(param_tcrfr_indep)
    if 'rr' in set_name:
        methods.append(method_rr)
        params.append(param_rr)
    if 'lb' in set_name:
        methods.append(method_lb)
        params.append(param_rr)
    if 'svr' in set_name:
        methods.append(method_svr)
        params.append(param_svr)
    if 'krr' in set_name:
        methods.append(method_krr)
        params.append(param_krr)
    if 'tr' in set_name:
        methods.append(method_transductive_regression)
        params.append(param_tr)
    if 'flexmix' in set_name:
        methods.append(method_flexmix)
        params.append(param_flx)
    return methods, params


if __name__ == '__main__':
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +
                                '%(message)s'), level=logging.INFO)

    parser = argparse.ArgumentParser()
    # plot results arguments
    parser.add_argument("-a", "--plot_results", help="Show results plot (default=False).", default=False, type=bool)
    parser.add_argument("-b", "--results_filename", help="Set results filename (default='res_toy_[1, 2, 3, 4, 5, 6].npz').", default='res_toy_[1, 2, 3, 4, 5, 6].npz', type=str)
    # experiment arguments
    parser.add_argument("-s", "--states", help="List of states for testing (default=3).", default='1,2,3,4,5', type=str)
    parser.add_argument("-f", "--train_frac", help="Fraction of training exms (default=0.15)", default='0.1', type=str)
    parser.add_argument("-d", "--datapoints", help="Amount of data points (default=1000)", default=1000, type=int)
    parser.add_argument("-r", "--reps", help="Number of repetitions (default 10)", default=3, type=int)
    parser.add_argument("-m", "--method_set", help="Select active method set. (default 'full')", default='lb,rr,svr,flexmix,krr,tr', type=str)
    # grid computing arguments
    parser.add_argument("-p", "--processes", help="Number of processes (default 4)", default=1, type=int)
    parser.add_argument("-l", "--local", help="Run local or distribute? (default True)", default=True, type=bool)
    arguments = parser.parse_args(sys.argv[1:])
    print arguments

    # Plotting is done locally
    if arguments.plot_results:
        plot_results(arguments.results_filename)
        exit(0)

    # this is for generating a nice looking motivational example
    (vecX, vecy, vecz) = get_1d_toy_data(num_exms=arguments.datapoints, plot=False)

    # generate parameter sets
    methods, params = generate_param_set(arguments.method_set)

    MEASURES = 6
    REPS = arguments.reps
    states = np.array(arguments.states.split(','), dtype=np.int).tolist()
    train_fracs = np.array(arguments.train_frac.split(','), dtype=np.float).tolist()[0]
    print train_fracs
    mse = {}
    results = []
    if arguments.local:
        # This is necessary for using profiler
        print("Local computations.")
        for s in states:
            if s not in mse:
                mse[s] = np.zeros((REPS, MEASURES*len(methods)))
            for n in range(REPS):
                (names, res) = main_run(methods, params, vecX, vecy, vecz, train_fracs, 0.1, s, False)
                perf = mse[s]
                cnt = 0
                for p in range(MEASURES):
                    for m in range(len(methods)):
                        perf[n, cnt] = res[m][0][p]
                        cnt += 1
    else:
        jobs = []
        sn_map = {}
        cnt = 0
        print("Distribute computations.")
        for s in states:
            if s not in mse:
                mse[s] = np.zeros((REPS, MEASURES*len(methods)))
            for n in range(REPS):
                job = Job(main_run, [methods, params, vecX, vecy, vecz, train_fracs, 0.1, s, False],
                          mem_max='8G', mem_free='16G', name='TCRFR it({0}) state({1})'.format(n, s))
                jobs.append(job)
                sn_map[cnt] = (s, n)
                cnt += 1

            processedJobs = process_jobs(jobs, max_processes=arguments.processes, local=arguments.local >= 1)
            for (i, result) in enumerate(processedJobs):
                print "Job #", i
                (names, res) = result
                (s, n) = sn_map[i]
                perf = mse[s]
                cnt = 0
                for p in range(MEASURES):
                    for m in range(len(methods)):
                        perf[n, cnt] = res[m][0][p]
                        cnt += 1

    measure_names = res[0][1]
    means = np.zeros((len(states), MEASURES*len(methods)))
    stds = np.zeros((len(states), MEASURES*len(methods)))

    print '\n'
    print '================================================ FINAL RESULT'
    idx = 0
    for key in mse.iterkeys():
        means[idx, :] = np.mean(mse[key], axis=0)
        stds[idx, :] = np.std(mse[key], axis=0)

        print '------------------------------------------------'
        print 'States(',key, ')  Train_Frac(',train_fracs,')'
        m = means[idx, :].reshape((len(names), MEASURES), order='F')
        s = stds[idx, :].reshape((len(names), MEASURES), order='F')
        print ''.ljust(44), '', res[0][1]
        for i in range(len(names)):
            name = names[i].ljust(45)
            for j in range(MEASURES):
                name += '  {0:+3.4f}/{1:1.2f}'.format(m[i,j], s[i,j]).ljust(23)
            print name
        print '------------------------------------------------'
        idx += 1

    print '================================================ FINAL RESULT END'

    # save results
    np.savez('res_toy_{0}.npz'.format(states), MEASURES=MEASURES, methods=methods, params=params,
             means=means, stds=stds, states=states, measure_names=measure_names, names=names, train_fracs=train_fracs)

    # ..and stop
    print('Finish!')
