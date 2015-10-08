# import matplotlib
# # import matplotlib.rcsetup as rcsetup
# # print(rcsetup.all_backends)
# matplotlib.use('MacOSX')
# # # change to type 1 fonts!
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

import argparse, sys
from experiment_toy_seq import *
from gridmap import Job, process_jobs
import logging


if __name__ == '__main__':
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +
                                '%(message)s'), level=logging.INFO)

    parser = argparse.ArgumentParser()
    # plot results arguments
    parser.add_argument("-a", "--plot_results", help="Show results plot (default=False).", default=False, type=bool)
    parser.add_argument("-b", "--results_filename", help="Set results filename (default='').", default='res_toy_frac_[0.02, 0.05, 0.1, 0.15].npz', type=str)
    # experiment arguments
    parser.add_argument("-s", "--states", help="List of states for testing (default=3).", default='3', type=str)
    parser.add_argument("-f", "--train_frac", help="Fraction of training exms (default=0.15)", default='0.05,0.075,0.1,0.125,0.15', type=str)
    parser.add_argument("-d", "--datapoints", help="Amount of data points (default=1000)", default=1000, type=int)
    parser.add_argument("-r", "--reps", help="Number of repetitions (default 10)", default=20, type=int)
    parser.add_argument("-m", "--method_set", help="Select active method set. (default 'full')", default='lb,rr,svr,flexmix,krr,tr,tcrfr_pl,tcrfr_qp', type=str)
    # grid computing arguments
    parser.add_argument("-p", "--processes", help="Number of processes (default 4)", default=4, type=int)
    parser.add_argument("-g", "--gridmap", help="Use gridmap? (default True)", default=True, type=bool)
    parser.add_argument("-l", "--local", help="Run local? (default True)", default=False, type=bool)
    arguments = parser.parse_args(sys.argv[1:])
    print arguments

    # Plotting is done locally
    if arguments.plot_results:
        plot_frac_results(arguments.results_filename)
        exit(0)

    # this is for generating a nice looking motivational example
    (vecX, vecy, vecz) = get_1d_toy_data(num_exms=arguments.datapoints, plot=False)

    # generate parameter sets
    methods, params = generate_param_set(arguments.method_set)

    MEASURES = 7
    REPS = arguments.reps
    states = np.array(arguments.states.split(','), dtype=np.int).tolist()[0]
    train_fracs = np.array(arguments.train_frac.split(','), dtype=np.float).tolist()
    mse = {}
    results = []
    if arguments.gridmap:
        # This is necessary for using profiler
        print("Local computations.")
        for train_frac in train_fracs:
            if train_frac not in mse:
                mse[train_frac] = np.zeros((REPS, MEASURES*len(methods)))
            for n in range(REPS):
                (names, res) = main_run(methods, params, vecX, vecy, vecz, train_frac, 0.1, states, False)
                perf = mse[train_frac]
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
        for train_frac in train_fracs:
            if train_frac not in mse:
                mse[train_frac] = np.zeros((REPS, MEASURES*len(methods)))
            for n in range(REPS):
                job = Job(main_run, [methods, params, vecX, vecy, vecz, train_frac, 0.1, states, False],
                          mem_max='1G', mem_free='1G', name='TCRFR it({0}) frac({1})'.format(n, train_frac))
                jobs.append(job)
                sn_map[cnt] = (train_frac, n)
                cnt += 1
        processedJobs = process_jobs(jobs, max_processes=arguments.processes, local=arguments.local)
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
    means = np.zeros((len(train_fracs), MEASURES*len(methods)))
    stds = np.zeros((len(train_fracs), MEASURES*len(methods)))

    print '\n'
    print '================================================ FINAL RESULT'
    idx = 0
    for key in mse.iterkeys():
        means[idx, :] = np.mean(mse[key], axis=0)
        stds[idx, :] = np.std(mse[key], axis=0)

        print '------------------------------------------------'
        print 'Train_Frac(',key, ')  States(',states,')'
        m = means[idx, :].reshape((len(names), MEASURES), order='F')
        s = stds[idx, :].reshape((len(names), MEASURES), order='F')
        print ''.ljust(44), '', res[0][1]
        for i in range(len(names)):
            name = names[i].ljust(45)
            for j in range(MEASURES):
                name += ' {0:+3.4f}/{1:1.2f}'.format(m[i,j], s[i,j]).ljust(23)
            print name
        print '------------------------------------------------'
        idx += 1

    print '================================================ FINAL RESULT END'

    # save results
    np.savez('res_toy_frac_{0}.npz'.format(train_fracs), MEASURES=MEASURES, methods=methods, params=params,
             means=means, stds=stds, states=states, measure_names=measure_names, names=names, train_fracs=train_fracs)

    # ..and stop
    print('Finish!')
