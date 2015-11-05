__author__ = 'nicococo'
import argparse, sys
import logging

from experiment_toy_seq import *

if __name__ == '__main__':
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +
                                '%(message)s'), level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--states", help="List of states for testing (default=3).", default='3', type=int)
    parser.add_argument("-f", "--train_frac", help="Fraction of training exms (default=0.15)", default='0.5', type=float)
    parser.add_argument("-d", "--datapoints", help="Amount of data points (default=1000)", default='500', type=int)
    parser.add_argument("-r", "--reps", help="Number of repetitions (default 10)", default=1, type=int)
    parser.add_argument("-m", "--method_set", help="Select active method set. (default 'full')", default='lb,rr,tcrfr_pl,tcrfr_qp', type=str)
    arguments = parser.parse_args(sys.argv[1:])
    print arguments

    # generate parameter sets
    methods, params = generate_param_set(arguments.method_set, 'data')

    MEASURES = 7
    REPS = arguments.reps

    datapoints = arguments.datapoints
    states = arguments.states
    train_fracs = arguments.train_frac

    mse = {}
    results = []
    (vecX, vecy, vecz) = get_1d_toy_data(num_exms=datapoints, plot=False)

    if datapoints not in mse:
        mse[datapoints] = np.zeros((REPS, MEASURES*len(methods)))
    for n in range(REPS):
        (names, res) = main_run(methods, params, vecX, vecy, vecz, train_fracs, 0.1, states, False)
        perf = mse[datapoints]
        cnt = 0
        for p in range(MEASURES):
            for m in range(len(methods)):
                perf[n, cnt] = res[m][0][p]
                cnt += 1

    measure_names = res[0][1]
    means = np.zeros((1, MEASURES*len(methods)))
    stds = np.zeros((1, MEASURES*len(methods)))

    print '\n'
    print '================================================ FINAL RESULT'
    idx = 0
    key = datapoints
    means[idx, :] = np.mean(mse[key], axis=0)
    stds[idx, :] = np.std(mse[key], axis=0)
    print 'Datapoints:(',key, ')  States(',states,')  Train-Frac(',train_fracs,')'
    print '------------------------------------------------'
    m = means[idx, :].reshape((len(names), MEASURES), order='F')
    s = stds[idx, :].reshape((len(names), MEASURES), order='F')
    print ''.ljust(44), '', res[0][1]
    for i in range(len(names)):
        name = names[i].ljust(45)
        for j in range(MEASURES):
            name += ' {0:+3.4f}/{1:1.2f}'.format(m[i,j], s[i,j]).ljust(23)
        print name
    print '================================================ FINAL RESULT END'
