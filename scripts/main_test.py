import argparse, sys
from utils_experiment import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--states", help="List of states for testing (default=3).", default='2', type=int)
    parser.add_argument("-f", "--train_frac", help="Fraction of training exms (default=0.15)", default='0.25', type=float)
    parser.add_argument("-d", "--datapoints", help="Amount of data points (default=1000)", default='600', type=int)
    parser.add_argument("-r", "--reps", help="Number of repetitions (default 10)", default=3, type=int)
    parser.add_argument("-m", "--method_set", help="Select active method set. (default 'full')", default='lb,rr,tcrfr_lbpa,tcrfr_qp', type=str)
    arguments = parser.parse_args(sys.argv[1:])
    print arguments

    # generate parameter sets
    methods, params = generate_param_set(arguments.method_set)

    MEASURES = 7
    REPS = arguments.reps

    datapoints = arguments.datapoints
    states = arguments.states
    train_fracs = arguments.train_frac

    (vecX, vecy, vecz) = get_1d_toy_data(num_exms=datapoints, plot=True)
    mse = np.zeros((REPS, MEASURES*len(methods)))
    for n in range(REPS):
        (names, res) = main_run(methods, params, vecX, vecy, vecz, train_fracs, 0.5, states, False)
        cnt = 0
        for p in range(MEASURES):
            for m in range(len(methods)):
                mse[n, cnt] = res[m][0][p]
                cnt += 1

    measure_names = res[0][1]
    means = np.zeros((MEASURES*len(methods)))
    stds = np.zeros((MEASURES*len(methods)))

    print '\n'
    print '================================================ FINAL RESULT'
    means = np.mean(mse, axis=0)
    stds  = np.std(mse, axis=0)
    print 'Datapoints:(', datapoints, ')  States(', states,')  Train-Frac(', train_fracs,')'
    print '------------------------------------------------'
    m = means.reshape((len(names), MEASURES), order='F')
    s = stds.reshape((len(names), MEASURES), order='F')
    print ''.ljust(44), '', res[0][1]
    for i in range(len(names)):
        name = names[i].ljust(45)
        for j in range(MEASURES):
            name += ' {0:+3.4f}/{1:1.2f}'.format(m[i,j], s[i,j]).ljust(23)
        print name
    print '================================================ FINAL RESULT END'
