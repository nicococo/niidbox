from utils_experiment import *


def accs_datapoints(fname, method_set='lb,rr', reps=3, datapoints=[400, 800], states=2, train_fracs=0.5):
    # generate parameter sets
    methods, params = generate_param_set(method_set)
    MEASURES = 7

    mse = np.zeros((reps, MEASURES*len(methods), len(datapoints)))
    for n in range(reps):
        for d in range(len(datapoints)):
            (vecX, vecy, vecz) = get_1d_toy_data(num_exms=datapoints[d], plot=False)
            (names, res) = main_run(methods, params, vecX, vecy, vecz,
                                    train_fracs, 0.5, states, False)
            cnt = 0
            for p in range(MEASURES):
                for m in range(len(methods)):
                    mse[n, cnt, d] = res[m][0][p]
                    cnt += 1
    measure_names = res[0][1]

    res_means = np.zeros((len(names), MEASURES, len(datapoints)))
    res_stds = np.zeros((len(names), MEASURES, len(datapoints)))
    print '\n'
    print '================================================ FINAL RESULT'
    for d in range(len(datapoints)):
        means = np.mean(mse[:,:,d], axis=0)
        stds  = np.std(mse[:,:,d], axis=0)
        print '------------------------------------------------'
        print 'Datapoints:(', datapoints[d], ')  States(', states,')  Train-Frac(', train_fracs,')'
        print '------------------------------------------------'
        m = means.reshape((len(names), MEASURES), order='F')
        s = stds.reshape((len(names), MEASURES), order='F')
        res_means[:, :, d] = m
        res_stds[:, :, d] = s
        print ''.ljust(44), '', measure_names
        for i in range(len(names)):
            name = names[i].ljust(45)
            for j in range(MEASURES):
                name += ' {0:+3.4f}/{1:1.2f}'.format(m[i,j], s[i,j]).ljust(23)
            print name
    print '================================================ FINAL RESULT END'
    np.savez(fname, methods=method_set, method_names=names, states=states,
             reps=reps, train_frac=train_fracs, measure_names=measure_names,
             means=res_means, stds=res_stds, datapoints=datapoints)


def accs_fraction(fname, method_set='lb,rr', reps=3, datapoints=800, states=2, train_fracs=[0.5, 0.75]):
    # generate parameter sets
    methods, params = generate_param_set(method_set)
    MEASURES = 7

    mse = np.zeros((reps, MEASURES*len(methods), len(train_fracs)))
    for n in range(reps):
        for d in range(len(train_fracs)):
            (vecX, vecy, vecz) = get_1d_toy_data(num_exms=datapoints, plot=False)
            (names, res) = main_run(methods, params, vecX, vecy, vecz,
                                    train_fracs[d], 0.5, states, False)
            cnt = 0
            for p in range(MEASURES):
                for m in range(len(methods)):
                    mse[n, cnt, d] = res[m][0][p]
                    cnt += 1
    measure_names = res[0][1]

    res_means = np.zeros((len(names), MEASURES, len(train_fracs)))
    res_stds = np.zeros((len(names), MEASURES, len(train_fracs)))
    print '\n'
    print '================================================ FINAL RESULT'
    for d in range(len(train_fracs)):
        means = np.mean(mse[:,:,d], axis=0)
        stds  = np.std(mse[:,:,d], axis=0)
        print '------------------------------------------------'
        print 'Datapoints:(', datapoints, ')  States(', states,')  Train-Frac(', train_fracs[d],')'
        print '------------------------------------------------'
        m = means.reshape((len(names), MEASURES), order='F')
        s = stds.reshape((len(names), MEASURES), order='F')
        res_means[:, :, d] = m
        res_stds[:, :, d] = s
        print ''.ljust(44), '', measure_names
        for i in range(len(names)):
            name = names[i].ljust(45)
            for j in range(MEASURES):
                name += ' {0:+3.4f}/{1:1.2f}'.format(m[i,j], s[i,j]).ljust(23)
            print name
    print '================================================ FINAL RESULT END'
    np.savez(fname, methods=method_set, method_names=names, states=states,
             reps=reps, train_fracs=train_fracs, measure_names=measure_names,
             means=res_means, stds=res_stds, datapoints=datapoints)


def accs_states(fname, method_set='lb,rr', reps=3, datapoints=800, states=[2,3], train_fracs=0.5):
    # generate parameter sets
    methods, params = generate_param_set(method_set)
    MEASURES = 7

    mse = np.zeros((reps, MEASURES*len(methods), len(states)))
    for n in range(reps):
        for d in range(len(states)):
            (vecX, vecy, vecz) = get_1d_toy_data(num_exms=datapoints, plot=False)
            (names, res) = main_run(methods, params, vecX, vecy, vecz,
                                    train_fracs, 0.5, states[d], False)
            cnt = 0
            for p in range(MEASURES):
                for m in range(len(methods)):
                    mse[n, cnt, d] = res[m][0][p]
                    cnt += 1
    measure_names = res[0][1]

    res_means = np.zeros((len(names), MEASURES, len(states)))
    res_stds = np.zeros((len(names), MEASURES, len(states)))
    print '\n'
    print '================================================ FINAL RESULT'
    for d in range(len(states)):
        means = np.mean(mse[:,:,d], axis=0)
        stds  = np.std(mse[:,:,d], axis=0)
        print '------------------------------------------------'
        print 'Datapoints:(', datapoints, ')  States(', states[d],')  Train-Frac(', train_fracs,')'
        print '------------------------------------------------'
        m = means.reshape((len(names), MEASURES), order='F')
        s = stds.reshape((len(names), MEASURES), order='F')
        res_means[:, :, d] = m
        res_stds[:, :, d] = s
        print ''.ljust(44), '', measure_names
        for i in range(len(names)):
            name = names[i].ljust(45)
            for j in range(MEASURES):
                name += ' {0:+3.4f}/{1:1.2f}'.format(m[i,j], s[i,j]).ljust(23)
            print name
    print '================================================ FINAL RESULT END'
    np.savez(fname, methods=method_set, method_names=names, states=states,
             reps=reps, train_fracs=train_fracs, measure_names=measure_names,
             means=res_means, stds=res_stds, datapoints=datapoints)


def plot_fractions(fname):
    foo = np.load(fname)\
    # methods=method_set, method_names=names, states=states,
    # reps=reps, train_fracs=train_fracs, measure_names=measure_names,
    # means=res_means, stds=res_stds, datapoints=datapoints)
    plt.figure(1)
    # names = foo['method_names']
    # names[-1] = 'MoE (FlexMix)'
    # names[0] = 'Optimal'

    styles = [['--o', 4, 4, [0.9, 0.2, 0.1]], # lower bound
              [ '-o', 3, 4, [0.4, 0.4, 0.9]], # tcrfr qp
              [ '-o', 3, 4, [0.4, 0.6, 0.9]], # tcrfr lbpa
              [ '-.', 4, 2, [0.1, 0.2, 0.1]], # ridge regr.
              [ '-x', 3, 2, [0.1, 0.9, 0.1]], # svr
              [ '-x', 3, 2, [0.1, 0.7, 0.1]], # svr (rbf)
              [ '-x', 3, 2, [0.1, 0.5, 0.1]], # lap svr
              [ '-s', 4, 4, [0.8, 0.8, 0.6]], # k-means + regression
              [ '-h', 4, 4, [0.9, 0.2, 0.8]], # transd. regr.
              [ '-H', 4, 4, [0.1, 0.8, 0.6]], # flexmix
              [ '-x', 4, 4, [0.8, 0.8, 0.6]], # -
              [ '-x', 4, 4, [0.8, 0.1, 0.6]], # -
              ]

    names = ['Optimal']
    theta = 0.0
    for i in range(11):
        names.append('TCRFR (theta={0})'.format(theta))
        theta += 0.1

    for d in range(7):
        plt.subplot(2, 4, d+1)
        for m in range(len(foo['method_names'])):
            # plt.plot(foo['train_fracs'], foo['means'][m, d, :])
            plt.errorbar(foo['train_fracs'], foo['means'][m, d, :], foo['stds'][m, d, :],
                         linewidth=styles[m][1], alpha=0.7, elinewidth=1,
                         markersize=styles[m][2], fmt=styles[m][0],
                         color=styles[m][3])
            plt.ylabel(foo['measure_names'][d], fontsize=16)
            plt.xlabel('Fraction of labeled samples', fontsize=16)
            plt.xticks(foo['train_fracs'], fontsize=14, rotation=40)
            plt.grid('on')

    plt.legend(names, fontsize=16, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_neighbors(fname):
    foo = np.load(fname)\
    # methods=method_set, method_names=names, states=states,
    # reps=reps, train_fracs=train_fracs, measure_names=measure_names,
    # means=res_means, stds=res_stds, datapoints=datapoints)
    plt.figure(1)
    names = foo['method_names']
    names[0] = 'Optimal'

    styles = [['--o', 4, 4, [0.9, 0.2, 0.1]], # lower bound
              [ '-.o', 3, 4, [0.2, 0.2, 0.9]], # tcrfr 1
              [ '--o', 3, 4, [0.2, 0.4, 0.9]], # tcrfr 2
              [ '-o', 3, 4, [0.4, 0.4, 0.9]], # tcrfr 4
              [ '--o', 3, 4, [0.4, 0.6, 0.9]], # tcrfr 6
              [ '--o', 3, 4, [0.4, 0.8, 0.9]], # tcrfr 8
              [ '-.o', 3, 4, [0.6, 0.8, 0.9]], # tcrfr 12
              ]
    m = 1
    names = ['Optimal']
    for n in [1,2,4,6,8,12]:
        names.append('TCRFR-LBPA ({0})'.format(n))
        m += 1
    for d in range(7):
        plt.subplot(2, 4, d+1)
        for m in range(len(foo['method_names'])):
            # plt.plot(foo['train_fracs'], foo['means'][m, d, :])
            plt.errorbar(foo['train_fracs'], foo['means'][m, d, :], foo['stds'][m, d, :],
                         linewidth=styles[m][1], alpha=0.7, elinewidth=1,
                         markersize=styles[m][2], fmt=styles[m][0],
                         color=styles[m][3])
            plt.ylabel(foo['measure_names'][d], fontsize=16)
            plt.xlabel('Fraction of labeled samples', fontsize=16)
            plt.xticks(foo['train_fracs'], fontsize=14, rotation=40)
            plt.grid('on')

    plt.legend(names, fontsize=16, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_states(fname):
    foo = np.load(fname)\
    # methods=method_set, method_names=names, states=states,
    # reps=reps, train_fracs=train_fracs, measure_names=measure_names,
    # means=res_means, stds=res_stds, datapoints=datapoints)
    plt.figure(1)
    names = foo['method_names']
    names[-1] = 'MoE (FlexMix)'
    names[0] = 'Optimal'

    styles = [['--o', 4, 4, [0.9, 0.2, 0.1]], # lower bound
              [ '-o', 3, 4, [0.4, 0.4, 0.9]], # tcrfr qp
              [ '-o', 3, 4, [0.4, 0.6, 0.9]], # tcrfr lbpa
              [ '-.', 4, 2, [0.1, 0.2, 0.1]], # ridge regr.
              [ '-s', 4, 4, [0.8, 0.8, 0.6]], # k-means + regression
              [ '-H', 4, 4, [0.1, 0.8, 0.6]], # flexmix
              ]

    for d in range(7):
        plt.subplot(2, 4, d+1)
        for m in range(len(foo['method_names'])):
            if m > 0:
                plt.errorbar(foo['states'], foo['means'][m, d, :], foo['stds'][m, d, :],
                             linewidth=styles[m][1], alpha=0.7, elinewidth=1,
                             markersize=styles[m][2], fmt=styles[m][0],
                             color=styles[m][3])
            else:
                plt.errorbar(foo['states'][1:], foo['means'][m, d, 1:], foo['stds'][m, d, 1:],
                             linewidth=styles[m][1], alpha=0.7, elinewidth=1,
                             markersize=styles[m][2], fmt=styles[m][0],
                             color=styles[m][3])
            plt.ylabel(foo['measure_names'][d], fontsize=16)
            plt.xlabel('Number of latent states', fontsize=16)
            plt.xticks(foo['states'], fontsize=14, rotation=40)
            plt.grid('on')

    plt.legend(names, fontsize=16, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_datapoints(fname):
    foo = np.load(fname)\
    # methods=method_set, method_names=names, states=states,
    # reps=reps, train_fracs=train_fracs, measure_names=measure_names,
    # means=res_means, stds=res_stds, datapoints=datapoints)
    plt.figure(1)
    names = foo['method_names']
    names[0] = 'Optimal'

    styles = [['--o', 4, 4, [0.9, 0.2, 0.1]], # lower bound
              [ '-o', 3, 4, [0.4, 0.4, 0.9]], # tcrfr qp
              [ '-o', 3, 4, [0.4, 0.6, 0.9]], # tcrfr lbpa
              [ '-.', 4, 2, [0.1, 0.2, 0.1]], # ridge regr.
              [ '-x', 3, 2, [0.1, 0.9, 0.1]], # svr
              [ '-x', 3, 2, [0.1, 0.7, 0.1]], # svr (rbf)
              [ '-x', 3, 2, [0.1, 0.5, 0.1]], # lap svr
              [ '-s', 4, 4, [0.8, 0.8, 0.6]], # k-means + regression
              [ '-h', 4, 4, [0.9, 0.2, 0.8]], # transd. regr.
              [ '-H', 4, 4, [0.1, 0.8, 0.6]], # flexmix
              ]

    for d in [6]:
        for m in range(len(foo['method_names'])):
            # plt.plot(foo['train_fracs'], foo['means'][m, d, :])
            plt.errorbar(foo['datapoints'], foo['means'][m, d, :], foo['stds'][m, d, :],
                         linewidth=styles[m][1], alpha=0.7, elinewidth=1,
                         markersize=styles[m][2], fmt=styles[m][0],
                         color=styles[m][3])
            plt.ylabel(foo['measure_names'][d], fontsize=16)
            plt.xlabel('Number of samples', fontsize=16)
            plt.xticks(foo['datapoints'], fontsize=14, rotation=40)
            plt.grid('on')

    plt.legend(names, fontsize=16, loc='upper left')
    plt.show()


if __name__ == '__main__':
    # plot_datapoints('res_acc_dpts_20.npz')
    # plot_states('res_acc_states_20.npz')
    # plot_fractions('res_acc_fracs_20.npz')
    # plot_neighbors('res_acc_fracs_neighb_20.npz')

    # plot_fractions('res_acc_fracs_theta_10.npz')

    # accs_fraction('res_acc_fracs_theta_10',
    #                 reps=10,
    #                 datapoints=800,
    #                 train_fracs=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75],
    #                 states=2,
    #                 method_set='lb,theta')

    # accs_fraction('res_acc_fracs_neighb_20',
    #                 reps=20,
    #                 datapoints=800,
    #                 train_fracs=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75],
    #                 states=2,
    #                 method_set='lb,neighbors')

    # accs_states('res_acc_states_20',
    #                 reps=20,
    #                 datapoints=800,
    #                 train_fracs=0.5,
    #                 states=[1, 2, 3, 4, 5],
    #                 method_set='lb,tcrfr_lbpa,tcrfr_qp,krr,flexmix')
    # accs_fraction('res_acc_fracs_20',
    #                 reps=20,
    #                 datapoints=800,
    #                 train_fracs=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75],
    #                 # train_fracs=[0.05, 0.75],
    #                 states=2,
    #                 method_set='lb,tcrfr_lbpa,tcrfr_qp,rr,svr,ksvr,laprls,krr,tr,flexmix')

    # accs_datapoints('res_acc_dpts_20',
    #                 reps=20,
    #                 datapoints=[50, 100, 200, 400, 800, 1000],
    #                 train_fracs=0.25,
    #                 states=2,
    #                 method_set='lb,tcrfr_lbpa,tcrfr_qp')


    accs_datapoints('res_acc_dpts_20',
                    reps=1,
                    datapoints=[50, 100, 200, 400, 800, 1000],
                    train_fracs=0.25,
                    states=2,
                    method_set='lb,tcrfr_lbpa,tcrfr_qp')