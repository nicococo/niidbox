import numpy as np
import cvxopt as co
import matplotlib.pyplot as plt

from lccad import LCCAD


if __name__ == '__main__':
    REPS = 5
    THETAS = 5
    EXMS_C0 = 150
    EXMS_C1 = 150


    # ---------------------------
    # Generate the data set
    # ---------------------------
    z = np.append(np.ones(EXMS_C0)*0, np.ones(EXMS_C1)*1)
    x = np.zeros((EXMS_C0 + EXMS_C1, 2))
    x[:, 0] = np.append(np.random.randn(EXMS_C0) - 1, np.random.randn(EXMS_C1) + 1)
    x[:, 1] = np.append(np.random.randn(EXMS_C0) - 3, np.random.randn(EXMS_C1) - 1)
    # x[:, 1] = 0.25*x[:, 0]*z + 0.5*np.append(np.random.randn(EXMS_C0)*0.1, np.random.randn(EXMS_C1)*0.1 )
    # x[:, 1] = z+ 0.5*np.append(np.random.randn(EXMS_C0)*0.1, np.random.randn(EXMS_C1)*0.1 )
    # x[:, 1] = z
    # x[:, 2] = 1.
    x = x - np.mean(x, axis=0)
    num_exms = x.shape[0]

    # ---------------------------
    # Build connectivity graph
    # ---------------------------
    edges = np.zeros((num_exms, 3), dtype=np.int64)
    neighbors = np.zeros(num_exms, dtype=np.int32)
    A = co.spmatrix(0, [], [], (num_exms, num_exms), tc='d')
    for i in range(1, num_exms):
        e1 = i - 1
        e2 = i
        A[e1, e2] = 1
        A[e2, e1] = 1
        edges[i - 1, :] = (e1, e2, 1)
        neighbors[e1] += 1
        neighbors[e2] += 1

    # ---------------------------
    # Train and Test
    # ---------------------------
    scores = np.zeros((REPS, THETAS, num_exms))
    res = np.zeros((REPS, THETAS))
    theta_vals = np.linspace(0, 1, THETAS)
    theta_vals = [0., 0.05, 0.1, 0.2, 1.0]
    zs = np.zeros((THETAS, num_exms))
    v = []
    for r in range(REPS):
        for t in range(THETAS):
            qp = LCCAD(x.T, states=2, A=A, reg_theta=theta_vals[t])
            qp.fit(auto_adjust=True, use_grads=False)
            v.append(qp.unpack_v(qp.v)[:4])
            scores[r, t, :], _ = qp.predict()
            z_inv = z.copy()
            z_inv[z == 0] = 1.
            z_inv[z == 1] = 0.
            res[r, t] = np.max([np.sum(z == qp.latent), np.sum(z_inv == qp.latent)])
            zs[t, :] = qp.latent

    print theta_vals
    print '\n --- \n'
    print np.mean(res, axis=0)
    print '\n --- \n'
    print np.mean(res / np.float(z.size), axis=0)

    print v

    for t in range(THETAS):
        plt.subplot(2, THETAS, t+1)
        plt.title('theta={0}'.format(theta_vals[t]))
        plt.plot(x[z==0, 0], x[z==0, 1], '.b', linewidth=1)
        plt.plot(x[z==1, 0], x[z==1, 1], '.r', linewidth=1)
        plt.plot(qp.u[0, 0], qp.u[1, 0], 'og', linewidth=2)
        plt.plot(qp.u[0, 1], qp.u[1, 1], 'og', linewidth=2)

        zt = zs[t, :]
        zt_inv = zt.copy()
        zt_inv[zt == 0] = 1.
        zt_inv[zt == 1] = 0.
        if np.sum(zt_inv == z) > np.sum(zt == z):
            zt = zt_inv

        inds = np.where(zt != z)[0]
        plt.plot(x[inds, 0], x[inds, 1], 'xk', linewidth=1)

        # anomaly scores
        plt.subplot(2, THETAS, THETAS + t + 1)

        s = np.mean(scores[:, t, :], axis=0)
        # print scores[0,t,:]
        if np.sum(zt == 0) > 0:
            s[zt == 0] /= np.max(s[zt == 0])
        if np.sum(zt == 1) > 0:
            s[zt == 1] /= np.max(s[zt == 1])
        s -= 0.4

        plt.scatter(x[:, 0], x[:, 1], np.sign(s*8.), 'k', linewidth=1)

        #
        # plt.plot(x[z==0,0], x[z==0,1], '.r', linewidth=2)
        # plt.plot(x[z==1,0], x[z==1,1], '.b', linewidth=2)
        #
        # plt.plot(x[zs[t,:]==0,0], x[zs[t,:]==0,1], 'xr', linewidth=1)
        # plt.plot(x[zs[t,:]==1,0], x[zs[t,:]==1,1], 'xb', linewidth=1)

    # plt.subplot(1, 3, 2)
    # plt.plot(x, z, '.k', linewidth=2)
    # plt.plot(x, zs[2, :], 'or', linewidth=1)
    # plt.subplot(1, 3, 3)
    # plt.plot(x, z, '.k', linewidth=2)
    # plt.plot(x, zs[-1, :], 'or', linewidth=1)
    plt.show()
