import csv
import numpy as np
import matplotlib.pyplot as plt


from ssvm import SSVM
from so_well import SOWell


def load_well_data(fname):
    feature_names = ['ID','TIME','VS','IS','VP','AI','GR','RHOB','FACIES']
    # unprocessed data
    data = []
    with open(fname, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = reader.next()    
        cnt = 0
        for row in reader:
            f_well = int(row[0][-1])
            f_time = float(row[1])
            f_vs = float(row[2])
            f_is = float(row[3])
            f_vp = float(row[4])
            f_ai = float(row[5])
            f_gr = float(row[6])
            f_rhob = float(row[7])
            f_facies = int(row[8])-101
            data.append([f_well, f_facies, f_time, f_vs, f_is, f_vp, f_ai, f_gr, f_rhob])
            cnt += 1
    print cnt
    # convert to numpy data-label arrays
    D = np.array(data)
    well_id = np.uint8(D[:,0])
    X = D[:,[5, 6, 8]]
    y = np.array(D[:,1], dtype='int')
    return well_id,X,y

def split_well_inds(sel_id, well_id):
    inds = np.where(well_id==sel_id)[0]
    return inds, np.setxor1d(xrange(well_id.size),inds)

def normalize(means, var, X):
    return (X-means)/var

def make_chunks(lens, add, X, y, well_id):
    Xc = []
    yc = []
    well_idc = []
    seqn = y.size
    chunks = 0
    pos = 0
    while (pos<seqn):
        if pos+lens<seqn:
            Xc.append(X[pos:pos+lens,:].T)
            #print Xc[-1]
            yc.append(y[pos:pos+lens].reshape(lens))
            well_idc.append(well_id[pos:pos+lens])
        else:
            Xc.append(X[pos:,:].T)
            seqlen = y[pos:].size
            yc.append(y[pos:].reshape(seqlen))
            well_idc.append(well_id[pos:])
        pos += add
        chunks += 1

    print('Made {0} chunks with length {1}'.format(chunks, lens))
    print yc[0].shape
    return Xc, yc, well_idc


if __name__=='__main__':
    TEST_WELL = 1

    (well_id, X, y) = load_well_data('facies_simple.csv')
   
    # well indices
    well_inds = []
    rest_inds = []
    for i in range(4):
        (winds, rwinds) = split_well_inds(i+1, well_id)
        well_inds.append(winds)
        rest_inds.append(rwinds)
    tinds = well_inds[TEST_WELL]
    rinds = rest_inds[TEST_WELL]

    # select a whole well for testing
    (tinds, rinds) = split_well_inds(TEST_WELL, well_id)
    test_X = X[tinds,:].copy()
    test_y = y[tinds].copy()
    test_well_id = well_id[tinds].copy()

    # training data
    train_X = X[rinds,:].copy()
    train_y = y[rinds].copy()
    train_well_id = well_id[rinds].copy()

    # normalize data
    Xmean = np.mean(train_X, axis=0)
    Xstd = np.std(train_X, axis=0)
    train_X = normalize(Xmean, Xstd, train_X)
    test_X = normalize(Xmean, Xstd, test_X)

    # build training and test chunks
    Xt = []
    yt = []
    wt = []
    for i in range(4):
        if not i+1==TEST_WELL:
            inds = np.where(train_well_id==i+1)[0]
            (Xta, yta, wta) = make_chunks(95, 5, train_X[inds,:], train_y[inds], train_well_id[inds])
            print len(Xta)
            print Xta[1].size
            Xt.extend(Xta)
            yt.extend(yta)
            wt.extend(wta)
    (Xtest, ytest, wtest) = make_chunks(95, 5, test_X, test_y, test_well_id)

    print '-------------'
    print len(Xt)
    print Xt[0].shape
    print yt[0].size

    model_train = SOWell(Xt, yt)

    #Xtest = Xt
    #ytest = yt

    model_test = SOWell(Xtest, ytest)

    ssvm = SSVM(model_train)
    ssvm.train()
    (foo, preds) = ssvm.apply(model_test)

    (err, err_exm) = model_test.evaluate(preds)

    print err_exm
    print err

    # show plots
    plt.figure(1)
    pos = 0
    for i in range(len(ytest)):
        lens = ytest[i].size
        plt.plot(range(pos,pos+lens),ytest[i].T,'.-b')
        plt.plot(range(pos,pos+lens),preds[i].T,'-r')
        pos += lens
        plt.ylim((-1.0,3.0))
    

    plt.show()

    # plt.figure(1)
    # plt.plot(train_y,'-r')
    # plt.plot(train_X[:,0],'.-b')
    # plt.plot(train_X[:,1],'.-g')
    # plt.plot(train_X[:,2],'-k')

    # plt.figure(2)
    # plt.plot(test_y,'-r')
    # plt.plot(test_X[:,0],'.-b')
    # plt.plot(test_X[:,1],'.-g')
    # plt.plot(test_X[:,2],'-k')
    #plt.show()

    # ..and stop
    print('Finish!')

