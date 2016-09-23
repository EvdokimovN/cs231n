import numpy as np

a = np.array([[1,3,4],
              [2,1,8],
              [3,1,2]])

b = np.array([[1, 1, 1],
              [2, 2, 3]])
test = np.zeros((b.shape[0], a.shape[0]))
test[1,1] = 2
print test
print a.shape, test.shape
print
print "test: %s" %np.sum((abs(b[0,:] - a)), axis=1)

for i in xrange(b.shape[0]):
    test[i,:] = np.sum(np.square(b[i,:] - a), axis=1)
print a
print "dist: %s" %test

idx = np.array([1,1])
c  = np.array([1,2,4,5,6])
print c[np.argsort(a[1])[:2]]




g = np.split(a,3)
l = np.arange(1,4)
print l.shape
mask = np.ones((g.__len__(),),dtype=bool)
print mask.shape
mask[1] = 0
print np.concatenate(g[:1] + g[6:])
pr = {}
pr[1] = np.mean(4)
pr[1] = np.mean((2,pr[1]))
pr[2] = []
pr[2].append(2)
print pr[2]
print type(float(pr[1]))