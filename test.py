import numpy as np

a = np.array([[1,2,3,4,0],
             [5,6,7,8,9]])

#a[0] = a[0] + a[1]


print a[0]
print a

b = np.array([1,0])
c = np.array([1,3,4,5,6])


print np.tile(c.T, (5,1))
g = np.choose(b, a.T)
#print a - g.T
print "g: ", g
print "a: ", a
mat = (a.T - g).T
print mat
print np.sum(mat, axis=1)-1
print "mat: ", mat[b,[3,1]]
print "mat2: ", mat[[0,1],[3,1]]
#d = np.array
t = {}

t[(1,2)] = ('h','h')
print t
for u in np.arange(1,2,0.4):
    print u
print a+1
print np.max(a, axis=1)

sub = np.ones((a.shape[0]*4, a.shape[1]))

print sub

print (a.T - np.max(a, axis=1)).T
