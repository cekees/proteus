"""
A simple implementation of basic 3D Euclidean Geometry
"""
import math
from math import *
import numpy

X=0
Y=1
Z=2

def EVec(x=0.0,y=0.0,z=0.0):
    v = numpy.zeros((3,),'d')
    v[X] = x
    v[Y] = y
    v[Z] = z
    return v

#  class EVec(tuple):

#      """A vector in 3D Euclidean space.

#      The vector is represented by a 3-tuple of Cartesian coordinates.
#      EVec's are immuitable and can be used as dictionary keys.
#      Comparisons are done lexicographically, first z, then y, then x; therefore,
#      if a list of EVec's is sorted, it will vary first in x, then y, then z.
#      + and - are provided.
#      * is provided as scalar multiplication.
#      scalar, vector, tensor, and triple products are defined as edot,ecross,
#      etensor, and etriple."""

#      def __new__(type,x,y=0.0,z=0.0):
#          return tuple.__new__(type,(z,y,x))
#      def asArray(self):
#          return numpy.array([self[X],self[Y],self[Z]],'d')
#      def x(self):
#          return self[X]
#      def y(self):
#          return self[Y]
#      def z(self):
#          return self[Z]
#      def __str__(self):
#          return '(%15.8e,%15.8e,%15.8e)' % (self[X],self[Y],self[Z])
#      def __repr__(self):
#          return '(%15.8e,%15.8e,%15.8e)' % (self[X],self[Y],self[Z])
#      def __add__(self, other):
#          return EVec(self[X] + other[X],self[Y] + other[Y],self[Z] + other[Z])
#      def __iadd__(self, other):
#          return EVec(self[X] + other[X],self[Y] + other[Y],self[Z] + other[Z])
#      def __neg__(self):
#          return EVec(-self[X],-self[Y],-self[Z])
#      def __sub__(self, other):
#          return EVec(self[X] - other[X],self[Y] - other[Y],self[Z] - other[Z])
#      def __isub__(self, other):
#          return EVec(self[X] - other[X],self[Y] - other[Y],self[Z] - other[Z])
#      def __mul__(self,a):
#          return EVec(a*self[X],a*self[Y],a*self[Z])
#      def __imul__(self,a):
#          return EVec(a*self[X],a*self[Y],a*self[Z])
#      def __rmul__(self,a):
#          return (self*a)
#      def __div__(self,a):
#          return EVec(self[X]/a,self[Y]/a,self[Z]/a)
#      def __idiv__(self,a):
#          return EVec(self[X]/a,self[Y]/a,self[Z]/a)
#      def __rdiv__(self, a):
#          return EVec(a/self[X],a/self[Y],a/self[Z])

def enorm(v):
    return sqrt(v[X]**2 + v[Y]**2 + v[Z]**2)
def norm(v):
    if v.shape[0]==1:
        return abs(v[X])
    if v.shape[0]==2:
        return sqrt(v[X]**2 + v[Y]**2)
    else:
        return sqrt(v[X]**2 + v[Y]**2 + v[Z]**2)

def edot(v0,v1):
    return v0[X]*v1[X] + v0[Y]*v1[Y] + v0[Z]*v1[Z]

def dot(v0,v1):
    if v0.shape[0] == 1:
        return v0[X]*v1[X]
    elif v0.shape[0] == 2:
        return v0[X]*v1[X] + v0[Y]*v1[Y]
    elif v0.shape[0] == 3:
        return v0[X]*v1[X] + v0[Y]*v1[Y] + v0[Z]*v1[Z]

def ecross(v0,v1):
    return EVec(v0[Y]*v1[Z] - v0[Z]*v1[Y],
                v0[Z]*v1[X] - v0[X]*v1[Z],
                v0[X]*v1[Y] - v0[Y]*v1[X])

def etriple(v0,v1,v2):
    return edot(v0,ecross(v1,v2))

def eListDaxpy(aList,vList):
    ux=0.0
    uy=0.0
    uz=0.0
    for ai,vi in zip(aList,vList):
        ux+=ai*vi[X]
        uy+=ai*vi[Y]
        uz+=ai*vi[Z]
    return EVec(ux,uy,uz)

def ETen(rx=EVec(),ry=EVec(),rz=EVec()):
    t = numpy.zeros((3,3),'d')
    t[X][:] = rx
    t[Y][:] = ry
    t[Z][:] = rz
    return t

#  class ETen(tuple):

#      """A tensor in 3D Euclidean space.

#      The tensor is represented as a 3-tuple of EVec's (the rows).
#      ETen's are immutable and can be used as dictionary keys.
#      Comparisons are done lexicographically, first row z, then row y, then row x.
#      + and - are provided
#      * is provided for  scalar multiplication.
#      transpose, determinant, cofactor,adjugate,inverse, tensor multiplication, tensor-vector multiplication
#      are also defined"""

#      def __new__(type,rx,ry,rz):
#          return tuple.__new__(type,(rz,ry,rx))
#      def asArray(self):
#          return numpy.array([[self[X][X],self[X][Y],self[X][Z]],
#                        [self[Y][X],self[Y][Y],self[Y][Z]],
#                        [self[Z][X],self[Z][Y],self[Z][Z]]],'d')
#      def x(self):
#          return self[X]
#      def y(self):
#          return self[Y]
#      def z(self):
#          return self[Z]
#      def xx(self):
#          return self[X][X]
#      def xy(self):
#          return self[X][Y]
#      def xz(self):
#          return self[X][Z]
#      def yx(self):
#          return self[Y][X]
#      def yy(self):
#          return self[Y][Y]
#      def yz(self):
#          return self[Y][Z]
#      def zx(self):
#          return self[Z][X]
#      def zy(self):
#          return self[Z][Y]
#      def zz(self):
#          return self[Z][Z]
#      def __str__(self):
#          return '['+str(self[X])+'\n'+str(self[Y])+'\n'+str(self[Z])+']'
#      def __repr__(self):
#          return '['+str(self[X])+'\n'+str(self[Y])+'\n'+str(self[Z])+']'
#      def __add__(self, other):
#          return ETen(self[X] + other[X],self[Y] + other[Y],self[Z] + other[Z])
#      def __iadd__(self, other):
#          return ETen(self[X] + other[X],self[Y] + other[Y],self[Z] + other[Z])
#      def __neg__(self):
#          return ETen(-self[X],-self[Y],-self[Z])
#      def __sub__(self, other):
#          return ETen(self[X] - other[X],self[Y] - other[Y],self[Z] - other[Z])
#      def __isub__(self, other):
#          return ETen(self[X] - other[X],self[Y] - other[Y],self[Z] - other[Z])
#      def __mul__(self,a):
#          return ETen(a*self[X],a*self[Y],a*self[Z])
#      def __imul__(self,a):
#          return ETen(a*self[X],a*self[Y],a*self[Z])
#      def __rmul__(self,a):
#          return (self*a)
#      def __div__(self,a):
#          return ETen(self[X]/a,self[Y]/a,self[Z]/a)
#      def __idiv__(self,a):
#          return ETen(self[X]/a,self[Y]/a,self[Z]/a)
#      def __rdiv__(self, other):
#          return ETen(a/self[X],a/self[Y],a/self[Z])

def etensor(u,v):
    return ETen(EVec(u[X]*v[X],u[X]*v[Y],u[X]*v[Z]),
                EVec(u[Y]*v[X],u[Y]*v[Y],u[Y]*v[Z]),
                EVec(u[Z]*v[X],u[Z]*v[Y],u[Z]*v[Z]))

def etrans(t):
    return ETen(EVec(t[X][X],t[Y][X],t[Z][X]),
                EVec(t[X][Y],t[Y][Y],t[Z][Y]),
                EVec(t[X][Z],t[Y][Z],t[Z][Z]))
def trans(t):
    if t.shape[0] == 1:
        return t
    elif t.shape[0] == 2:
        tt = numpy.zeros(t.shape,'d')
        tt[X,X] = t[X,X]
        tt[X,Y] = t[Y,X]
        tt[Y,X] = t[X,Y]
        tt[Y,Y] = t[Y,Y]
        return tt
    else:
        return etrans(t)

def edet(t):
    """The determinant of the the 3x3 tensor [v0,v1,v2]."""
    return t[X][X]*(t[Y][Y]*t[Z][Z] - t[Y][Z]*t[Z][Y]) - \
           t[X][Y]*(t[Y][X]*t[Z][Z] - t[Y][Z]*t[Z][X]) + \
           t[X][Z]*(t[Y][X]*t[Z][Y] - t[Y][Y]*t[Z][X])
def det(t):
    if t.shape[0] == 1:
        return t[X,X]
    elif t.shape[0] == 2:
        return t[X,X]*t[Y,Y]-t[X,Y]*t[Y,X]
    else:
        return edet(t)

def ecof(t):
    return ETen(ecross(t[Y],t[Z]),ecross(t[Z],t[X]),ecross(t[X],t[Y]))

def cof(t):
    if t.shape[0] == 1:
        return numpy.array([[1.0]])
    elif t.shape[0] == 2:
        tcof = numpy.zeros(t.shape,'d')
        tcof[X,X] = t[Y,Y]
        tcof[X,Y] = -t[Y,X]
        tcof[Y,X] = -t[X,Y]
        tcof[Y,Y] = t[X,X]
        return tcof
    else:
        return ETen(ecross(t[Y],t[Z]),ecross(t[Z],t[X]),ecross(t[X],t[Y]))

def eadj(t):
    return etrans(ecof(t))

def adj(t):
    return trans(cof(t))

def einv(t):
    return eadj(t)/edet(t)

def inv(t):
    return adj(t)/det(t)

def etenten(r,t):
    tt = etrans(t)
    return ETen(EVec(edot(r[X],tt[X]),edot(r[X],tt[Y]),edot(r[X],tt[Z])),
                EVec(edot(r[Y],tt[X]),edot(r[Y],tt[Y]),edot(r[Y],tt[Z])),
                EVec(edot(r[Z],tt[X]),edot(r[Z],tt[Y]),edot(r[Z],tt[Z])))

def etenvec(t,v):
    return EVec(edot(t[X],v),edot(t[Y],v),edot(t[Z],v))

def tenvec(t,v):
    if v.shape[0] == 1:
        return numpy.array([dot(t[X],v)])
    elif v.shape[0] == 2:
        return numpy.array([dot(t[X],v),dot(t[Y],v)])
    elif v.shape[0] == 3:
        return EVec(edot(t[X],v),edot(t[Y],v),edot(t[Z],v))

def etenvecdotvec(t,v,w):
    return edot(t[X],v)*w[X]+edot(t[Y],v)*w[Y] + edot(t[Z],v)*w[Z]

## @}

if __name__=='__main__':
    import random
    #ex=EVec(1.0,0.0,0.0)
    ex=EVec(1.0)
    #ey=EVec(0.0,1.0,0.0)
    ey=EVec(0.0,1.0)
    ez=EVec(0.0,0.0,1.0)
    print 'ex = '+`ex`
    print 'ey = '+`ey`
    print 'ez = '+`ez`
    et=ETen(ex,ey,ez)
    print 'et = '+`et`
    #perturb the standard basis
    eps=0.1
    epsv=EVec(random.uniform(-eps,eps),
             random.uniform(-eps,eps),
             random.uniform(-eps,eps))
    u = ex + epsv
    eps=0.01
    epsv=EVec(random.uniform(-eps,eps),
             random.uniform(-eps,eps),
             random.uniform(-eps,eps))
    v = ey + epsv
    eps=0.001
    epsv=EVec(random.uniform(-eps,eps),
             random.uniform(-eps,eps),
             random.uniform(-eps,eps))
    w = ez + epsv
    t = ETen(u,v,w)
    print 'u = '+`u`
    print 'u[X] = '+`u[X]`
    print 'u[Y] = '+`u[Y]`
    print 'u[Z] = '+`u[Z]`
    print 'v = '+`v`
    print 'u + v = '+`u+v`
    print '2.0*u = '+`2.0*u`
    print 'u*2.0 = '+`u*2.0`
    print 'u/2.0 = '+`u/2.0`
    print '2.0/u = '+`2.0/u`
    print '-u = '+`-u`
    print '2.0*u + v  = ' + `2.0*u+v`
    print 'u - v = '+`u-v`
    print 'u = '+`u`
    u*=2.0
    print 'u *= 2.0 ' + `u`
    u/=2.0
    print 'u /= 2.0 ' + `u`
    u+=v
    print 'u += v '+` u`
    print 'u = '+`u`
    print 'enorm(u) = '+`enorm(u)`
    print 'enorm(ex) = '+`enorm(ex)`
    print 'enorm(ey) = '+`enorm(ey)`
    print 'enorm(ez) = '+`enorm(ez)`
    print 'u = '+`u`
    print 'edot(u,ex) = '+`edot(u,ex)`
    print 'edot(u,ey) = '+`edot(u,ey)`
    print 'edot(u,ez) = '+`edot(u,ez)`
    print 'edot(ex,u) = '+`edot(ex,u)`
    print 'edot(ey,u) = '+`edot(ey,u)`
    print 'edot(ez,u) = '+`edot(ez,u)`
    print 'ecross(ex,ey) = '+`ecross(ex,ey)`
    print 'ecross(ex,ez) = '+`ecross(ex,ez)`
    print 'ecross(ey,ez) = '+`ecross(ey,ez)`
    print 'ecross(ey,ex) = '+`ecross(ey,ex)`
    print 'ecross(ez,ex) = '+`ecross(ez,ex)`
    print 'ecross(ez,ey) = '+`ecross(ez,ey)`
    print 'etriple(u,v,w) = '+`etriple(u,v,w)`
    print 'etriple(u,u,v) = '+`etriple(u,u,v)`
    print 'u = '+`u`
    print 'v = '+`v`
    print 'etensor(u,v) = '+`etensor(u,v)`
    print 't = '+`t`
    print 't[X] = '+`t[X]`
    print 't[Y] = '+`t[Y]`
    print 't[Z] = '+`t[Z]`
    print 't[X][X] = '+`t[X][X]`
    print 't[X][Y] = '+`t[X][Y]`
    print 't[X][Z] = '+`t[X][Z]`
    print 't[Y][X] = '+`t[Y][X]`
    print 't[Y][Y] = '+`t[Y][Y]`
    print 't[Y][Z] = '+`t[Y][Z]`
    print 't[Z][X] = '+`t[Z][X]`
    print 't[Z][Y] = '+`t[Z][Y]`
    print 't[Z][Z] = '+`t[Z][Z]`
    print 't+et = '+`t+et`
    print '2.0*et = '+`2.0*et`
    print '-et = '+`-et`
    print 't+2.0*et = '+`t+2.0*et`
    print 't-et = '+`t-et`
    s=ETen()
    s[:][:]=t
    print 's = '+`s`
    s*=2.0
    print 's*=2.0'+`s`
    s/=2.0
    print 's/=2.0'+`s`
    s+=s
    print 's+=s'+`s`
    s-=s
    print 's-=s'+`s`
    print 'etrans(t) = '+`etrans(t)`
    print 'edet(t) = '+`edet(t)`
    print 'ecof(t) = '+`ecof(t)`
    print 'eadj(t) = '+`eadj(t)`
    print 'einv(t) = '+`einv(t)`
    print 'etenten(t,et) = '+`etenten(t,et)`
    print 'etenten(t,t) = '+`etenten(t,t)`
    print 'etenten(t,einv(t)) = '+`etenten(t,einv(t))`
    print 'etenten(einv(t),t) = '+`etenten(einv(t),t)`
    print 'etenvec(t,ex) = '+`etenvec(t,ex)`
    print 'etenvec(t,ey) = '+`etenvec(t,ey)`
    print 'etenvec(t,ez) = '+`etenvec(t,ez)`
    print 'etenvec(t,u) = '+`etenvec(t,u)`
    vl = [w,v,u]
    print 'vl = '+`vl`
    vl.sort()
    print 'sorted vl = '+`vl`
