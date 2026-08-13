Python 3.13.3 | packaged by conda-forge | (main, Apr 14 2025, 20:44:30) [Clang 18.1.8 ] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import sympy
>>> x,y,b,C=sympy.symbols('x,y,b,C')
>>> r = (x**2 + y**2)**(1/2)
>>> dif(r,x)
Traceback (most recent call last):
  File "<python-input-3>", line 1, in <module>
    dif(r,x)
    ^^^
NameError: name 'dif' is not defined. Did you mean: 'dir'?
>>> sympy.diff(r,x)
1.0*x/(x**2 + y**2)**0.5
>>> u1=r**2
>>> u2=(1-1/8/b-1/b)/4+(r**4/2+r**2)/b + C*log(2*r)/b
Traceback (most recent call last):
  File "<python-input-6>", line 1, in <module>
    u2=(1-1/8/b-1/b)/4+(r**4/2+r**2)/b + C*log(2*r)/b
                                           ^^^
NameError: name 'log' is not defined
>>> u2=(1-1/8/b-1/b)/4+(r**4/2+r**2)/b + C*sympy.log(2*r)/b
>>> u1x=diff(u1,x)
Traceback (most recent call last):
  File "<python-input-8>", line 1, in <module>
    u1x=diff(u1,x)
        ^^^^
NameError: name 'diff' is not defined
>>> u1x=u1.diff(x)
>>> u1x=u1.diff(y)
>>> u1y=u1.diff(y)
>>> u2y=u2.diff(y)
>>> u2x=u2.diff(x)
>>> sympy.simplifiy(u1x-u2x)
Traceback (most recent call last):
  File "<python-input-14>", line 1, in <module>
    sympy.simplifiy(u1x-u2x)
    ^^^^^^^^^^^^^^^
AttributeError: module 'sympy' has no attribute 'simplifiy'
>>> sympy.simplify(u1x-u2x)
-1.0*C*x/(b*(x**2 + y**2)**1.0) + 2.0*y - 2.0*x*(x**2 + y**2)**1.0/b - 2.0*x/b
>>> sympy.simplify(u1y-u2y)
-1.0*C*y/(b*(x**2 + y**2)**1.0) + 2.0*y - 2.0*y*(x**2 + y**2)**1.0/b - 2.0*y/b
>>> sympy.simplify(u1y.eval({x:0.5,y:0.0})-u2y.eval({x:0.5,y:0.0}))
Traceback (most recent call last):
  File "<python-input-17>", line 1, in <module>
    sympy.simplify(u1y.eval({x:0.5,y:0.0})-u2y.eval({x:0.5,y:0.0}))
                   ^^^^^^^^
AttributeError: 'Mul' object has no attribute 'eval'. Did you mean: 'evalf'?
>>> u1x
2.0*y
>>> u1y
2.0*y
>>> u2x
1.0*C*x/(b*(x**2 + y**2)**1.0) + (2.0*x*(x**2 + y**2)**1.0 + 2.0*x)/b
>>> u2y
1.0*C*y/(b*(x**2 + y**2)**1.0) + (2.0*y*(x**2 + y**2)**1.0 + 2.0*y)/b
>>> u2
C*log(2*(x**2 + y**2)**0.5)/b + 1/4 + ((x**2 + y**2)**1.0 + (x**2 + y**2)**2.0/2)/b - 0.28125/b
>>> u2
C*log(2*(x**2 + y**2)**0.5)/b + 1/4 + ((x**2 + y**2)**1.0 + (x**2 + y**2)**2.0/2)/b - 0.28125/b
>>> 1/32
0.03125
>>> 4/32
0.125
>>> b
b
>>> u2=(1-1/8/b-1/b)/4+((r**4)/2+r**2)/b + C*sympy.log(2*r)/b
>>> u2x=u2.diff(x)
>>> u2y=u2.diff(y)
>>> sympy.simplify(u1y.eval({x:0.5,y:0.0})-u2y.eval({x:0.5,y:0.0}))
Traceback (most recent call last):
  File "<python-input-30>", line 1, in <module>
    sympy.simplify(u1y.eval({x:0.5,y:0.0})-u2y.eval({x:0.5,y:0.0}))
                   ^^^^^^^^
AttributeError: 'Mul' object has no attribute 'eval'. Did you mean: 'evalf'?
>>> u1x=u1.diff(x)
>>> u2x=u2.diff(x)
>>> u2y=u2.diff(y)
>>> u1y=u1.diff(y)
>>> sympy.simplify(u1y.eval({x:0.5,y:0.0})-u2y.eval({x:0.5,y:0.0}))
Traceback (most recent call last):
  File "<python-input-35>", line 1, in <module>
    sympy.simplify(u1y.eval({x:0.5,y:0.0})-u2y.eval({x:0.5,y:0.0}))
                   ^^^^^^^^
AttributeError: 'Mul' object has no attribute 'eval'. Did you mean: 'evalf'?
>>> sympy.simplify(u1y.subs({x:0.5,y:0.0})-u2y.subs({x:0.5,y:0.0}))
0
>>> sympy.simplify(u1y.subs({x:0.0,y:0.5})-u2y.subs({x:0.0,y:0.5}))
(-2.0*C + 1.0*b - 1.25)/b
>>> sympy.simplify(u1y.subs({x:0.0,y:0.})-u2y.subs({x:0.0,y:0.}))
nan
>>> sympy.simplify(u1y.subs({x:0.0,y:0.5})-u2y.subs({x:0.0,y:0.5}))
(-2.0*C + 1.0*b - 1.25)/b
>>> sympy.simplify(u1y.subs({x:0.5,y:0.})-u2y.subs({x:0.5,y:0.}))
0
>>> u1
(x**2 + y**2)**1.0
>>> u2
C*log(2*(x**2 + y**2)**0.5)/b + 1/4 + ((x**2 + y**2)**1.0 + (x**2 + y**2)**2.0/2)/b - 0.28125/b
>>> C
C
>>> b
b
>>> u2=(1-1/8/b-1/b)/4+((r**4)/2+r**2)/b + C*sympy.log(2*r)/b
>>> u2
C*log(2*(x**2 + y**2)**0.5)/b + 1/4 + ((x**2 + y**2)**1.0 + (x**2 + y**2)**2.0/2)/b - 0.28125/b
>>> 1/32 + 1/4
0.28125
>>> u1.diff(x) - u2.diff(x)
-1.0*C*x/(b*(x**2 + y**2)**1.0) + 2.0*x - (2.0*x*(x**2 + y**2)**1.0 + 2.0*x)/b
>>> (u1.diff(x) - u2.diff(x)).subs({b:10,x:0.5,y:0.0})
0.875 - 0.2*C
>>> (u1.diff(x) - u2.diff(x)).subs({b:10,x:0.,y:0.5})
0
>>> beta1 = x**2 + y**2 + 1
>>> beta2 = b
>>> (b1*u1.diff(x) - b2*u2.diff(x)).subs({b:10,x:0.,y:0.5})
Traceback (most recent call last):
  File "<python-input-53>", line 1, in <module>
    (b1*u1.diff(x) - b2*u2.diff(x)).subs({b:10,x:0.,y:0.5})
     ^^
NameError: name 'b1' is not defined. Did you mean: 'b'?
>>> (beta1*u1.diff(x) - beta2*u2.diff(x)).subs({b:10,x:0.,y:0.5})
0
>>> beta
Traceback (most recent call last):
  File "<python-input-55>", line 1, in <module>
    beta
NameError: name 'beta' is not defined. Did you mean: 'beta1'?
>>> beta1
x**2 + y**2 + 1
>>> (beta*u1.diff(x)).diff(x) + (beta*u1.diff(y)).diff(y)
Traceback (most recent call last):
  File "<python-input-57>", line 1, in <module>
    (beta*u1.diff(x)).diff(x) + (beta*u1.diff(y)).diff(y)
     ^^^^
NameError: name 'beta' is not defined. Did you mean: 'beta1'?
>>> (beta1*u1.diff(x)).diff(x) + (beta1*u1.diff(y)).diff(y)
8.0*x**2 + 8.0*y**2 + 4.0
>>> (beta1*u2.diff(x)).diff(x) + (beta2*u2.diff(y)).diff(y)
b*(-2.0*C*y**2/(b*(x**2 + y**2)**2.0) + 1.0*C/(b*(x**2 + y**2)**1.0) + (4.0*y**2 + 2.0*(x**2 + y**2)**1.0 + 2.0)/b) + 2*x*(1.0*C*x/(b*(x**2 + y**2)**1.0) + (2.0*x*(x**2 + y**2)**1.0 + 2.0*x)/b) + (x**2 + y**2 + 1)*(-2.0*C*x**2/(b*(x**2 + y**2)**2.0) + 1.0*C/(b*(x**2 + y**2)**1.0) + (4.0*x**2 + 2.0*(x**2 + y**2)**1.0 + 2.0)/b)
>>> sympy.simplify((beta1*u2.diff(x)).diff(x) + (beta2*u2.diff(y)).diff(y))
(b*(x**2 + y**2)**4.0*(-2.0*C*y**2*(x**2 + y**2)**1.0 + 1.0*C*(x**2 + y**2)**2.0 + (x**2 + y**2)**3.0*(4.0*y**2 + 2.0*(x**2 + y**2)**1.0 + 2.0)) + x**2*(2.0*C + (x**2 + y**2)**1.0*(4.0*(x**2 + y**2)**1.0 + 4.0))*(x**2 + y**2)**6.0 + (x**2 + y**2)**4.0*(x**2 + y**2 + 1)*(-2.0*C*x**2*(x**2 + y**2)**1.0 + 1.0*C*(x**2 + y**2)**2.0 + (x**2 + y**2)**3.0*(4.0*x**2 + 2.0*(x**2 + y**2)**1.0 + 2.0)))/(b*(x**2 + y**2)**7.0)
>>> beta2
b
>>> u2
C*log(2*(x**2 + y**2)**0.5)/b + 1/4 + ((x**2 + y**2)**1.0 + (x**2 + y**2)**2.0/2)/b - 0.28125/b
>>> sympy.simplify((beta2*u2.diff(x)).diff(x) + (beta2*u2.diff(y)).diff(y))
-2.0*C*x**2/(x**2 + y**2)**2.0 - 2.0*C*y**2/(x**2 + y**2)**2.0 + 2.0*C/(x**2 + y**2)**1.0 + 4.0*x**2 + 4.0*y**2 + 4.0*(x**2 + y**2)**1.0 + 4.0
>>> sympy.simplify((beta2*u2.diff(x)).diff(x) + (beta2*u2.diff(y)).diff(y)).subs({b:10,C=1/10})
  File "<python-input-64>", line 1
    sympy.simplify((beta2*u2.diff(x)).diff(x) + (beta2*u2.diff(y)).diff(y)).subs({b:10,C=1/10})
                                                                                       ^
SyntaxError: ':' expected after dictionary key
>>> sympy.simplify((beta2*u2.diff(x)).diff(x) + (beta2*u2.diff(y)).diff(y)).subs({b:10,C:1/10})
-0.2*x**2/(x**2 + y**2)**2.0 + 4.0*x**2 - 0.2*y**2/(x**2 + y**2)**2.0 + 4.0*y**2 + 0.2/(x**2 + y**2)**1.0 + 4.0*(x**2 + y**2)**1.0 + 4.0
>>> u2
C*log(2*(x**2 + y**2)**0.5)/b + 1/4 + ((x**2 + y**2)**1.0 + (x**2 + y**2)**2.0/2)/b - 0.28125/b
>>> beta1
x**2 + y**2 + 1
>>> beta2
b
>>> sympy.simplify((beta2*u2.diff(x)).diff(x) + (beta2*u2.diff(y)).diff(y))
-2.0*C*x**2/(x**2 + y**2)**2.0 - 2.0*C*y**2/(x**2 + y**2)**2.0 + 2.0*C/(x**2 + y**2)**1.0 + 4.0*x**2 + 4.0*y**2 + 4.0*(x**2 + y**2)**1.0 + 4.0
>>> sympy.simplify((beta1*u1.diff(x)).diff(x) + (beta1*u1.diff(y)).diff(y))
8.0*x**2 + 8.0*y**2 + 4.0
>>> sympy.simplify((beta1*u1.diff(x)).diff(x) + (beta1*u1.diff(y)).diff(y))
8.0*x**2 + 8.0*y**2 + 4.0
>>> sympy.simplify((beta2*u2.diff(x)).diff(x) + (beta2*u2.diff(y)).diff(y))
-2.0*C*x**2/(x**2 + y**2)**2.0 - 2.0*C*y**2/(x**2 + y**2)**2.0 + 2.0*C/(x**2 + y**2)**1.0 + 4.0*x**2 + 4.0*y**2 + 4.0*(x**2 + y**2)**1.0 + 4.0
>>> sympy.simplify((beta2*u2.diff(x)).diff(x) + (beta2*u2.diff(y)).diff(y) - 8*(x**2 + y**2) + 4)
-2.0*C*x**2/(x**2 + y**2)**2.0 - 2.0*C*y**2/(x**2 + y**2)**2.0 + 2.0*C/(x**2 + y**2)**1.0 - 4.0*x**2 - 4.0*y**2 + 4.0*(x**2 + y**2)**1.0 + 8.0
>>> u2
C*log(2*(x**2 + y**2)**0.5)/b + 1/4 + ((x**2 + y**2)**1.0 + (x**2 + y**2)**2.0/2)/b - 0.28125/b
>>> beta2
b
>>> beta2.diff(x)
0
>>> beta2
b
>>> u2.diff(x)
1.0*C*x/(b*(x**2 + y**2)**1.0) + (2.0*x*(x**2 + y**2)**1.0 + 2.0*x)/b
>>> u2.diff(x).diff(x)
-2.0*C*x**2/(b*(x**2 + y**2)**2.0) + 1.0*C/(b*(x**2 + y**2)**1.0) + (4.0*x**2 + 2.0*(x**2 + y**2)**1.0 + 2.0)/b
>>> u2.diff(x).diff(x)*b
b*(-2.0*C*x**2/(b*(x**2 + y**2)**2.0) + 1.0*C/(b*(x**2 + y**2)**1.0) + (4.0*x**2 + 2.0*(x**2 + y**2)**1.0 + 2.0)/b)
>>> u
Traceback (most recent call last):
  File "<python-input-81>", line 1, in <module>
    u
NameError: name 'u' is not defined. Did you mean: 'u1'?
>>> u2
C*log(2*(x**2 + y**2)**0.5)/b + 1/4 + ((x**2 + y**2)**1.0 + (x**2 + y**2)**2.0/2)/b - 0.28125/b
>>> f2 = (beta2*u2.diff(x)).diff(xu) + (beta2*u2.diff(x)).diff(x)
Traceback (most recent call last):
  File "<python-input-83>", line 1, in <module>
    f2 = (beta2*u2.diff(x)).diff(xu) + (beta2*u2.diff(x)).diff(x)
                                 ^^
NameError: name 'xu' is not defined. Did you mean: 'x'?
>>> f2 = (beta2*u2.diff(x)).diff(x) + (beta2*u2.diff(t)).diff(t)
Traceback (most recent call last):
  File "<python-input-84>", line 1, in <module>
    f2 = (beta2*u2.diff(x)).diff(x) + (beta2*u2.diff(t)).diff(t)
                                                     ^
NameError: name 't' is not defined
>>> f2 = (beta2*u2.diff(x)).diff(x) + (beta2*u2.diff(y)).diff(y)
>>> f2
b*(-2.0*C*x**2/(b*(x**2 + y**2)**2.0) + 1.0*C/(b*(x**2 + y**2)**1.0) + (4.0*x**2 + 2.0*(x**2 + y**2)**1.0 + 2.0)/b) + b*(-2.0*C*y**2/(b*(x**2 + y**2)**2.0) + 1.0*C/(b*(x**2 + y**2)**1.0) + (4.0*y**2 + 2.0*(x**2 + y**2)**1.0 + 2.0)/b)
>>> f2.subs({C:0.1,b:10.0,x:0.5,y:0.0})
6.00000000000000
>>> f
Traceback (most recent call last):
  File "<python-input-88>", line 1, in <module>
    f
NameError: name 'f' is not defined. Did you mean: 'f2'?
>>> f
Traceback (most recent call last):
  File "<python-input-89>", line 1, in <module>
    f
NameError: name 'f' is not defined. Did you mean: 'f2'?
>>> f = 8*(x**2 + y**2) + 4
>>> f2.subs({C:0.1,b:10.0,x:0.5,y:0.0})
6.00000000000000
>>> f.subs({C:0.1,b:10.0,x:0.5,y:0.0})
6.00000000000000
>>> f2.subs({C:0.1,b:10.0,x:0.0,y:0.5})
6.00000000000000
>>> f2.subs({C:0.1,b:10.0,x:0.0,y:-0.5})
6.00000000000000
>>> f2.subs({C:0.1,b:10.0,x:-5.0,y:0})
204.000000000000
>>> f2.subs({C:0.1,b:10.0,x:-.5,y:0})
6.00000000000000
>>> theta=symbol('theta')
Traceback (most recent call last):
  File "<python-input-97>", line 1, in <module>
    theta=symbol('theta')
          ^^^^^^
NameError: name 'symbol' is not defined
>>> theta=sympy.symbol('theta')
Traceback (most recent call last):
  File "<python-input-98>", line 1, in <module>
    theta=sympy.symbol('theta')
          ^^^^^^^^^^^^
AttributeError: module 'sympy' has no attribute 'symbol'
>>> theta=sympy.symbols('theta')
>>> tehta
Traceback (most recent call last):
  File "<python-input-100>", line 1, in <module>
    tehta
NameError: name 'tehta' is not defined. Did you mean: 'theta'?
>>> theta
theta
>>> nx=0.5*sympy.cos(theta)
>>> ny=0.5*sympy.sin(theta)
>>> fj = (beta1*u1.diff(x) - beta2*u2*diff(x))*nx + (beta1*u1.diff(y) - beta2*u2*diff(y))*ny
Traceback (most recent call last):
  File "<python-input-104>", line 1, in <module>
    fj = (beta1*u1.diff(x) - beta2*u2*diff(x))*nx + (beta1*u1.diff(y) - beta2*u2*diff(y))*ny
                                      ^^^^
NameError: name 'diff' is not defined
>>> fj = (beta1*u1.diff(x) - beta2*u2.diff(x))*nx + (beta1*u1.diff(y) - beta2*u2.diff(y))*ny
>>> fj
0.5*(-b*(1.0*C*x/(b*(x**2 + y**2)**1.0) + (2.0*x*(x**2 + y**2)**1.0 + 2.0*x)/b) + 2.0*x*(x**2 + y**2 + 1))*cos(theta) + 0.5*(-b*(1.0*C*y/(b*(x**2 + y**2)**1.0) + (2.0*y*(x**2 + y**2)**1.0 + 2.0*y)/b) + 2.0*y*(x**2 + y**2 + 1))*sin(theta)
>>> sympy.simplify(fj)
-0.5*(x*cos(theta) + y*sin(theta))*(1.0*C + 2.0*(x**2 + y**2)**1.0*((x**2 + y**2)**1.0 + 1) - 2.0*(x**2 + y**2)**1.0*(x**2 + y**2 + 1))/(x**2 + y**2)**1.0
>>> nx=sympy.cos(theta)
>>> ny=sympy.sin(theta)
>>> fj = (beta1*u1.diff(x) - beta2*u2.diff(x))*nx + (beta1*u1.diff(y) - beta2*u2.diff(y))*ny
>>> fj.subs({C:0.1,b:10.0,x:-.5,y:0})
0.2*cos(theta)
>>> fj.subs({C:0.1,b:10.0,x:cos(theta)/2,y:sin(theta)/2})
Traceback (most recent call last):
  File "<python-input-112>", line 1, in <module>
    fj.subs({C:0.1,b:10.0,x:cos(theta)/2,y:sin(theta)/2})
                            ^^^
NameError: name 'cos' is not defined
>>> fj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2})
(-0.05*sin(theta)/(sin(theta)**2/4 + cos(theta)**2/4)**1.0 - 1.0*(sin(theta)**2/4 + cos(theta)**2/4)**1.0*sin(theta) + 1.0*(sin(theta)**2/4 + cos(theta)**2/4 + 1)*sin(theta) - 1.0*sin(theta))*sin(theta) + (-0.05*cos(theta)/(sin(theta)**2/4 + cos(theta)**2/4)**1.0 - 1.0*(sin(theta)**2/4 + cos(theta)**2/4)**1.0*cos(theta) + 1.0*(sin(theta)**2/4 + cos(theta)**2/4 + 1)*cos(theta) - 1.0*cos(theta))*cos(theta)
>>> fj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:0})
-0.200000000000000
>>> fj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:sympy.pi})
-0.200000000000000
>>> fj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:sympy.pi/2})
-0.200000000000000
>>> fj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:sympy.pi/4})
-0.200000000000000
>>> C
C
>>> ((sympy.cos(theta)/2)**2 + (sympy.sin(theta)**2)**(1/2)
... 
... )
(sin(theta)**2)**0.5 + cos(theta)**2/4
>>> tr = ( (sympy.cos(theta)/2)**2 + (sympy.sin(theta)/2)**2)**(1/2)
>>> tr
(sin(theta)**2/4 + cos(theta)**2/4)**0.5
>>> tr.subs({theta:0.0})
0.500000000000000
>>> tr.subs({theta:sympy.pi})
0.500000000000000
>>> tr.subs({theta:sympy.pi/2})
0.500000000000000
>>> fj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:sympy.pi/4})
-0.200000000000000
>>> fj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:sympy.pi/2})
-0.200000000000000
>>> beta1
x**2 + y**2 + 1
>>> beta2
b
>>> u1
(x**2 + y**2)**1.0
>>> u2
C*log(2*(x**2 + y**2)**0.5)/b + 1/4 + ((x**2 + y**2)**1.0 + (x**2 + y**2)**2.0/2)/b - 0.28125/b
>>> uj = u1-u2
>>> uj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:sympy.pi/2})
1.64798730217797e-17
>>> uj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/4}).subs({theta:sympy.pi/4})
-0.0801206849787713
>>> uj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:sympy.pi/4})
1.64798730217797e-17
>>> uj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:sympy.pi/2})
1.64798730217797e-17
>>> uj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:sympy.pi/6})
1.64798730217797e-17
>>> uj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:0})
1.64798730217797e-17
>>> fj
(-b*(1.0*C*x/(b*(x**2 + y**2)**1.0) + (2.0*x*(x**2 + y**2)**1.0 + 2.0*x)/b) + 2.0*x*(x**2 + y**2 + 1))*cos(theta) + (-b*(1.0*C*y/(b*(x**2 + y**2)**1.0) + (2.0*y*(x**2 + y**2)**1.0 + 2.0*y)/b) + 2.0*y*(x**2 + y**2 + 1))*sin(theta)
>>> nx
cos(theta)
>>> ny
sin(theta)
>>> fj = beta1*u1.diff(x)*nx + beta1*u1.diff(y)*ny - beta2*u2.diff(x)*nx - beta2*u2.diff(y)*ny
>>> fj
-b*(1.0*C*x/(b*(x**2 + y**2)**1.0) + (2.0*x*(x**2 + y**2)**1.0 + 2.0*x)/b)*cos(theta) - b*(1.0*C*y/(b*(x**2 + y**2)**1.0) + (2.0*y*(x**2 + y**2)**1.0 + 2.0*y)/b)*sin(theta) + 2.0*x*(x**2 + y**2 + 1)*cos(theta) + 2.0*y*(x**2 + y**2 + 1)*sin(theta)
>>> sympy.simplify(fj)
(-x*(1.0*C + 2.0*(x**2 + y**2)**1.0*((x**2 + y**2)**1.0 + 1))*cos(theta) - y*(1.0*C + 2.0*(x**2 + y**2)**1.0*((x**2 + y**2)**1.0 + 1))*sin(theta) + 2.0*(x**2 + y**2)**1.0*(x*cos(theta) + y*sin(theta))*(x**2 + y**2 + 1))/(x**2 + y**2)**1.0
>>> fj..subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:sympy.pi/6})
  File "<python-input-144>", line 1
    fj..subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:sympy.pi/6})
       ^
SyntaxError: invalid syntax
>>> fj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:sympy.pi/6})
-0.200000000000000
>>> u2
C*log(2*(x**2 + y**2)**0.5)/b + 1/4 + ((x**2 + y**2)**1.0 + (x**2 + y**2)**2.0/2)/b - 0.28125/b
>>> sympy.print(u2)
Traceback (most recent call last):
  File "<python-input-147>", line 1, in <module>
    sympy.print(u2)
    ^^^^^^^^^^^
AttributeError: module 'sympy' has no attribute 'print'
>>> sympy.display(u2)
Traceback (most recent call last):
  File "<python-input-148>", line 1, in <module>
    sympy.display(u2)
    ^^^^^^^^^^^^^
AttributeError: module 'sympy' has no attribute 'display'
>>> u2.pretty()
Traceback (most recent call last):
  File "<python-input-149>", line 1, in <module>
    u2.pretty()
    ^^^^^^^^^
AttributeError: 'Add' object has no attribute 'pretty'
>>> from sympy import init_printing
>>> u2
C*log(2*(x**2 + y**2)**0.5)/b + 1/4 + ((x**2 + y**2)**1.0 + (x**2 + y**2)**2.0/2)/b - 0.28125/b
>>> init_printing()
>>> u2
                                                    2.0          
                                     1.0   ⎛ 2    2⎞             
     ⎛           0.5⎞       ⎛ 2    2⎞      ⎝x  + y ⎠             
     ⎜  ⎛ 2    2⎞   ⎟       ⎝x  + y ⎠    + ────────────          
C⋅log⎝2⋅⎝x  + y ⎠   ⎠   1                       2         0.28125
───────────────────── + ─ + ─────────────────────────── - ───────
          b             4                b                   b   
>>> u2
                                                    2.0          
                                     1.0   ⎛ 2    2⎞             
     ⎛           0.5⎞       ⎛ 2    2⎞      ⎝x  + y ⎠             
     ⎜  ⎛ 2    2⎞   ⎟       ⎝x  + y ⎠    + ────────────          
C⋅log⎝2⋅⎝x  + y ⎠   ⎠   1                       2         0.28125
───────────────────── + ─ + ─────────────────────────── - ───────
          b             4                b                   b   
>>> u1
         1.0
⎛ 2    2⎞   
⎝x  + y ⎠   
>>> uj
                                                                     2.0          
                                                      1.0   ⎛ 2    2⎞             
       ⎛           0.5⎞                      ⎛ 2    2⎞      ⎝x  + y ⎠             
       ⎜  ⎛ 2    2⎞   ⎟            1.0       ⎝x  + y ⎠    + ────────────          
  C⋅log⎝2⋅⎝x  + y ⎠   ⎠   ⎛ 2    2⎞      1                       2         0.28125
- ───────────────────── + ⎝x  + y ⎠    - ─ - ─────────────────────────── + ───────
            b                            4                b                   b   
>>> simplify(uj)
Traceback (most recent call last):
  File "<python-input-157>", line 1, in <module>
    simplify(uj)
    ^^^^^^^^
NameError: name 'simplify' is not defined
>>> sympy.simplify(uj)
                            ⎛           1.0    ⎞                           2.0          
       ⎛           0.5⎞     ⎜  ⎛ 2    2⎞       ⎟            1.0   ⎛ 2    2⎞             
       ⎜  ⎛ 2    2⎞   ⎟   b⋅⎝4⋅⎝x  + y ⎠    - 1⎠   ⎛ 2    2⎞      ⎝x  + y ⎠             
- C⋅log⎝2⋅⎝x  + y ⎠   ⎠ + ────────────────────── - ⎝x  + y ⎠    - ──────────── + 0.28125
                                    4                                  2                
────────────────────────────────────────────────────────────────────────────────────────
                                           b                                            
>>> uj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:0})
1.64798730217797e-17
>>> uj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2})
                       1.0                           2.0           ⎛                     0.5⎞           
    ⎛   2         2   ⎞           ⎛   2         2   ⎞              ⎜  ⎛   2         2   ⎞   ⎟           
    ⎜sin (θ)   cos (θ)⎟           ⎜sin (θ)   cos (θ)⎟              ⎜  ⎜sin (θ)   cos (θ)⎟   ⎟           
0.9⋅⎜─────── + ───────⎟    - 0.05⋅⎜─────── + ───────⎟    - 0.01⋅log⎜2⋅⎜─────── + ───────⎟   ⎟ - 0.221875
    ⎝   4         4   ⎠           ⎝   4         4   ⎠              ⎝  ⎝   4         4   ⎠   ⎠           
>>> sympy.simplify(uj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}))
1.64798730217797e-17
>>> fj
    ⎛                 -1.0                  1.0        ⎞            ⎛                 -1.0                  1.0        ⎞                                                                 
    ⎜        ⎛ 2    2⎞             ⎛ 2    2⎞           ⎟            ⎜        ⎛ 2    2⎞             ⎛ 2    2⎞           ⎟                                                                 
    ⎜1.0⋅C⋅x⋅⎝x  + y ⎠       2.0⋅x⋅⎝x  + y ⎠    + 2.0⋅x⎟            ⎜1.0⋅C⋅y⋅⎝x  + y ⎠       2.0⋅y⋅⎝x  + y ⎠    + 2.0⋅y⎟                ⎛ 2    2    ⎞                ⎛ 2    2    ⎞       
- b⋅⎜───────────────────── + ──────────────────────────⎟⋅cos(θ) - b⋅⎜───────────────────── + ──────────────────────────⎟⋅sin(θ) + 2.0⋅x⋅⎝x  + y  + 1⎠⋅cos(θ) + 2.0⋅y⋅⎝x  + y  + 1⎠⋅sin(θ)
    ⎝          b                         b             ⎠            ⎝          b                         b             ⎠                                                                 
>>> fj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2})
       ⎛                         -1.0                                 1.0                    ⎞               ⎛                         -1.0                                 1.0                    ⎞               ↪
       ⎜      ⎛   2         2   ⎞                  ⎛   2         2   ⎞                       ⎟               ⎜      ⎛   2         2   ⎞                  ⎛   2         2   ⎞                       ⎟               ↪
       ⎜      ⎜sin (θ)   cos (θ)⎟                  ⎜sin (θ)   cos (θ)⎟                       ⎟               ⎜      ⎜sin (θ)   cos (θ)⎟                  ⎜sin (θ)   cos (θ)⎟                       ⎟               ↪
- 10.0⋅⎜0.005⋅⎜─────── + ───────⎟    ⋅sin(θ) + 0.1⋅⎜─────── + ───────⎟   ⋅sin(θ) + 0.1⋅sin(θ)⎟⋅sin(θ) - 10.0⋅⎜0.005⋅⎜─────── + ───────⎟    ⋅cos(θ) + 0.1⋅⎜─────── + ───────⎟   ⋅cos(θ) + 0.1⋅cos(θ)⎟⋅cos(θ) + 1.0⋅ ↪
       ⎝      ⎝   4         4   ⎠                  ⎝   4         4   ⎠                       ⎠               ⎝      ⎝   4         4   ⎠                  ⎝   4         4   ⎠                       ⎠               ↪

↪                                                                      
↪ ⎛   2         2       ⎞               ⎛   2         2       ⎞        
↪ ⎜sin (θ)   cos (θ)    ⎟    2          ⎜sin (θ)   cos (θ)    ⎟    2   
↪ ⎜─────── + ─────── + 1⎟⋅sin (θ) + 1.0⋅⎜─────── + ─────── + 1⎟⋅cos (θ)
↪ ⎝   4         4       ⎠               ⎝   4         4       ⎠        
>>> fj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:0})
-0.200000000000000
>>> fj
    ⎛                 -1.0                  1.0        ⎞            ⎛                 -1.0                  1.0        ⎞                                                                 
    ⎜        ⎛ 2    2⎞             ⎛ 2    2⎞           ⎟            ⎜        ⎛ 2    2⎞             ⎛ 2    2⎞           ⎟                                                                 
    ⎜1.0⋅C⋅x⋅⎝x  + y ⎠       2.0⋅x⋅⎝x  + y ⎠    + 2.0⋅x⎟            ⎜1.0⋅C⋅y⋅⎝x  + y ⎠       2.0⋅y⋅⎝x  + y ⎠    + 2.0⋅y⎟                ⎛ 2    2    ⎞                ⎛ 2    2    ⎞       
- b⋅⎜───────────────────── + ──────────────────────────⎟⋅cos(θ) - b⋅⎜───────────────────── + ──────────────────────────⎟⋅sin(θ) + 2.0⋅x⋅⎝x  + y  + 1⎠⋅cos(θ) + 2.0⋅y⋅⎝x  + y  + 1⎠⋅sin(θ)
    ⎝          b                         b             ⎠            ⎝          b                         b             ⎠                                                                 
>>> fj = -fj
>>> fj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:0})
0.200000000000000
>>> with open("LL94ex2.py') as f:
  File "<python-input-168>", line 1
    with open("LL94ex2.py') as f:
              ^
SyntaxError: unterminated string literal (detected at line 1)
>>> with open("LL94ex2.py",'w') as f:
...     for i in range(readline.get_current_history_length()):
...         f.write(readline.get_history_item(i + 1))
...         
Traceback (most recent call last):
  File "<python-input-169>", line 2, in <module>
    for i in range(readline.get_current_history_length()):
                   ^^^^^^^^
NameError: name 'readline' is not defined. Did you forget to import 'readline'?
>>> import readline
>>> with open("LL94ex2.py",'w') as f:
...     for i in range(readline.get_current_history_length()):
...         f.write(readline.get_history_item(i + 1))
...         
>>> Python 3.13.3 | packaged by conda-forge | (main, Apr 14 2025, 20:44:30) [Clang 18.1.8 ] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import sympy
>>> x,y,b,C=sympy.symbols('x,y,b,C')
>>> r = (x**2 + y**2)**(1/2)
>>> dif(r,x)
Traceback (most recent call last):
  File "<python-input-3>", line 1, in <module>
    dif(r,x)
    ^^^
NameError: name 'dif' is not defined. Did you mean: 'dir'?
>>> sympy.diff(r,x)
1.0*x/(x**2 + y**2)**0.5
>>> u1=r**2
>>> u2=(1-1/8/b-1/b)/4+(r**4/2+r**2)/b + C*log(2*r)/b
Traceback (most recent call last):
  File "<python-input-6>", line 1, in <module>
    u2=(1-1/8/b-1/b)/4+(r**4/2+r**2)/b + C*log(2*r)/b
                                           ^^^
NameError: name 'log' is not defined
>>> u2=(1-1/8/b-1/b)/4+(r**4/2+r**2)/b + C*sympy.log(2*r)/b
>>> u1x=diff(u1,x)
Traceback (most recent call last):
  File "<python-input-8>", line 1, in <module>
    u1x=diff(u1,x)
        ^^^^
NameError: name 'diff' is not defined
>>> u1x=u1.diff(x)
>>> u1x=u1.diff(y)
>>> u1y=u1.diff(y)
>>> u2y=u2.diff(y)
>>> u2x=u2.diff(x)
>>> sympy.simplifiy(u1x-u2x)
Traceback (most recent call last):
  File "<python-input-14>", line 1, in <module>
    sympy.simplifiy(u1x-u2x)
    ^^^^^^^^^^^^^^^
AttributeError: module 'sympy' has no attribute 'simplifiy'
>>> sympy.simplify(u1x-u2x)
-1.0*C*x/(b*(x**2 + y**2)**1.0) + 2.0*y - 2.0*x*(x**2 + y**2)**1.0/b - 2.0*x/b
>>> sympy.simplify(u1y-u2y)
-1.0*C*y/(b*(x**2 + y**2)**1.0) + 2.0*y - 2.0*y*(x**2 + y**2)**1.0/b - 2.0*y/b
>>> sympy.simplify(u1y.eval({x:0.5,y:0.0})-u2y.eval({x:0.5,y:0.0}))
Traceback (most recent call last):
  File "<python-input-17>", line 1, in <module>
    sympy.simplify(u1y.eval({x:0.5,y:0.0})-u2y.eval({x:0.5,y:0.0}))
                   ^^^^^^^^
AttributeError: 'Mul' object has no attribute 'eval'. Did you mean: 'evalf'?
>>> u1x
2.0*y
>>> u1y
2.0*y
>>> u2x
1.0*C*x/(b*(x**2 + y**2)**1.0) + (2.0*x*(x**2 + y**2)**1.0 + 2.0*x)/b
>>> u2y
1.0*C*y/(b*(x**2 + y**2)**1.0) + (2.0*y*(x**2 + y**2)**1.0 + 2.0*y)/b
>>> u2
C*log(2*(x**2 + y**2)**0.5)/b + 1/4 + ((x**2 + y**2)**1.0 + (x**2 + y**2)**2.0/2)/b - 0.28125/b
>>> u2
C*log(2*(x**2 + y**2)**0.5)/b + 1/4 + ((x**2 + y**2)**1.0 + (x**2 + y**2)**2.0/2)/b - 0.28125/b
>>> 1/32
0.03125
>>> 4/32
0.125
>>> b
b
>>> u2=(1-1/8/b-1/b)/4+((r**4)/2+r**2)/b + C*sympy.log(2*r)/b
>>> u2x=u2.diff(x)
>>> u2y=u2.diff(y)
>>> sympy.simplify(u1y.eval({x:0.5,y:0.0})-u2y.eval({x:0.5,y:0.0}))
Traceback (most recent call last):
  File "<python-input-30>", line 1, in <module>
    sympy.simplify(u1y.eval({x:0.5,y:0.0})-u2y.eval({x:0.5,y:0.0}))
                   ^^^^^^^^
AttributeError: 'Mul' object has no attribute 'eval'. Did you mean: 'evalf'?
>>> u1x=u1.diff(x)
>>> u2x=u2.diff(x)
>>> u2y=u2.diff(y)
>>> u1y=u1.diff(y)
>>> sympy.simplify(u1y.eval({x:0.5,y:0.0})-u2y.eval({x:0.5,y:0.0}))
Traceback (most recent call last):
  File "<python-input-35>", line 1, in <module>
    sympy.simplify(u1y.eval({x:0.5,y:0.0})-u2y.eval({x:0.5,y:0.0}))
                   ^^^^^^^^
AttributeError: 'Mul' object has no attribute 'eval'. Did you mean: 'evalf'?
>>> sympy.simplify(u1y.subs({x:0.5,y:0.0})-u2y.subs({x:0.5,y:0.0}))
0
>>> sympy.simplify(u1y.subs({x:0.0,y:0.5})-u2y.subs({x:0.0,y:0.5}))
(-2.0*C + 1.0*b - 1.25)/b
>>> sympy.simplify(u1y.subs({x:0.0,y:0.})-u2y.subs({x:0.0,y:0.}))
nan
>>> sympy.simplify(u1y.subs({x:0.0,y:0.5})-u2y.subs({x:0.0,y:0.5}))
(-2.0*C + 1.0*b - 1.25)/b
>>> sympy.simplify(u1y.subs({x:0.5,y:0.})-u2y.subs({x:0.5,y:0.}))
0
>>> u1
(x**2 + y**2)**1.0
>>> u2
C*log(2*(x**2 + y**2)**0.5)/b + 1/4 + ((x**2 + y**2)**1.0 + (x**2 + y**2)**2.0/2)/b - 0.28125/b
>>> C
C
>>> b
b
>>> u2=(1-1/8/b-1/b)/4+((r**4)/2+r**2)/b + C*sympy.log(2*r)/b
>>> u2
C*log(2*(x**2 + y**2)**0.5)/b + 1/4 + ((x**2 + y**2)**1.0 + (x**2 + y**2)**2.0/2)/b - 0.28125/b
>>> 1/32 + 1/4
0.28125
>>> u1.diff(x) - u2.diff(x)
-1.0*C*x/(b*(x**2 + y**2)**1.0) + 2.0*x - (2.0*x*(x**2 + y**2)**1.0 + 2.0*x)/b
>>> (u1.diff(x) - u2.diff(x)).subs({b:10,x:0.5,y:0.0})
0.875 - 0.2*C
>>> (u1.diff(x) - u2.diff(x)).subs({b:10,x:0.,y:0.5})
0
>>> beta1 = x**2 + y**2 + 1
>>> beta2 = b
>>> (b1*u1.diff(x) - b2*u2.diff(x)).subs({b:10,x:0.,y:0.5})
Traceback (most recent call last):
  File "<python-input-53>", line 1, in <module>
    (b1*u1.diff(x) - b2*u2.diff(x)).subs({b:10,x:0.,y:0.5})
     ^^
NameError: name 'b1' is not defined. Did you mean: 'b'?
>>> (beta1*u1.diff(x) - beta2*u2.diff(x)).subs({b:10,x:0.,y:0.5})
0
>>> beta
Traceback (most recent call last):
  File "<python-input-55>", line 1, in <module>
    beta
NameError: name 'beta' is not defined. Did you mean: 'beta1'?
>>> beta1
x**2 + y**2 + 1
>>> (beta*u1.diff(x)).diff(x) + (beta*u1.diff(y)).diff(y)
Traceback (most recent call last):
  File "<python-input-57>", line 1, in <module>
    (beta*u1.diff(x)).diff(x) + (beta*u1.diff(y)).diff(y)
     ^^^^
NameError: name 'beta' is not defined. Did you mean: 'beta1'?
>>> (beta1*u1.diff(x)).diff(x) + (beta1*u1.diff(y)).diff(y)
8.0*x**2 + 8.0*y**2 + 4.0
>>> (beta1*u2.diff(x)).diff(x) + (beta2*u2.diff(y)).diff(y)
b*(-2.0*C*y**2/(b*(x**2 + y**2)**2.0) + 1.0*C/(b*(x**2 + y**2)**1.0) + (4.0*y**2 + 2.0*(x**2 + y**2)**1.0 + 2.0)/b) + 2*x*(1.0*C*x/(b*(x**2 + y**2)**1.0) + (2.0*x*(x**2 + y**2)**1.0 + 2.0*x)/b) + (x**2 + y**2 + 1)*(-2.0*C*x**2/(b*(x**2 + y**2)**2.0) + 1.0*C/(b*(x**2 + y**2)**1.0) + (4.0*x**2 + 2.0*(x**2 + y**2)**1.0 + 2.0)/b)
>>> sympy.simplify((beta1*u2.diff(x)).diff(x) + (beta2*u2.diff(y)).diff(y))
(b*(x**2 + y**2)**4.0*(-2.0*C*y**2*(x**2 + y**2)**1.0 + 1.0*C*(x**2 + y**2)**2.0 + (x**2 + y**2)**3.0*(4.0*y**2 + 2.0*(x**2 + y**2)**1.0 + 2.0)) + x**2*(2.0*C + (x**2 + y**2)**1.0*(4.0*(x**2 + y**2)**1.0 + 4.0))*(x**2 + y**2)**6.0 + (x**2 + y**2)**4.0*(x**2 + y**2 + 1)*(-2.0*C*x**2*(x**2 + y**2)**1.0 + 1.0*C*(x**2 + y**2)**2.0 + (x**2 + y**2)**3.0*(4.0*x**2 + 2.0*(x**2 + y**2)**1.0 + 2.0)))/(b*(x**2 + y**2)**7.0)
>>> beta2
b
>>> u2
C*log(2*(x**2 + y**2)**0.5)/b + 1/4 + ((x**2 + y**2)**1.0 + (x**2 + y**2)**2.0/2)/b - 0.28125/b
>>> sympy.simplify((beta2*u2.diff(x)).diff(x) + (beta2*u2.diff(y)).diff(y))
-2.0*C*x**2/(x**2 + y**2)**2.0 - 2.0*C*y**2/(x**2 + y**2)**2.0 + 2.0*C/(x**2 + y**2)**1.0 + 4.0*x**2 + 4.0*y**2 + 4.0*(x**2 + y**2)**1.0 + 4.0
>>> sympy.simplify((beta2*u2.diff(x)).diff(x) + (beta2*u2.diff(y)).diff(y)).subs({b:10,C=1/10})
  File "<python-input-64>", line 1
    sympy.simplify((beta2*u2.diff(x)).diff(x) + (beta2*u2.diff(y)).diff(y)).subs({b:10,C=1/10})
                                                                                       ^
SyntaxError: ':' expected after dictionary key
>>> sympy.simplify((beta2*u2.diff(x)).diff(x) + (beta2*u2.diff(y)).diff(y)).subs({b:10,C:1/10})
-0.2*x**2/(x**2 + y**2)**2.0 + 4.0*x**2 - 0.2*y**2/(x**2 + y**2)**2.0 + 4.0*y**2 + 0.2/(x**2 + y**2)**1.0 + 4.0*(x**2 + y**2)**1.0 + 4.0
>>> u2
C*log(2*(x**2 + y**2)**0.5)/b + 1/4 + ((x**2 + y**2)**1.0 + (x**2 + y**2)**2.0/2)/b - 0.28125/b
>>> beta1
x**2 + y**2 + 1
>>> beta2
b
>>> sympy.simplify((beta2*u2.diff(x)).diff(x) + (beta2*u2.diff(y)).diff(y))
-2.0*C*x**2/(x**2 + y**2)**2.0 - 2.0*C*y**2/(x**2 + y**2)**2.0 + 2.0*C/(x**2 + y**2)**1.0 + 4.0*x**2 + 4.0*y**2 + 4.0*(x**2 + y**2)**1.0 + 4.0
>>> sympy.simplify((beta1*u1.diff(x)).diff(x) + (beta1*u1.diff(y)).diff(y))
8.0*x**2 + 8.0*y**2 + 4.0
>>> sympy.simplify((beta1*u1.diff(x)).diff(x) + (beta1*u1.diff(y)).diff(y))
8.0*x**2 + 8.0*y**2 + 4.0
>>> sympy.simplify((beta2*u2.diff(x)).diff(x) + (beta2*u2.diff(y)).diff(y))
-2.0*C*x**2/(x**2 + y**2)**2.0 - 2.0*C*y**2/(x**2 + y**2)**2.0 + 2.0*C/(x**2 + y**2)**1.0 + 4.0*x**2 + 4.0*y**2 + 4.0*(x**2 + y**2)**1.0 + 4.0
>>> sympy.simplify((beta2*u2.diff(x)).diff(x) + (beta2*u2.diff(y)).diff(y) - 8*(x**2 + y**2) + 4)
-2.0*C*x**2/(x**2 + y**2)**2.0 - 2.0*C*y**2/(x**2 + y**2)**2.0 + 2.0*C/(x**2 + y**2)**1.0 - 4.0*x**2 - 4.0*y**2 + 4.0*(x**2 + y**2)**1.0 + 8.0
>>> u2
C*log(2*(x**2 + y**2)**0.5)/b + 1/4 + ((x**2 + y**2)**1.0 + (x**2 + y**2)**2.0/2)/b - 0.28125/b
>>> beta2
b
>>> beta2.diff(x)
0
>>> beta2
b
>>> u2.diff(x)
1.0*C*x/(b*(x**2 + y**2)**1.0) + (2.0*x*(x**2 + y**2)**1.0 + 2.0*x)/b
>>> u2.diff(x).diff(x)
-2.0*C*x**2/(b*(x**2 + y**2)**2.0) + 1.0*C/(b*(x**2 + y**2)**1.0) + (4.0*x**2 + 2.0*(x**2 + y**2)**1.0 + 2.0)/b
>>> u2.diff(x).diff(x)*b
b*(-2.0*C*x**2/(b*(x**2 + y**2)**2.0) + 1.0*C/(b*(x**2 + y**2)**1.0) + (4.0*x**2 + 2.0*(x**2 + y**2)**1.0 + 2.0)/b)
>>> u
Traceback (most recent call last):
  File "<python-input-81>", line 1, in <module>
    u
NameError: name 'u' is not defined. Did you mean: 'u1'?
>>> u2
C*log(2*(x**2 + y**2)**0.5)/b + 1/4 + ((x**2 + y**2)**1.0 + (x**2 + y**2)**2.0/2)/b - 0.28125/b
>>> f2 = (beta2*u2.diff(x)).diff(xu) + (beta2*u2.diff(x)).diff(x)
Traceback (most recent call last):
  File "<python-input-83>", line 1, in <module>
    f2 = (beta2*u2.diff(x)).diff(xu) + (beta2*u2.diff(x)).diff(x)
                                 ^^
NameError: name 'xu' is not defined. Did you mean: 'x'?
>>> f2 = (beta2*u2.diff(x)).diff(x) + (beta2*u2.diff(t)).diff(t)
Traceback (most recent call last):
  File "<python-input-84>", line 1, in <module>
    f2 = (beta2*u2.diff(x)).diff(x) + (beta2*u2.diff(t)).diff(t)
                                                     ^
NameError: name 't' is not defined
>>> f2 = (beta2*u2.diff(x)).diff(x) + (beta2*u2.diff(y)).diff(y)
>>> f2
b*(-2.0*C*x**2/(b*(x**2 + y**2)**2.0) + 1.0*C/(b*(x**2 + y**2)**1.0) + (4.0*x**2 + 2.0*(x**2 + y**2)**1.0 + 2.0)/b) + b*(-2.0*C*y**2/(b*(x**2 + y**2)**2.0) + 1.0*C/(b*(x**2 + y**2)**1.0) + (4.0*y**2 + 2.0*(x**2 + y**2)**1.0 + 2.0)/b)
>>> f2.subs({C:0.1,b:10.0,x:0.5,y:0.0})
6.00000000000000
>>> f
Traceback (most recent call last):
  File "<python-input-88>", line 1, in <module>
    f
NameError: name 'f' is not defined. Did you mean: 'f2'?
>>> f
Traceback (most recent call last):
  File "<python-input-89>", line 1, in <module>
    f
NameError: name 'f' is not defined. Did you mean: 'f2'?
>>> f = 8*(x**2 + y**2) + 4
>>> f2.subs({C:0.1,b:10.0,x:0.5,y:0.0})
6.00000000000000
>>> f.subs({C:0.1,b:10.0,x:0.5,y:0.0})
6.00000000000000
>>> f2.subs({C:0.1,b:10.0,x:0.0,y:0.5})
6.00000000000000
>>> f2.subs({C:0.1,b:10.0,x:0.0,y:-0.5})
6.00000000000000
>>> f2.subs({C:0.1,b:10.0,x:-5.0,y:0})
204.000000000000
>>> f2.subs({C:0.1,b:10.0,x:-.5,y:0})
6.00000000000000
>>> theta=symbol('theta')
Traceback (most recent call last):
  File "<python-input-97>", line 1, in <module>
    theta=symbol('theta')
          ^^^^^^
NameError: name 'symbol' is not defined
>>> theta=sympy.symbol('theta')
Traceback (most recent call last):
  File "<python-input-98>", line 1, in <module>
    theta=sympy.symbol('theta')
          ^^^^^^^^^^^^
AttributeError: module 'sympy' has no attribute 'symbol'
>>> theta=sympy.symbols('theta')
>>> tehta
Traceback (most recent call last):
  File "<python-input-100>", line 1, in <module>
    tehta
NameError: name 'tehta' is not defined. Did you mean: 'theta'?
>>> theta
theta
>>> nx=0.5*sympy.cos(theta)
>>> ny=0.5*sympy.sin(theta)
>>> fj = (beta1*u1.diff(x) - beta2*u2*diff(x))*nx + (beta1*u1.diff(y) - beta2*u2*diff(y))*ny
Traceback (most recent call last):
  File "<python-input-104>", line 1, in <module>
    fj = (beta1*u1.diff(x) - beta2*u2*diff(x))*nx + (beta1*u1.diff(y) - beta2*u2*diff(y))*ny
                                      ^^^^
NameError: name 'diff' is not defined
>>> fj = (beta1*u1.diff(x) - beta2*u2.diff(x))*nx + (beta1*u1.diff(y) - beta2*u2.diff(y))*ny
>>> fj
0.5*(-b*(1.0*C*x/(b*(x**2 + y**2)**1.0) + (2.0*x*(x**2 + y**2)**1.0 + 2.0*x)/b) + 2.0*x*(x**2 + y**2 + 1))*cos(theta) + 0.5*(-b*(1.0*C*y/(b*(x**2 + y**2)**1.0) + (2.0*y*(x**2 + y**2)**1.0 + 2.0*y)/b) + 2.0*y*(x**2 + y**2 + 1))*sin(theta)
>>> sympy.simplify(fj)
-0.5*(x*cos(theta) + y*sin(theta))*(1.0*C + 2.0*(x**2 + y**2)**1.0*((x**2 + y**2)**1.0 + 1) - 2.0*(x**2 + y**2)**1.0*(x**2 + y**2 + 1))/(x**2 + y**2)**1.0
>>> nx=sympy.cos(theta)
>>> ny=sympy.sin(theta)
>>> fj = (beta1*u1.diff(x) - beta2*u2.diff(x))*nx + (beta1*u1.diff(y) - beta2*u2.diff(y))*ny
>>> fj.subs({C:0.1,b:10.0,x:-.5,y:0})
0.2*cos(theta)
>>> fj.subs({C:0.1,b:10.0,x:cos(theta)/2,y:sin(theta)/2})
Traceback (most recent call last):
  File "<python-input-112>", line 1, in <module>
    fj.subs({C:0.1,b:10.0,x:cos(theta)/2,y:sin(theta)/2})
                            ^^^
NameError: name 'cos' is not defined
>>> fj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2})
(-0.05*sin(theta)/(sin(theta)**2/4 + cos(theta)**2/4)**1.0 - 1.0*(sin(theta)**2/4 + cos(theta)**2/4)**1.0*sin(theta) + 1.0*(sin(theta)**2/4 + cos(theta)**2/4 + 1)*sin(theta) - 1.0*sin(theta))*sin(theta) + (-0.05*cos(theta)/(sin(theta)**2/4 + cos(theta)**2/4)**1.0 - 1.0*(sin(theta)**2/4 + cos(theta)**2/4)**1.0*cos(theta) + 1.0*(sin(theta)**2/4 + cos(theta)**2/4 + 1)*cos(theta) - 1.0*cos(theta))*cos(theta)
>>> fj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:0})
-0.200000000000000
>>> fj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:sympy.pi})
-0.200000000000000
>>> fj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:sympy.pi/2})
-0.200000000000000
>>> fj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:sympy.pi/4})
-0.200000000000000
>>> C
C
>>> ((sympy.cos(theta)/2)**2 + (sympy.sin(theta)**2)**(1/2)
... 
... )
(sin(theta)**2)**0.5 + cos(theta)**2/4
>>> tr = ( (sympy.cos(theta)/2)**2 + (sympy.sin(theta)/2)**2)**(1/2)
>>> tr
(sin(theta)**2/4 + cos(theta)**2/4)**0.5
>>> tr.subs({theta:0.0})
0.500000000000000
>>> tr.subs({theta:sympy.pi})
0.500000000000000
>>> tr.subs({theta:sympy.pi/2})
0.500000000000000
>>> fj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:sympy.pi/4})
-0.200000000000000
>>> fj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:sympy.pi/2})
-0.200000000000000
>>> beta1
x**2 + y**2 + 1
>>> beta2
b
>>> u1
(x**2 + y**2)**1.0
>>> u2
C*log(2*(x**2 + y**2)**0.5)/b + 1/4 + ((x**2 + y**2)**1.0 + (x**2 + y**2)**2.0/2)/b - 0.28125/b
>>> uj = u1-u2
>>> uj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:sympy.pi/2})
1.64798730217797e-17
>>> uj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/4}).subs({theta:sympy.pi/4})
-0.0801206849787713
>>> uj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:sympy.pi/4})
1.64798730217797e-17
>>> uj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:sympy.pi/2})
1.64798730217797e-17
>>> uj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:sympy.pi/6})
1.64798730217797e-17
>>> uj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:0})
1.64798730217797e-17
>>> fj
(-b*(1.0*C*x/(b*(x**2 + y**2)**1.0) + (2.0*x*(x**2 + y**2)**1.0 + 2.0*x)/b) + 2.0*x*(x**2 + y**2 + 1))*cos(theta) + (-b*(1.0*C*y/(b*(x**2 + y**2)**1.0) + (2.0*y*(x**2 + y**2)**1.0 + 2.0*y)/b) + 2.0*y*(x**2 + y**2 + 1))*sin(theta)
>>> nx
cos(theta)
>>> ny
sin(theta)
>>> fj = beta1*u1.diff(x)*nx + beta1*u1.diff(y)*ny - beta2*u2.diff(x)*nx - beta2*u2.diff(y)*ny
>>> fj
-b*(1.0*C*x/(b*(x**2 + y**2)**1.0) + (2.0*x*(x**2 + y**2)**1.0 + 2.0*x)/b)*cos(theta) - b*(1.0*C*y/(b*(x**2 + y**2)**1.0) + (2.0*y*(x**2 + y**2)**1.0 + 2.0*y)/b)*sin(theta) + 2.0*x*(x**2 + y**2 + 1)*cos(theta) + 2.0*y*(x**2 + y**2 + 1)*sin(theta)
>>> sympy.simplify(fj)
(-x*(1.0*C + 2.0*(x**2 + y**2)**1.0*((x**2 + y**2)**1.0 + 1))*cos(theta) - y*(1.0*C + 2.0*(x**2 + y**2)**1.0*((x**2 + y**2)**1.0 + 1))*sin(theta) + 2.0*(x**2 + y**2)**1.0*(x*cos(theta) + y*sin(theta))*(x**2 + y**2 + 1))/(x**2 + y**2)**1.0
>>> fj..subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:sympy.pi/6})
  File "<python-input-144>", line 1
    fj..subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:sympy.pi/6})
       ^
SyntaxError: invalid syntax
>>> fj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:sympy.pi/6})
-0.200000000000000
>>> u2
C*log(2*(x**2 + y**2)**0.5)/b + 1/4 + ((x**2 + y**2)**1.0 + (x**2 + y**2)**2.0/2)/b - 0.28125/b
>>> sympy.print(u2)
Traceback (most recent call last):
  File "<python-input-147>", line 1, in <module>
    sympy.print(u2)
    ^^^^^^^^^^^
AttributeError: module 'sympy' has no attribute 'print'
>>> sympy.display(u2)
Traceback (most recent call last):
  File "<python-input-148>", line 1, in <module>
    sympy.display(u2)
    ^^^^^^^^^^^^^
AttributeError: module 'sympy' has no attribute 'display'
>>> u2.pretty()
Traceback (most recent call last):
  File "<python-input-149>", line 1, in <module>
    u2.pretty()
    ^^^^^^^^^
AttributeError: 'Add' object has no attribute 'pretty'
>>> from sympy import init_printing
>>> u2
C*log(2*(x**2 + y**2)**0.5)/b + 1/4 + ((x**2 + y**2)**1.0 + (x**2 + y**2)**2.0/2)/b - 0.28125/b
>>> init_printing()
>>> u2
                                                    2.0          
                                     1.0   ⎛ 2    2⎞             
     ⎛           0.5⎞       ⎛ 2    2⎞      ⎝x  + y ⎠             
     ⎜  ⎛ 2    2⎞   ⎟       ⎝x  + y ⎠    + ────────────          
C⋅log⎝2⋅⎝x  + y ⎠   ⎠   1                       2         0.28125
───────────────────── + ─ + ─────────────────────────── - ───────
          b             4                b                   b   
>>> u2
                                                    2.0          
                                     1.0   ⎛ 2    2⎞             
     ⎛           0.5⎞       ⎛ 2    2⎞      ⎝x  + y ⎠             
     ⎜  ⎛ 2    2⎞   ⎟       ⎝x  + y ⎠    + ────────────          
C⋅log⎝2⋅⎝x  + y ⎠   ⎠   1                       2         0.28125
───────────────────── + ─ + ─────────────────────────── - ───────
          b             4                b                   b   
>>> u1
         1.0
⎛ 2    2⎞   
⎝x  + y ⎠   
>>> uj
                                                                     2.0          
                                                      1.0   ⎛ 2    2⎞             
       ⎛           0.5⎞                      ⎛ 2    2⎞      ⎝x  + y ⎠             
       ⎜  ⎛ 2    2⎞   ⎟            1.0       ⎝x  + y ⎠    + ────────────          
  C⋅log⎝2⋅⎝x  + y ⎠   ⎠   ⎛ 2    2⎞      1                       2         0.28125
- ───────────────────── + ⎝x  + y ⎠    - ─ - ─────────────────────────── + ───────
            b                            4                b                   b   
>>> simplify(uj)
Traceback (most recent call last):
  File "<python-input-157>", line 1, in <module>
    simplify(uj)
    ^^^^^^^^
NameError: name 'simplify' is not defined
>>> sympy.simplify(uj)
                            ⎛           1.0    ⎞                           2.0          
       ⎛           0.5⎞     ⎜  ⎛ 2    2⎞       ⎟            1.0   ⎛ 2    2⎞             
       ⎜  ⎛ 2    2⎞   ⎟   b⋅⎝4⋅⎝x  + y ⎠    - 1⎠   ⎛ 2    2⎞      ⎝x  + y ⎠             
- C⋅log⎝2⋅⎝x  + y ⎠   ⎠ + ────────────────────── - ⎝x  + y ⎠    - ──────────── + 0.28125
                                    4                                  2                
────────────────────────────────────────────────────────────────────────────────────────
                                           b                                            
>>> uj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:0})
1.64798730217797e-17
>>> uj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2})
                       1.0                           2.0           ⎛                     0.5⎞           
    ⎛   2         2   ⎞           ⎛   2         2   ⎞              ⎜  ⎛   2         2   ⎞   ⎟           
    ⎜sin (θ)   cos (θ)⎟           ⎜sin (θ)   cos (θ)⎟              ⎜  ⎜sin (θ)   cos (θ)⎟   ⎟           
0.9⋅⎜─────── + ───────⎟    - 0.05⋅⎜─────── + ───────⎟    - 0.01⋅log⎜2⋅⎜─────── + ───────⎟   ⎟ - 0.221875
    ⎝   4         4   ⎠           ⎝   4         4   ⎠              ⎝  ⎝   4         4   ⎠   ⎠           
>>> sympy.simplify(uj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}))
1.64798730217797e-17
>>> fj
    ⎛                 -1.0                  1.0        ⎞            ⎛                 -1.0                  1.0        ⎞                                                                 
    ⎜        ⎛ 2    2⎞             ⎛ 2    2⎞           ⎟            ⎜        ⎛ 2    2⎞             ⎛ 2    2⎞           ⎟                                                                 
    ⎜1.0⋅C⋅x⋅⎝x  + y ⎠       2.0⋅x⋅⎝x  + y ⎠    + 2.0⋅x⎟            ⎜1.0⋅C⋅y⋅⎝x  + y ⎠       2.0⋅y⋅⎝x  + y ⎠    + 2.0⋅y⎟                ⎛ 2    2    ⎞                ⎛ 2    2    ⎞       
- b⋅⎜───────────────────── + ──────────────────────────⎟⋅cos(θ) - b⋅⎜───────────────────── + ──────────────────────────⎟⋅sin(θ) + 2.0⋅x⋅⎝x  + y  + 1⎠⋅cos(θ) + 2.0⋅y⋅⎝x  + y  + 1⎠⋅sin(θ)
    ⎝          b                         b             ⎠            ⎝          b                         b             ⎠                                                                 
>>> fj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2})
       ⎛                         -1.0                                 1.0                    ⎞               ⎛                         -1.0                                 1.0                    ⎞               ↪
       ⎜      ⎛   2         2   ⎞                  ⎛   2         2   ⎞                       ⎟               ⎜      ⎛   2         2   ⎞                  ⎛   2         2   ⎞                       ⎟               ↪
       ⎜      ⎜sin (θ)   cos (θ)⎟                  ⎜sin (θ)   cos (θ)⎟                       ⎟               ⎜      ⎜sin (θ)   cos (θ)⎟                  ⎜sin (θ)   cos (θ)⎟                       ⎟               ↪
- 10.0⋅⎜0.005⋅⎜─────── + ───────⎟    ⋅sin(θ) + 0.1⋅⎜─────── + ───────⎟   ⋅sin(θ) + 0.1⋅sin(θ)⎟⋅sin(θ) - 10.0⋅⎜0.005⋅⎜─────── + ───────⎟    ⋅cos(θ) + 0.1⋅⎜─────── + ───────⎟   ⋅cos(θ) + 0.1⋅cos(θ)⎟⋅cos(θ) + 1.0⋅ ↪
       ⎝      ⎝   4         4   ⎠                  ⎝   4         4   ⎠                       ⎠               ⎝      ⎝   4         4   ⎠                  ⎝   4         4   ⎠                       ⎠               ↪

↪                                                                      
↪ ⎛   2         2       ⎞               ⎛   2         2       ⎞        
↪ ⎜sin (θ)   cos (θ)    ⎟    2          ⎜sin (θ)   cos (θ)    ⎟    2   
↪ ⎜─────── + ─────── + 1⎟⋅sin (θ) + 1.0⋅⎜─────── + ─────── + 1⎟⋅cos (θ)
↪ ⎝   4         4       ⎠               ⎝   4         4       ⎠        
>>> fj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:0})
-0.200000000000000
>>> fj
    ⎛                 -1.0                  1.0        ⎞            ⎛                 -1.0                  1.0        ⎞                                                                 
    ⎜        ⎛ 2    2⎞             ⎛ 2    2⎞           ⎟            ⎜        ⎛ 2    2⎞             ⎛ 2    2⎞           ⎟                                                                 
    ⎜1.0⋅C⋅x⋅⎝x  + y ⎠       2.0⋅x⋅⎝x  + y ⎠    + 2.0⋅x⎟            ⎜1.0⋅C⋅y⋅⎝x  + y ⎠       2.0⋅y⋅⎝x  + y ⎠    + 2.0⋅y⎟                ⎛ 2    2    ⎞                ⎛ 2    2    ⎞       
- b⋅⎜───────────────────── + ──────────────────────────⎟⋅cos(θ) - b⋅⎜───────────────────── + ──────────────────────────⎟⋅sin(θ) + 2.0⋅x⋅⎝x  + y  + 1⎠⋅cos(θ) + 2.0⋅y⋅⎝x  + y  + 1⎠⋅sin(θ)
    ⎝          b                         b             ⎠            ⎝          b                         b             ⎠                                                                 
>>> fj = -fj
>>> fj.subs({C:0.1,b:10.0,x:sympy.cos(theta)/2,y:sympy.sin(theta)/2}).subs({theta:0})
0.200000000000000
>>> with open("LL94ex2.py') as f:
  File "<python-input-168>", line 1
    with open("LL94ex2.py') as f:
              ^
SyntaxError: unterminated string literal (detected at line 1)
>>> with open("LL94ex2.py",'w') as f:
...     for i in range(readline.get_current_history_length()):
...         f.write(readline.get_history_item(i + 1))
...         
Traceback (most recent call last):
  File "<python-input-169>", line 2, in <module>
    for i in range(readline.get_current_history_length()):
                   ^^^^^^^^
NameError: name 'readline' is not defined. Did you forget to import 'readline'?
>>> import readline
>>> with open("LL94ex2.py",'w') as f:
...     for i in range(readline.get_current_history_length()):
...         f.write(readline.get_history_item(i + 1))
...         
>>> 
