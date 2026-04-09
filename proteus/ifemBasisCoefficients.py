from symengine import symbols, diff, linsolve, Rational

# ------------------------------------------------------------------
# Solver
# ------------------------------------------------------------------
x,y,a1,a2,a3,a4,a5,a6,b1,b2,b3,b4,b5,b6 = symbols(
    'x y a1 a2 a3 a4 a5 a6 b1 b2 b3 b4 b5 b6'
)
# v1,v2,v3,v4,v5,v6 = symbols('v1 v2 v3 v4 v5 v6')
# nx,ny,x0,y0,ma,mb = symbols('nx ny x0 y0 ma mb')
# fax,fay,fbx,fby = symbols('fax fay fbx fby')
# Jit00,Jit01,Jit10,Jit11 = symbols('Jit00 Jit01 Jit10 Jit11')
def solveCoefficients(ifem_basis_order, x0, y0, nx, ny, ma, mb, jf, Jit00, Jit01, Jit10, Jit11, nodal_values):

    """
    Solve for IFEM basis coefficients and return as numeric array.
    
    Parameters:
    -----------
    ifem_basis_order : int
        Order of the IFEM basis (1 for P1 linear, 2 for P2 quadratic)
    x0, y0 : float
        Interface location parameters
    nx, ny : float
        Interface normal components
    ma, mb : float
        Material parameters
    jf : float
        Flux jump parameter
    Jit00, Jit01, Jit10, Jit11 : float
        Jacobian inverse transpose components
    nodal_values : list or tuple
        Nodal values to constrain the basis
    
    Returns:
    --------
    list
        Coefficients for the IFEM basis [a1,...,b1,...]
    """
    
    if ifem_basis_order == 1:
        return _solveCoefficients_P1(x0, y0, nx, ny, ma, mb, jf, Jit00, Jit01, Jit10, Jit11, nodal_values)
    elif ifem_basis_order == 2:
        return _solveCoefficients_P2(x0, y0, nx, ny, ma, mb, jf, Jit00, Jit01, Jit10, Jit11, nodal_values)
    else:
        raise ValueError(f"Unsupported IFEM basis order: {ifem_basis_order}")


def _solveCoefficients_P1(x0, y0, nx, ny, ma, mb, jf, Jit00, Jit01, Jit10, Jit11, nodal_values):
    """
    Solve for P1 linear basis coefficients.
    Returns a list of 6 coefficients [a1,a2,a3,b1,b2,b3].
    """
        
    # ------------------------------------------------------------------
    # Nodal values
    # ------------------------------------------------------------------
    v1, v2, v3 = nodal_values[:3]
    
    # ------------------------------------------------------------------
    # Linear basis (P1)
    # ------------------------------------------------------------------
    va = a1 + a2*x + a3*y
    vb = b1 + b2*x + b3*y

    # First derivatives
    va_x = diff(va, x)  # a2
    va_y = diff(va, y)  # a3
    vb_x = diff(vb, x)  # b2
    vb_y = diff(vb, y)  # b3

    # ------------------------------------------------------------------
    # Constraints (use exact rationals!)
    # ------------------------------------------------------------------
    # Point constraints
    c1 = va.subs({x:0, y:0}) - v1
    c2 = vb.subs({x:1, y:0}) - v2
    c3 = vb.subs({x:0, y:1}) - v3
    
    # Basis continuity at interface
    c4 = vb.subs({x:x0, y:0}) - va.subs({x:x0, y:0})
    c5 = vb.subs({x:0, y:y0}) - va.subs({x:0, y:y0})
    
    # Flux continuity at interface midpoint
    flux_b_x0 = (Jit00*vb_x + Jit01*vb_y)*nx + (Jit10*vb_x + Jit11*vb_y)*ny
    flux_a_x0 = (Jit00*va_x + Jit01*va_y)*nx + (Jit10*va_x + Jit11*va_y)*ny
    c6 = mb*flux_b_x0 - ma*flux_a_x0 - jf

    eqs = [c1, c2, c3, c4, c5, c6]
    unknowns = [a1, a2, a3, b1, b2, b3]

    # ------------------------------------------------------------------
    # Linear solve (SymEngine)
    # ------------------------------------------------------------------
    sol = linsolve(eqs, unknowns)
    if not sol:
        raise RuntimeError("No solution found for P1 basis with given interface position and parameters")
    
    return [float(coefficient) for coefficient in sol]


def _solveCoefficients_P2(x0, y0, nx, ny, ma, mb, jf, Jit00, Jit01, Jit10, Jit11, nodal_values):
    """
    Solve for P2 quadratic basis coefficients.
    Returns a list of 12 coefficients [a1,a2,a3,a4,a5,a6,b1,b2,b3,b4,b5,b6].
    """
    
    # ------------------------------------------------------------------
    # Nodal values
    # ------------------------------------------------------------------
    v1, v2, v3, v4, v5, v6 = nodal_values
    # ------------------------------------------------------------------
    # Quadratic basis
    # ------------------------------------------------------------------
    va = a1 + a2*x + a3*y + a4*x*y + a5*x**2 + a6*y**2
    vb = b1 + b2*x + b3*y + b4*x*y + b5*x**2 + b6*y**2

    # First derivatives
    va_x = diff(va, x) # a2 + a4*y + 2*a5*x 
    va_y = diff(va, y) # a3 + a4*x + 2*a6*y
    vb_x = diff(vb, x) # b2 + b4*y + 2*b5*x
    vb_y = diff(vb, y) # b3 + b4*x + 2*b6*y

    # Second derivatives
    va_xx = diff(va_x, x) # 2*a5
    va_xy = diff(va_x, y) # a4
    va_yy = diff(va_y, y) # 2*a6

    vb_xx = diff(vb_x, x) # 2*b5
    vb_xy = diff(vb_x, y) # b4
    vb_yy = diff(vb_y, y) # 2*b6

    # Normal Laplacians
    tx = Jit00*nx + Jit10*ny
    ty = Jit01*nx + Jit11*ny

    va_nn = tx**2 * va_xx + 2*tx*ty * va_xy + ty**2 * va_yy
    vb_nn = tx**2 * vb_xx + 2*tx*ty * vb_xy + ty**2 * vb_yy

    assert not ((x0==0 and y0==0) or (x0==1 and y0==0) or (x0==0 and y0==1)), "Invalid interface location for quadratic basis functions. Interface passes through one of the triangle vertices."
    assert not (x0>0.5 and y0<=0.5), f"Invalid interface location for quadratic basis functions. You set the interface with {x0}>0.5 and {y0}<=0.5 but it can be transformed into {x0}<=0.5 and {y0}>0.5."
    # ------------------------------------------------------------------
    # Constraints (use exact rationals!)
    # ------------------------------------------------------------------
    c1 = va.subs({x:0, y:0}) - v1
    c2 = vb.subs({x:1, y:0}) - v2
    c3 = vb.subs({x:0, y:1}) - v3
    
    if (x0<=0.5 and y0<=0.5):
        c4 = vb.subs({x:Rational(1,2),y:0})  - v4
        c6 = vb.subs({x:0,y:Rational(1,2)})  - v6
    # elif (x0>0.5 and y0<=0.5):
    #     c4 = va.subs({x:Rational(1,2),y:0})  - v4
    #     c6 = vb.subs({x:0,y:Rational(1,2)})  - v6
    elif (x0<=0.5 and y0>0.5):
        c4 = vb.subs({x:Rational(1,2), y:0}) - v4
        c6 = va.subs({x:0, y:Rational(1,2)}) - v6
    elif (x0>0.5 and y0>0.5):
        c4 = va.subs({x:Rational(1,2), y:0}) - v4
        c6 = va.subs({x:0, y:Rational(1,2)}) - v6

    c5 = vb.subs({x:Rational(1,2), y:Rational(1,2)}) - v5

    # Basis continuity
    c7 = vb.subs({x:x0, y:0}) - va.subs({x:x0, y:0})
    c8 = vb.subs({x:0, y:y0}) - va.subs({x:0, y:y0})
    c9 = vb.subs({x:x0/2, y:y0/2}) - va.subs({x:x0/2, y:y0/2})

    # Flux continuity
    flux_b_x0 = (Jit00*vb_x.subs({x:x0, y:0}) + Jit01*vb_y.subs({x:x0, y:0}))*nx \
              + (Jit10*vb_x.subs({x:x0, y:0}) + Jit11*vb_y.subs({x:x0, y:0}))*ny

    flux_a_x0 = (Jit00*va_x.subs({x:x0, y:0}) + Jit01*va_y.subs({x:x0, y:0}))*nx \
              + (Jit10*va_x.subs({x:x0, y:0}) + Jit11*va_y.subs({x:x0, y:0}))*ny

    c10 = mb*flux_b_x0 - ma*flux_a_x0 - jf

    flux_b_y0 = (Jit00*vb_x.subs({x:0, y:y0}) + Jit01*vb_y.subs({x:0, y:y0}))*nx \
              + (Jit10*vb_x.subs({x:0, y:y0}) + Jit11*vb_y.subs({x:0, y:y0}))*ny

    flux_a_y0 = (Jit00*va_x.subs({x:0, y:y0}) + Jit01*va_y.subs({x:0, y:y0}))*nx \
              + (Jit10*va_x.subs({x:0, y:y0}) + Jit11*va_y.subs({x:0, y:y0}))*ny

    c11 = mb*flux_b_y0 - ma*flux_a_y0 - jf

    # Normal Laplacian continuity
    c12 = mb*vb_nn.subs({x:x0/2, y:y0/2}) - ma*va_nn.subs({x:x0/2, y:y0/2})

    eqs = [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12]

    unknowns = [a1,a2,a3,a4,a5,a6,b1,b2,b3,b4,b5,b6]

    # ------------------------------------------------------------------
    # Linear solve (SymEngine)
    # ------------------------------------------------------------------
    sol = linsolve(eqs, unknowns)
    if not sol:
        raise RuntimeError("No solution found for P2 basis with given interface position and parameters")
    
    # linsolve returns a FiniteSet - iterate directly over it to get the 12 solutions
    return [float(coefficient) for coefficient in sol]

