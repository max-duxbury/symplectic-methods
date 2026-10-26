"""
Modified equations of one-step methods, after Hairer & Lubich 
(Asymptotic expansions and backward analysis for numerical integrators)
"""

import sympy as sp

h = sp.symbols('h')

def modified_equation(method, f, y, N):
    phi = method(f, y, h)
    phi_coe, d = {}, phi
    for n in range(1, N + 1):
        d = d.diff(h)
        phi_coe[n] = d.subs(h, 0) / sp.factorial(n)
    fcoe = {1: f(y)}
    for n in range(2, N + 1):
        modeq = sum((h**j*fcoe[j+1] for j in range(n - 1)), sp.zeros(*y.shape))
        diffy, y_tilde = y, y
        for k in range(1, n + 1):
            diffy = diffy.jacobian(y) * modeq
            y_tilde = y_tilde + h**k * diffy / sp.factorial(k)
        res = y_tilde - phi
        fcoe[n] = -(y_tilde.applyfunc(lambda e: sp.expand(e).coeff(h, n)) - phi_coe[n]).applyfunc(sp.simplify)
    return fcoe

def hamiltonian(fj, y):
    d = y.rows // 2
    q, p = list(y[:d, :]), list(y[d:, :])
    grad = dict(zip(q + p, list(-fj[d:, :]) + list(fj[:d, :])))
    H = 0
    for x in p + q:
        H += sp.integrate(sp.simplify(grad[x] - sp.diff(H, x)), x, conds='none')
    assert all(sp.simplify(sp.diff(H,x) -g) == 0 for x, g in grad.items()), "Err: Not Hamiltonian"
    return sp.simplify(H)

def explicit_euler(f, y, h):
    return y + h*f(y)

def heun(f, y, h):
    return y + h*0.5*(f(y) + f(y + h*f(y)))

def symplectic_euler(f, y, h):
    d = y.rows // 2
    q, p = y[:d, :], y[d:, :]
    p1 = p + h*f(y)[d:, :]
    q1 = q + h*f(q.col_join(p1))[:d, :]
    return q1.col_join(p1)

def stormer_verlet(f, y, h):
    d = y.rows // 2
    q, p = y[:d, :], y[d:, :]
    p_half = p + h/2*f(y)[d:, :]
    q1 = q + h*f(q.col_join(p_half))[:d, :]
    p1 = p_half + h/2*f(q1.col_join(p_half))[d:, :]
    return q1.col_join(p1)


if __name__ == "__main__":
    y = sp.symbols('y')
    ee = modified_equation(explicit_euler, lambda Y: Y.applyfunc(lambda v: v**2), sp.Matrix([y]), 6)
    assert [sp.factor(ee[j][0]) for j in range(1, 7)] == [y**2, -y**3, sp.Rational(3, 2)*y**4, -sp.Rational(8, 3)*y**5, sp.Rational(31, 6)*y**6, -sp.Rational(157, 15)*y**7]
    q, p = sp.symbols('q p')
    QP = sp.Matrix([q, p])
    pendulum = lambda Y: sp.Matrix([Y[1], -sp.sin(Y[0])])
    assert modified_equation(heun, pendulum, QP, 2)[2] == sp.zeros(2, 1) 
    sv = modified_equation(stormer_verlet, pendulum, QP, 4)
    assert sv[2] == sp.zeros(2, 1) and sv[4] == sp.zeros(2, 1)                     
    se = modified_equation(symplectic_euler, pendulum, QP, 2)
    assert sp.simplify(hamiltonian(se[2], QP) + p*sp.sin(q)/2) == 0                
