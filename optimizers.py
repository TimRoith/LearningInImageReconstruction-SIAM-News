import torch


class optimizer:
    def __init__(self,
                 u = None,
                 max_it = 10,
                 verbosity = 1,
                 energy_fun = None):
            
        self.max_it = max_it
        self.num_it = 0
        self.verbosity = verbosity
        self.energy_fun=energy_fun
        self.cur_energy = float('inf')
        self.hist = {'energy': []}
        self.u = u

    def solve(self):
        while not self.terminate():
            self.step()

            self.update_history()
            self.print_cur_it()
        # return solution
        return self.u

    def print_cur_it(self,):
        if self.verbosity > 0:
            print('Iteration: ' + str(self.num_it))
            print('Energy: ' +str(self.cur_energy))

    def update_history(self,):
        self.hist['energy'].append(self.cur_energy)

    def update_energy(self,):
        if not self.energy_fun is None:
            self.cur_energy = self.energy_fun(self.u)

    def step(self,):
        self.pre_step()
        self.inner_step()
        self.post_step()

    def pre_step(self,):
        pass

    def inner_step(self,):
        pass

    def post_step(self,):
        self.num_it+=1
        self.update_energy()
        self.update_history()

    def terminate(self):
        if self.num_it >= self.max_it:
            return True
        else:
            return False


class admm_old:
    def __init__(self, x, h, phi, energy_fun = None, max_it = 100, verbosity = 0, compute_primal_res = None, compute_dual_res = None, nu = 1):
        if not callable(energy_fun):
            self.energy_fun = lambda u: 0.
        self.energy_hist = []

        if compute_primal_res is None:
            self.compute_primal_res = lambda: 0.
        else:
            self.compute_primal_res = compute_primal_res

        if compute_dual_res is None:
            self.compute_dual_res = lambda: 0.
        else:
            self.compute_dual_res = compute_dual_res

        self.max_it = max_it
        self.num_it = 0
        self.verbosity = verbosity
        self.nu = nu

        self.x = x
        self.h = h
        self.phi = phi

        self.v = torch.zeros_like(x, device = x.device)
        self.u = torch.zeros_like(x, device = x.device)

    def step(self):
        self.x = self.h(self.v - self.u, self.nu)
        self.v = self.phi.prox(self.x + self.u, self.nu)
        self.u = self.u + self.x - self.v

        if self.num_it%1==0:
            energy = self.energy_fun(self.u)

        self.energy_hist.append(energy)
        if self.verbosity > 0 or self.num_it%10 == 0:
            print('Iteration: ' + str(self.num_it) + ', energy: ' + str(energy))

        self.num_it += 1

class admm(optimizer):
    def __init__(self,
                 x, h, phi, compute_primal_res = None,
                 compute_dual_res = None, nu = 1,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.x = x
        self.h = h
        self.phi = phi
        self.compute_primal_res = compute_primal_res
        self.compute_dual_res = compute_dual_res
        self.nu = nu

        if self.u is None:
            self.u = torch.zeros_like(x, device = x.device)

        self.v = torch.zeros_like(x, device = x.device)

    def inner_step(self):
        self.x = self.h(self.v - self.u, self.nu)
        self.v = self.phi.prox(self.x + self.u, self.nu)
        self.u = self.u + self.x - self.v

class pdgh(optimizer):
    def __init__(self,
        K, fstar, g, p,
        theta=1.0,
        tau=0.1, sigma=0.1,
        compute_primal_res = None,
        compute_dual_res = None,
        **kwargs
        ):

        super().__init__(**kwargs)
        self.K = K
        self.fstar = fstar
        self.g = g

        if self.u is None:
            raise RuntimeError('Initial value must be given!')

        self.u_bar = self.u.clone()
        self.u_old = None
        self.p = p.clone()
        self.p_old = None
        self.theta = theta
        self.tau = tau
        self.sigma = sigma

        if compute_primal_res is None:
            self.compute_primal_res = lambda: 0.
        else:
            self.compute_primal_res = compute_primal_res

        if compute_dual_res is None:
            self.compute_dual_res = lambda: 0.
        else:
            self.compute_dual_res = compute_dual_res

    def pre_step(self,):
        self.u_old = self.u
        self.p_old = self.p

    def inner_step(self):
        self.p = self.fstar.prox(self.p + self.sigma * self.K(self.u_bar), self.sigma)
        self.u = self.g.prox(self.u - self.tau * (self.K.adjoint(self.p)), self.tau)
        self.u_bar = self.u + self.theta * (self.u - self.u_old)

        # compute primal and dual residuals
        self.primal_res = self.compute_primal_res(self)
        self.dual_res   = self.compute_dual_res(self)


class lscg(optimizer):
    '''
    Solve the linear system Ax=b using the conjugate gradient method

    This implements Algorithm 1 from [1], or Algorithm 3 from [2]

    References:
    ----------
    [1] Fast Conjugate Gradient Algorithms with Preconditioning for Compressed Sensing
        J. A. Tropp, A. C. Gilbert, M. A. Saunders
        IEEE Transactions on Information Theory, 2007

    [2] https://sites.stat.washington.edu/wxs/Stat538-w03/conjugate-gradients.pdf
    '''
    def __init__(self, A, b, u, verbosity=0,**kwargs):
        super().__init__(**kwargs)
        self.A = A
        self.b = b
        self.u = u.clone()
        self.r = b - A(u)

        self.g = self.A.adjoint(self.r)
        self.p = self.g
        self.verbosity = verbosity
        self.num_it = 0

    def step(self):
        self.gg = self.g.norm()**2
        if self.verbosity > 0:
            print('Iteration ' + str(self.num_it) + ', norm of g: ' + str(self.gg))

        if self.num_it > 0:
            self.beta = -self.gg / self.gg_old
            self.p = self.g - self.beta * self.p

        Ap = self.A(self.p)
        alpha = self.gg / Ap.norm()**2
        #alpha[alpha!=alpha]=0
        self.u = self.u + alpha * self.p
        self.r = self.r - alpha * Ap
        self.g = self.A.adjoint(self.r)
        self.gg_old = self.gg
        self.num_it += 1

class nlcg:
    '''
    Solve the non-linear system f(u)=0 using the conjugate gradient method
    see, e.g., p. 1194 of [1]

    References:
    ----------
    [1] https://onlinelibrary.wiley.com/doi/epdf/10.1002/mrm.21391
    '''

    def __init__(self, f, u, verbosity=0,
                 beta = 0.6, alpha = 0.05,
                 compute_line_grad = None):
        self.f = f
        self.u = u.clone()
        self.g = self.f.gradient(self.u)
        self.gg = self.g.norm()**2
        self.p = -self.g
        self.verbosity = verbosity
        self.num_it = 0
        self.beta = beta
        self.alpha = alpha
        if compute_line_grad is None:
            self.compute_line_grad = lambda: 0.
        else:
            self.compute_line_grad = compute_line_grad

        self.max_loc_it = 3

    def step(self):
        self.gg_old = self.gg
        self.line_grad = self.compute_line_grad(self)

        if self.verbosity > 0:
            print('Iteration ' + str(self.num_it) + ', norm of g: ' + str(self.gg))

        t = 1
        loc_it = 0
        while (self.f(self.u + t * self.p) > self.f(self.u) + self.alpha * t * self.line_grad) and loc_it  < self.max_loc_it:
            t *= self.beta
            loc_it += 1

        self.u = self.u + t * self.p
        self.g = self.f.gradient(self.u)
        self.gg = self.g.norm()**2
        self.gamma = self.gg / self.gg_old
        self.p = -self.g + self.gamma * self.p
        self.num_it += 1

class splitBregman(optimizer):
    def __init__(self, A, D, Phi, k,
                 alpha=1.0,
                 gamma=1.0,
                 lamda=1.0,
                 inner_verbosity = 0,
                 max_inner_it=5,
                 **kwargs):
        super().__init__(**kwargs)

        self.A = A
        self.alpha = alpha
        self.D = D
        self.gamma = gamma
        self.lamda = lamda
        self.Phi = Phi

        class cg_op:
            def __call__(self,x):
                return lv([alpha**0.5 * A(x), 1/(gamma**0.5) * D(x)])

            def adjoint(self, p):
                return alpha**0.5 * A.adjoint(p[0]) + 1/(gamma**0.5) * D.adjoint(p[1])

        self.cg_op = cg_op()
        self.k = k

        self.b = 0 * self.D(self.u)
        self.d = None
        self.inner_verbosity = inner_verbosity
        self.max_inner_it = max_inner_it


    def inner_step(self):
        self.d = self.Phi.prox(self.b + self.D(self.u), self.lamda * self.gamma)
        self.b = self.b + self.D(self.u) - self.d
        inner_rhs = lv([self.k, 1/(self.gamma**0.5) * (self.d - self.b)])
        self.u = self.solve_inner(inner_rhs)
        self.u = self.u/torch.linalg.vector_norm(self.u)
        print(self.u.max())

    def solve_inner(self, rhs):
        return lscg(self.cg_op, rhs, self.u,
                    verbosity = self.inner_verbosity,
                    max_it=self.max_inner_it).solve()
    
class lv:
    def __init__(self,l):
        self.l = l

    def __len__(self):
        return len(self.l)

    def __add__(self, other):
        other = self._check_allowed_dtypes(other)
        return lv([self[i] + other[i] for i in range(len(self))])

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self._check_allowed_dtypes(other)
        return lv([self[i] - other[i] for i in range(len(self))])

    def __rsub__(self, other):
        other = self._check_allowed_dtypes(other)
        return lv([-self[i] + other[i] for i in range(len(self))])

    def __mul__(self, other):
        other = self._check_allowed_dtypes(other)
        return lv([self[i] * other[i] for i in range(len(self))])

    def __rmul__(self, other):
        other = self._check_allowed_dtypes(other)
        return lv([other[i] * self[i] for i in range(len(self))])

    def clone(self):
        return lv([self[i].clone() for i in range(len(self))])

    def __str__(self):
        return str(self.l)

    def __getitem__(self, key):
        return self.l[key]
    
    def norm(self):
        return torch.sqrt(sum([torch.linalg.norm(self[i])**2 for i in range(len(self))]))


    def _check_allowed_dtypes(self, other):
        if not isinstance(other, lv):
            other = self._promote_scalar(other)

        return other

    def _promote_scalar(self, scalar):
        '''
        Promote a scalar to a lv of the same length as self.
        This is a lazy implemantation, which is not efficient for large lvs.
        '''
        return lv([scalar for i in range(len(self))])
