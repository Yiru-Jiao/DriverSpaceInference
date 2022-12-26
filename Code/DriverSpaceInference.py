import numpy as np
from scipy.misc import derivative
from scipy import optimize
from scipy.stats import t
import warnings
warnings.filterwarnings('ignore')

# Parameter estimation
# Please refer to subsection 2.3 in our article

class drspace:
    """The class is used to estimate parameters of driver space from vehicle pair samples."""

    def __init__(self, samples, eps=1e-4, width=2, length=4, percentile=0.5):

        #//Extract input variables
        self.width = width
        self.length = length
        self.samples = samples
        self.corrcoef_xy = np.corrcoef(self.samples.x, self.samples.y)[1,0]
        self.percentile = percentile

        #//Initialize parameters
        self.epsilon = eps          
        self.bx_plus = 2
        self.bx_minus = 2
        self.by_plus = 2
        self.by_minus = 2

        try:
            self.ry_plus = max(self.length/2, np.percentile(self.samples.y[(self.samples.y>=0)&(abs(self.samples.x)<self.width/2)], self.percentile))
        except:
            self.ry_plus = self.length / 2
        try:
            self.ry_minus = max(self.length/2, np.percentile(-self.samples.y[(self.samples.y<0)&(abs(self.samples.x)<self.width/2)], self.percentile))
        except:
            self.ry_minus = self.length / 2
        try:
            self.rx_plus = max(self.width/2, np.percentile(self.samples.x[(self.samples.x>0)&(self.samples.y<=max(self.length/2,self.ry_plus/2))&(self.samples.y>=-max(self.length/2,self.ry_minus/2))], self.percentile))
        except:
            self.rx_plus = self.width / 2
        try:
            self.rx_minus = max(self.width/2, np.percentile(-self.samples.x[(self.samples.x<0)&(self.samples.y<=max(self.length/2,self.ry_plus/2))&(self.samples.y>=-max(self.length/2,self.ry_minus/2))], self.percentile))
        except:
            self.rx_minus = self.width / 2
    
    # Log likelihood

    ## Log likelihood to all parameters

    def LogL(self, para):
        
        rx_plus, rx_minus, ry_plus, ry_minus, bx_plus, bx_minus, by_plus, by_minus = para

        rx = (1 + np.sign(self.x)) / 2 * rx_plus + (1 - np.sign(self.x)) / 2 * rx_minus
        bx = (1 + np.sign(self.x)) / 2 * bx_plus + (1 - np.sign(self.x)) / 2 * bx_minus
        ry = (1 + np.sign(self.y)) / 2 * ry_plus + (1 - np.sign(self.y)) / 2 * ry_minus
        by = (1 + np.sign(self.y)) / 2 * by_plus + (1 - np.sign(self.y)) / 2 * by_minus

        elements = np.log(1+self.epsilon-np.exp(-np.power(np.absolute(self.x / rx), bx) - np.power(np.absolute(self.y / ry), by)))

        return sum(elements)

    ## Log likelihood to rx_plus

    def LogL_rx_plus(self, rx_plus):
        
        rx = (1 + np.sign(self.x)) / 2 * rx_plus + (1 - np.sign(self.x)) / 2 * self.rx_minus
        bx = (1 + np.sign(self.x)) / 2 * self.bx_plus + (1 - np.sign(self.x)) / 2 * self.bx_minus
        ry = (1 + np.sign(self.y)) / 2 * self.ry_plus + (1 - np.sign(self.y)) / 2 * self.ry_minus
        by = (1 + np.sign(self.y)) / 2 * self.by_plus + (1 - np.sign(self.y)) / 2 * self.by_minus

        elements = np.log(1+self.epsilon-np.exp(-np.power(np.absolute(self.x / rx), bx) - np.power(np.absolute(self.y / ry), by)))

        return sum(elements)

    ## Second-order derivative of the Log likelihood to rx_plus
    
    def LogL_rx_plus_prime2(self, rx_plus): 
        return derivative(self.LogL_rx_plus, rx_plus, dx=0.5, n=2, order=3)

    ## Log likelihood to rx_minus

    def LogL_rx_minus(self, rx_minus):
        rx = (1 + np.sign(self.x)) / 2 * self.rx_plus + (1 - np.sign(self.x)) / 2 * rx_minus
        bx = (1 + np.sign(self.x)) / 2 * self.bx_plus + (1 - np.sign(self.x)) / 2 * self.bx_minus
        ry = (1 + np.sign(self.y)) / 2 * self.ry_plus + (1 - np.sign(self.y)) / 2 * self.ry_minus
        by = (1 + np.sign(self.y)) / 2 * self.by_plus + (1 - np.sign(self.y)) / 2 * self.by_minus

        elements = np.log(1+self.epsilon-np.exp(-np.power(np.absolute(self.x / rx), bx) - np.power(np.absolute(self.y / ry), by)))

        return sum(elements)

    ## Second-order derivative of the Log likelihood to rx_minus

    def LogL_rx_minus_prime2(self, rx_minus):
        return derivative(self.LogL_rx_minus, rx_minus, dx=0.5, n=2, order=3)

    ## Log likelihood to ry_plus

    def LogL_ry_plus(self, ry_plus):
        rx = (1 + np.sign(self.x)) / 2 * self.rx_plus + (1 - np.sign(self.x)) / 2 * self.rx_minus
        bx = (1 + np.sign(self.x)) / 2 * self.bx_plus + (1 - np.sign(self.x)) / 2 * self.bx_minus
        ry = (1 + np.sign(self.y)) / 2 * ry_plus + (1 - np.sign(self.y)) / 2 * self.ry_minus
        by = (1 + np.sign(self.y)) / 2 * self.by_plus + (1 - np.sign(self.y)) / 2 * self.by_minus
    
        elements = np.log(1+self.epsilon-np.exp(-np.power(np.absolute(self.x / rx), bx) - np.power(np.absolute(self.y / ry), by)))

        return sum(elements)

    ## Second-order derivative of the Log likelihood to ry_plus

    def LogL_ry_plus_prime2(self, ry_plus):
        return derivative(self.LogL_ry_plus, ry_plus, dx=0.5, n=2, order=3)

    ## Log likelihood to ry_minus

    def LogL_ry_minus(self, ry_minus):
        rx = (1 + np.sign(self.x)) / 2 * self.rx_plus + (1 - np.sign(self.x)) / 2 * self.rx_minus
        bx = (1 + np.sign(self.x)) / 2 * self.bx_plus + (1 - np.sign(self.x)) / 2 * self.bx_minus
        ry = (1 + np.sign(self.y)) / 2 * self.ry_plus + (1 - np.sign(self.y)) / 2 * ry_minus
        by = (1 + np.sign(self.y)) / 2 * self.by_plus + (1 - np.sign(self.y)) / 2 * self.by_minus
    
        elements = np.log(1+self.epsilon-np.exp(-np.power(np.absolute(self.x / rx), bx) - np.power(np.absolute(self.y / ry), by)))

        return sum(elements)

    ## Second-order derivative of the Log likelihood to ry_minus

    def LogL_ry_minus_prime2(self, ry_minus):
        return derivative(self.LogL_ry_minus, ry_minus, dx=0.5, n=2, order=3)

    ## Negative Log likelihood to betas

    def NLogL_betas(self, para):
    
        bx_plus, bx_minus, by_plus, by_minus = para
        
        rx = (1 + np.sign(self.x)) / 2 * self.rx_plus + (1 - np.sign(self.x)) / 2 * self.rx_minus
        bx = (1 + np.sign(self.x)) / 2 * bx_plus + (1 - np.sign(self.x)) / 2 * bx_minus
        ry = (1 + np.sign(self.y)) / 2 * self.ry_plus + (1 - np.sign(self.y)) / 2 * self.ry_minus
        by = (1 + np.sign(self.y)) / 2 * by_plus + (1 - np.sign(self.y)) / 2 * by_minus
    
        elements = np.log(1+self.epsilon-np.exp(-np.power(np.absolute(self.x / rx), bx) - np.power(np.absolute(self.y / ry), by)))

        return -sum(elements)

    # Optimize betas
    
    def optimize_betas(self, initial_guess):    
        betas = optimize.minimize(self.NLogL_betas,
                                  x0=initial_guess,
                                  method='L-BFGS-B',
                                  bounds=((2,np.inf),(2,np.inf),(2,np.inf),(2,np.inf)),
                                  jac=None,
                                  tol=None,
                                  options={'disp': None, 
                                           'maxcor': 10, 
                                           'ftol': 2.220446049250313e-09, 
                                           'gtol': 1e-05, 
                                           'eps': 1e-08, 
                                           'maxfun': 15000, 
                                           'maxiter': 15000, 
                                           'iprint': - 1,
                                           'maxls': 25,
                                           'finite_diff_rel_step': None})
        return betas

    # Inference

    def Inference(self, max_iter=50, workers=1):
        
        parameters = np.zeros((2, 8))
        stderr_betas = np.zeros((0, 4))
        pvalue_betas = np.zeros((0, 4))
        parameters[1, :] = [self.rx_plus, self.rx_minus, self.ry_plus, self.ry_minus, self.bx_plus, self.bx_minus, self.by_plus, self.by_minus]
      
        self.x = self.samples.x.values
        self.y = self.samples.y.values
        self.v = self.samples.v.values

        try:
            # Lower seraching bounds
            if self.samples.cangle.iloc[0]==0:
                percentage = (0.15 + 0.0003*self.v.mean()**2.7)/150
            elif self.samples.cangle.iloc[0]==1:
                percentage = 0.0005 + (0.002*self.v.mean())**2
            candidates = self.samples.y[(self.samples.y>=0)&(abs(self.samples.x)<self.width/2)]
            y_plus_llimit = max(self.length/2, candidates.nsmallest(max(10, int(len(candidates)*percentage))).nlargest(10).mean()*0.8)
            candidates = (-self.samples.y[(self.samples.y<0)&(abs(self.samples.x)<self.width/2)])
            y_minus_llimit = max(self.length/2, candidates.nsmallest(max(10, int(len(candidates)*percentage))).nlargest(10).mean()*0.8)
            # candidates = self.samples.x[(self.samples.x>0)&(self.samples.y<y_plus_llimit)&(self.samples.y>y_minus_llimit)]
            # x_plus_llimit = max(self.width/2, candidates.nsmallest(max(10, int(len(candidates)*percentage))).nlargest(10).mean()*0.8)
            # candidates = (-self.samples.x[(self.samples.x<0)&(self.samples.y<y_plus_llimit)&(self.samples.y>y_minus_llimit)])
            # x_minus_llimit = max(self.width/2, candidates.nsmallest(max(10, int(len(candidates)*percentage))).nlargest(10).mean()*0.8)
            x_plus_llimit = 1.5
            x_minus_llimit = 1.5

            # Upper seraching bounds
            conditionx = (self.samples.x<x_plus_llimit)&(self.samples.x>-x_minus_llimit)
            conditiony = (self.samples.y<y_plus_llimit)&(self.samples.y>-y_minus_llimit)

            percentage = 0.15 + 0.0003*self.v.mean()**2.7
            candidates = self.samples.y[conditionx&(self.samples.y>0)]
            y_plus_ulimit = candidates.nsmallest(int(len(candidates)*percentage)).nlargest(10).mean()
            candidates = -self.samples.y[conditionx&(self.samples.y<0)]
            y_minus_ulimit = candidates.nsmallest(int(len(candidates)*percentage)).nlargest(10).mean()
            # candidates = self.samples.x[conditiony&(self.samples.x>0)]
            # x_plus_ulimit = candidates.nsmallest(int(len(candidates)*0.25)).nlargest(10).mean()
            # candidates = -self.samples.x[conditiony&(self.samples.x<0)]
            # x_minus_ulimit = candidates.nsmallest(int(len(candidates)*0.25)).nlargest(10).mean()
            x_plus_ulimit = 5.5
            x_minus_ulimit = 5.5
            
            y_plus_llimit -= 0.5
            y_minus_llimit -= 0.5
            x_plus_llimit, x_minus_llimit, y_plus_llimit, y_minus_llimit = np.floor(np.array([x_plus_llimit, x_minus_llimit, y_plus_llimit, y_minus_llimit])*10)/10
            x_plus_ulimit, x_minus_ulimit, y_plus_ulimit, y_minus_ulimit = np.ceil(np.array([x_plus_ulimit, x_minus_ulimit, y_plus_ulimit, y_minus_ulimit])*10)/10
            limits = np.array([x_plus_llimit, x_plus_ulimit, x_minus_llimit, x_minus_ulimit, y_plus_llimit, y_plus_ulimit, y_minus_llimit, y_minus_ulimit])
            # print(limits)

            iteration = 0
            while np.any(parameters[-1] != parameters[-2]):

                #//Stop when reaching the maximum number of iterations
                if iteration > max_iter:
                    parameters = np.vstack((parameters, np.ones((1,8))*np.nan))
                    stderr_betas = np.vstack((stderr_betas, np.ones((1,4))*np.nan))
                    pvalue_betas = np.vstack((pvalue_betas, np.ones((1,4))*np.nan))
                    warnings.warn('Max. iterations reached', RuntimeWarning)
                    break
                
                #//Stop when several sets of values repeat and select the set of values with the smallest sum of p-values
                array2check = np.all(np.equal(np.array([parameters[-1],]*len(parameters[:-1])), parameters[:-1]), axis=1)
                if np.any(array2check):
                    repeated_parameters = parameters[np.where(array2check)[0][0]:-1]
                    repeated_stderrs = stderr_betas[-len(repeated_parameters)-1:-1]
                    repeated_pvals = pvalue_betas[-len(repeated_parameters)-1:-1]
                    idx_chosen = np.argmin(repeated_pvals.sum(axis=1))
                    parameters = np.vstack((parameters, repeated_parameters[idx_chosen]))
                    stderr_betas = np.vstack((stderr_betas, repeated_stderrs[idx_chosen]))
                    pvalue_betas = np.vstack((pvalue_betas, repeated_pvals[idx_chosen]))
                    warnings.warn('Repetitive alternatives; the alternative with smallest sum of pvals is chosen', RuntimeWarning)
                    break

                self.rx_plus, self.rx_minus, self.ry_plus, self.ry_minus = np.round([self.rx_plus, self.rx_minus, self.ry_plus, self.ry_minus],1)
                rx_plus = optimize.brute(self.LogL_rx_plus_prime2, (slice(x_plus_llimit, x_plus_ulimit+0.1, 0.1),), finish=None, workers=workers)
                rx_minus = optimize.brute(self.LogL_rx_minus_prime2, (slice(x_minus_llimit, x_minus_ulimit+0.1, 0.1),), finish=None, workers=workers)
                ry_plus = optimize.brute(self.LogL_ry_plus_prime2, (slice(y_plus_llimit, y_plus_ulimit+0.1, 0.1),), finish=None, workers=workers)
                ry_minus = optimize.brute(self.LogL_ry_minus_prime2, (slice(y_minus_llimit, y_minus_ulimit+0.1, 0.1),), finish=None, workers=workers)
                self.rx_plus, self.rx_minus, self.ry_plus, self.ry_minus = [rx_plus, rx_minus, ry_plus, ry_minus]

                betas = self.optimize_betas([self.bx_plus, self.bx_minus, self.by_plus, self.by_minus])
                self.bx_plus, self.bx_minus, self.by_plus, self.by_minus = np.round(betas.x,1)
                
                stderr = np.sqrt(np.diag(betas.hess_inv.todense()))
                tstat = betas.x / stderr
                pval = (1 - t.cdf(np.absolute(tstat), len(self.x)-1)) * 2

                parameters = np.vstack((parameters, [self.rx_plus, self.rx_minus, self.ry_plus, self.ry_minus, self.bx_plus, self.bx_minus, self.by_plus, self.by_minus]))
                stderr_betas = np.vstack((stderr_betas, stderr))
                pvalue_betas = np.vstack((pvalue_betas, pval))
                # print(parameters[-1])

                iteration += 1
        except:
            limits = np.ones(8)*np.nan
            parameters = np.vstack((parameters, np.ones((1,8))*np.nan))
            stderr_betas = np.vstack((stderr_betas, np.ones((1,4))*np.nan))
            pvalue_betas = np.vstack((pvalue_betas, np.ones((1,4))*np.nan))

        return (limits, parameters[-1].copy(), stderr_betas[-1].copy(), pvalue_betas[-1].copy())