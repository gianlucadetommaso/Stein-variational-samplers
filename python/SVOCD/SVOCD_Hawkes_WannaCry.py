import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
from IPython.display import display, Math, Latex, clear_output
import multiprocessing
from functools import partial
import pandas as pd
import time

        
###########################################################################################################################################      
### Bayesian Hawkes process model
        
class HAWKES:
    def __init__(self, tt, T0):
        self.T0 = T0
        self.tt = np.array(tt)
        self.m = self.tt.size
        if self.m > 1:            
            self.dtt = np.tril( self.tt[1:, np.newaxis] - self.tt[np.newaxis, :self.m-1] )
            
        
        self.DoF = 3
        self.mu_idx = 0
        self.gamma_idx = 1
        self.delta_idx = 2
        
        self.hyperMean = np.zeros( (self.DoF, 1) )
        self.hyperVar  = 10 * np.ones( (self.DoF, 1) )

    ## Posterior inference 
    
    def getMinusLogPrior(self, thetas):
        nSamples = thetas.size // self.DoF
        thetas = thetas.reshape(self.DoF, nSamples)
        
        shift = thetas - self.hyperMean
        tmp = 0.5 * np.sum( shift ** 2 / self.hyperVar, 0 )
        return tmp if nSamples > 1 else tmp.squeeze()
        
    def getIntensity(self, thetas):
        nSamples = thetas.size // self.DoF
        expthetas = np.maximum( np.exp( np.minimum( thetas.reshape(self.DoF, nSamples), 700 ) ), 1e-8 ) 
        mus = expthetas[self.mu_idx,:]
        if self.m > 1:
            gammas = expthetas[self.gamma_idx,:]
            deltas = expthetas[self.delta_idx,:]

            tmp = np.vstack( ( mus, \
              mus + gammas * ( \
                np.sum( np.exp( - deltas * self.dtt[:,:,np.newaxis] ), 1 ) \
                - np.arange(self.m - 2,-1,-1)[:,np.newaxis] \
                             ) \
                             ) )
        elif self.m == 1:
            tmp = mus
        tmp = np.maximum(tmp, 1e-6)
        return tmp if nSamples > 1 else tmp.squeeze()
    
    def getCompensator(self, thetas):
        nSamples = thetas.size // self.DoF
        expthetas = np.maximum( np.exp( np.minimum( thetas.reshape(self.DoF, nSamples), 700 ) ), 1e-8 ) 
        mus = expthetas[self.mu_idx,:]
        if self.m > 1:
            gammas = expthetas[self.gamma_idx,:]
            deltas = expthetas[self.delta_idx,:]

            tmp = mus * (self.tt[-1] - self.T0) + gammas / deltas * ( \
                    self.m - 1 - np.sum( np.exp( - deltas * self.dtt[-1,:,np.newaxis] ), 0 ) \
                                                        )
        elif self.m == 1:
            tmp =  mus * (self.tt - self.T0)
        return tmp if nSamples > 1 else tmp.squeeze()
    
    def getMinusLogLikelihood(self, thetas):
        if self.m > 1:
            return self.getCompensator(thetas) - np.sum( np.log( self.getIntensity(thetas) ), 0 )
        else:
            return self.getCompensator(thetas) - np.log( self.getIntensity(thetas) )
    
    def getMinusLogPosterior(self, thetas):
        if self.m == 0:
            return self.getMinusLogPrior(thetas)
        else:
            return self.getMinusLogPrior(thetas) + self.getMinusLogLikelihood(thetas)
    
    def getGradientMinusLogPrior(self, thetas):        
        nSamples = thetas.size // self.DoF
        thetas = thetas.reshape(self.DoF, nSamples)
        tmp = (thetas - self.hyperMean) / self.hyperVar
        return tmp if nSamples > 1 else tmp.squeeze()
    
    def getGradientLogIntensity(self, thetas):
        nSamples = thetas.size // self.DoF
        thetas = thetas.reshape(self.DoF, nSamples)
        expthetas = np.maximum( np.exp( np.minimum( thetas.reshape(self.DoF, nSamples), 700 ) ), 1e-8 ) 
        mus = expthetas[self.mu_idx,:]
        gammas = expthetas[self.gamma_idx,:]
        deltas = expthetas[self.delta_idx,:]
        if (nSamples > 1) or (self.m == 1):
            lams = self.getIntensity(thetas)
        else:
            np.array(self.getIntensity(thetas))[:,np.newaxis]
    
        if self.m > 1:
            expmdeltasdtt = np.exp( - deltas * self.dtt[:,:,np.newaxis] )
            sumexpmdeltasdtt = np.sum( expmdeltasdtt, 1 ) \
                                    - np.arange(self.m - 2,-1,-1)[:,np.newaxis]
            dttexpmdeltasdtt = np.sum( self.dtt[:,:,np.newaxis] * expmdeltasdtt, 1 )

            gllams = np.zeros( (self.DoF, self.m, nSamples ) )
            gllams[self.mu_idx,:,:]    = mus / lams
            gllams[self.gamma_idx,:,:] = gammas * np.vstack( \
                                (np.zeros(nSamples), sumexpmdeltasdtt / lams[1:]) )
            gllams[self.delta_idx,:,:] = deltas * np.vstack( \
                                (np.zeros(nSamples), - gammas * dttexpmdeltasdtt / lams[1:]) )
            return gllams
        elif self.m == 1:
            return np.vstack( (np.ones(nSamples), np.zeros( (self.DoF-1, nSamples) ) ) )
               
    def getGradientMinusLogLikelihood(self, thetas, *arg):
        nSamples = thetas.size // self.DoF
        thetas = thetas.reshape(self.DoF, nSamples)
        expthetas = np.maximum( np.exp( np.minimum( thetas.reshape(self.DoF, nSamples), 700 ) ), 1e-8 )  
        mus = expthetas[self.mu_idx,:]
        gammas = expthetas[self.gamma_idx,:]
        deltas = expthetas[self.delta_idx,:]
        
        if len(arg) == 0:
            gllams = self.getGradientLogIntensity(thetas)
        else:
            gllams = arg[0]
        # Expressions    
        if self.m > 1:
            expmdeltasdtm_tt = np.exp( - deltas * self.dtt[-1,:,np.newaxis] )
            f = self.m - 1 - np.sum( expmdeltasdtm_tt, 0 )
            df = np.sum( self.dtt[-1,:,np.newaxis] * expmdeltasdtm_tt, 0 )

            gcomp = np.zeros( (self.DoF, nSamples) )
            gcomp[self.mu_idx,:]    = mus * (self.tt[-1] - self.T0)
            gcomp[self.gamma_idx,:] = f * gammas / deltas
            gcomp[self.delta_idx,:] = gammas * ( df - f / deltas )
            tmp = gcomp - np.sum(gllams, 1)
        elif self.m == 1:
            gcomp = np.vstack( \
                        (mus * (self.tt - self.T0) * np.ones(nSamples), np.zeros( (self.DoF-1, nSamples) ) ) )
            tmp = gcomp - gllams
        return tmp if nSamples > 1 else tmp.squeeze()
    
    def getGradientMinusLogPosterior(self, thetas, *arg):
        nSamples = thetas.size // self.DoF
        if len(arg) == 0:
            gllams = self.getGradientLogIntensity(thetas)
        else:
            gllams = arg[0]
        tmp = self.getGradientMinusLogPrior(thetas) \
            + self.getGradientMinusLogLikelihood(thetas, gllams)
        return tmp if nSamples > 1 else tmp.squeeze()
    
    def getAsymptHessianMinusLogPosterior(self, thetas, *arg):
        nSamples = thetas.size // self.DoF
        if len(arg) == 0:
            gllams = self.getGradientLogIntensity(thetas)
        else:
            gllams = arg[0]
        tmp = np.sum( \
            gllams.reshape(self.DoF,1,self.m,nSamples) * \
            gllams.reshape(1,self.DoF,self.m,nSamples), 2) \
                + np.eye(self.DoF).reshape(self.DoF, self.DoF, 1) / self.hyperVar
        return tmp if nSamples > 1 else tmp.squeeze()       
                
    ## Prediction inference    
    def getPredIntensity(self, thetas, t, *arg):
        nSamples = thetas.size // self.DoF
        expthetas = np.maximum( np.exp( np.minimum( thetas.reshape(self.DoF, nSamples), 700 ) ), 1e-8 ) 
        mus = expthetas[self.mu_idx,:]
        if self.m != 0:
            gammas = expthetas[self.gamma_idx,:]
            deltas = expthetas[self.delta_idx,:]
            
            if len(arg) < 1:
                if self.m == 1:
                    self.dt_tt = t - self.tt
                else:
                    self.dt_tt = (t - self.tt)[:,np.newaxis]
            else:
                self.dt_tt = arg[0]

            tmp = mus + gammas * np.sum( np.exp( - deltas * self.dt_tt ), 0 )
        else:
            tmp = mus
        return tmp if nSamples > 1 else tmp.squeeze()
    
    def getPredCompensator(self, thetas, t, *arg):
        nSamples = thetas.size // self.DoF
        expthetas = np.maximum( np.exp( np.minimum( thetas.reshape(self.DoF, nSamples), 700 ) ), 1e-8 ) 
        mus = expthetas[self.mu_idx,:]
        if self.m != 0:
            gammas = expthetas[self.gamma_idx,:]
            deltas = expthetas[self.delta_idx,:]
            
            if len(arg) < 2:
                if self.m == 1:
                    self.dtm_tt = 0
                else:
                    self.dtm_tt = np.hstack( (0, self.dtt[-1,:]) )[:,np.newaxis]
            else:
                self.dtm_tt = arg[1]
            if len(arg) < 1:
                if self.m == 1:
                    self.dt_tt = np.array([t - self.tt])
                else:
                    self.dt_tt = (t - self.tt)[:,np.newaxis]
            else:
                self.dt_tt = arg[0]

            tmp =  mus * self.dt_tt[-1] + gammas / deltas * \
                np.sum( np.exp( - deltas * self.dtm_tt ) \
                       - np.exp( - deltas * self.dt_tt ) , 0 )
        else:
            tmp =  mus * (t - self.T0)
        return tmp if nSamples > 1 else tmp.squeeze()
    
    def getPredMinusLogLikelihood(self, thetas, t):
        return self.getPredCompensator(thetas, t) - np.log( self.getPredIntensity(thetas, t) )
    
    def simulateNewEvent(self, thetas):
        nSamples = thetas.size // self.DoF
        expthetas = np.maximum( np.exp( np.minimum( thetas.reshape(self.DoF, nSamples), 700 ) ), 1e-8 ) 
        mus = expthetas[self.mu_idx,:]
        S2 = - np.log( np.random.uniform(size = nSamples) ) / mus
        if self.m > 0:
            gammas = expthetas[self.gamma_idx,:]
            deltas = expthetas[self.delta_idx,:]

            lamsplus_mus = gammas * \
                (1 + np.sum( np.exp( - deltas * self.dtt[-1,:,np.newaxis] ), 0 ) ) \
                if self.m != 1 else gammas
            lamsplus_mus = np.maximum( lamsplus_mus, 1e-8 )

            D = 1 + deltas * np.log( np.random.uniform(size = nSamples) ) / lamsplus_mus
            S = np.zeros(nSamples)
            idxplus = np.where(D > 0)
            idxminus = np.where(D <= 0)
            S[idxplus] = np.min( np.vstack( (- np.log( D[idxplus] ) \
                                / deltas[idxplus], S2[idxplus]) ), 0 )
            S[idxminus] = S2[idxminus]
            tmp = S
        else:
            tmp = S2       
        return tmp if nSamples > 1 else tmp.squeeze()
    
    def getMAP(self, *arg):
        x0 = np.random.normal(size = self.DoF) if len(arg) == 0 else arg[0]
        res = optimize.minimize(self.getMinusLogPosterior, x0, method='L-BFGS-B')
        return res.x
   ###########################################################################################################################################

### Stein variational Newton (SVN)

class SVN:
    def __init__(self, model, *arg):
        self.model = model
        self.DoF = model.DoF
        self.nParticles = 100
        self.nIterations = 30
        self.stepsize = 1
        self.MAP = self.model.getMAP( np.random.normal( size = self.DoF ) )[:,np.newaxis]
        if len(arg) == 0:
            self.resetParticles(np.arange(self.nParticles))
        else:
            self.particles = arg[0]
            
    def apply(self):
        maxshiftold = np.inf
        Q = np.zeros( (self.DoF, self.nParticles) )
        for iter_ in range(self.nIterations):
            gllams = self.model.getGradientLogIntensity(self.particles)
            gmlpt = self.model.getGradientMinusLogPosterior(self.particles, gllams)
            Hmlpt = self.model.getAsymptHessianMinusLogPosterior(self.particles, gllams) 
            M = np.mean(Hmlpt, 2)
            
            for i_ in range(self.nParticles):
                sign_diff = self.particles[:,i_,np.newaxis] - self.particles
                Msd   = np.matmul(M, sign_diff)
                kern  = np.exp( - 0.5 * np.sum( sign_diff * Msd, 0 ) )
                gkern = Msd * kern
                
                mgJ = np.mean(- gmlpt * kern + gkern , 1)
                HJ  = np.mean(Hmlpt * kern ** 2, 2) + np.matmul(gkern, gkern.T) / self.nParticles
                Q[:,i_] = np.linalg.solve(HJ, mgJ)
            self.particles += self.stepsize * Q
            nanidx = np.where(np.isnan(self.particles).any(0))[0]
            if nanidx.size > 0:
                self.particles[:,nanidx] = np.random.normal( size = (self.DoF,nanidx.size) )
                      
            maxshift = np.linalg.norm(Q, np.inf)
            if np.isnan(maxshift) or (maxshift > 1e20):
                self.resetParticles(np.arange(self.nParticles))
                self.stepsize = 1
            elif maxshift < maxshiftold:
                self.stepsize *= 1.01
            else:
                self.stepsize *= 0.9
            maxshiftold = maxshift
                          
    def resetParticles(self, idx):
        lenidx = len(idx) if isinstance(idx, int) == 0 else 1
        if lenidx == self.nParticles:
            self.particles = self.MAP + np.random.normal( scale = 1, size = (self.DoF, lenidx) )
        else:
            self.particles[:,idx] = self.MAP + np.random.normal( scale = 1, size = (self.DoF, lenidx) )   
            
###########################################################################################################################################       
### Stein variational Online Changepoint Detection (SVOCD)
##  Generalized BOCPD via SVN
      
class BOCPD:
    def __init__(self, data):
        # Import data
        self.data = data
        self.nData = len(self.data)
        
        # Setup computational effort
        self.rmax = 30     # Max run length 
        self.npts = 100    # Number of posterior samples
        self.nrls = 100    # Number of run length samples
        
        # Setup credible interval
        self.flag_lci = 1      # Test left tail -> fire up drastic drecrease
        self.flag_rci = 1      # Test right tail -> fire up drastic increase
        self.risk_level_l = 5           # Left percentage of probability risk     
        self.risk_level_r = 0.00001     # Right percentage of probability risk
        self.pred_mean = np.zeros(self.nData)                         # Predictive mean
        if self.flag_lci: self.percentile_l = np.zeros(self.nData)    # Left percentile (if flagged up)
        if self.flag_rci: self.percentile_r = np.zeros(self.nData)    # Right percentile (if flagged up)
                    
        # Changepoint prior
        self.rlr = 100              # Run length rate hyper-parameter -> decrease to weigh changepoint prob more 
        self.H   = 1 / self.rlr     # Hazard rate
        
        # Initialize run length probabilities
        self.jp  = 1          # Joint 
        self.rlp = self.jp    # Posterior
        
        # Initialize models
        self.model = {}
        self.model[0] = HAWKES([], 0)
        
        # Initialize Stein Variational Newton samplers
        self.svn   = {}        
        
        # Initialize posterior samples and probabilities
        self.pts = {}             
        self.pts[0] = np.random.normal( size = (self.model[0].DoF, self.npts) )       
        self.ptp = {}             
        self.ptp[0]   = np.exp( - self.model[0].getMinusLogPosterior( self.pts[0] ) )
        self.ptp[0]  /= np.sum( self.ptp[0] )
        
        # Risk tolerance
        self.riskTol = 1e-2
        
        # Initialize changepoints
        self.changepoints = {}
    
    def apply(self):            
        for t_ in range(self.nData):
            print('Time:', t_)
            
            # Sample from posterior run length
            self.getRunLengthSamples(t_)      
            
            # Update models and samplers
            self.data2t = self.data[:t_]
            self.updateInferenceModels(t_)
            
            # Get predictive samples
            self.getPredictiveSamples(t_)
            
            # Get predictive statistics
            self.getPredictiveStats(t_)
            
            # Observe new data point
            datat = self.data[t_]
            
            # Check whether changepoint            
            self.checkIfChangepoint(t_, datat)
            
            # Get run length posterior           
            self.getRunLengthProbability(t_, datat)
        
    def getRunLengthSamples(self, t_):
        if t_ != 0:
            rld = stats.rv_discrete( values = ( np.arange( min(t_+1, self.rmax) ), self.rlp ) )
            self.rls = rld.rvs( size = self.nrls )
        else:
            self.rls = np.zeros(self.nrls)
            
    def updateInferenceModels(self, t_):
        if t_ != 0:
            self.trmax = min(t_ + 1, self.rmax)
            results = pool.map( self.updateInferenceModels2Pool, range(1, self.trmax) )
            for r_ in range(1, self.trmax): 
                self.model[r_] = results[r_-1][0]
                self.svn[r_]   = results[r_-1][1]
                self.pts[r_]   = self.svn[r_].particles
                self.ptp[r_]   = np.exp( - self.model[r_].getMinusLogPosterior( self.pts[r_] ) )
                self.ptp[r_]  /= np.sum( self.ptp[r_] )
                
    def updateInferenceModels2Pool(self, r_):
        if r_ < self.trmax - 1:
            model = HAWKES(self.data2t[-r_:], self.data2t[-r_-1])
        else:
            model = HAWKES(self.data2t[-r_:], 0)
        svn = SVN(model, self.pts[r_-1])
        svn.apply()
        return (model, svn)
    
    def getPredictiveSamples(self, t_):
        self.pps = np.array([])     
        r_vals, r_counts = np.unique(self.rls, return_counts = 1) 
        for k_ in range( len(r_vals) ): # PARALLELIZABLE!
            r_val = r_vals[k_]
            r_count = r_counts[k_]
            
            ptd = stats.rv_discrete( values = ( np.arange(self.npts), self.ptp[r_val] ) )
            thetas = self.pts[r_val][:,ptd.rvs(size = r_count)]
            if t_ == 0:
                tmp = self.model[r_val].simulateNewEvent(thetas)
            else:
                tmp = self.data[t_-1] + self.model[r_val].simulateNewEvent(thetas)
            self.pps = np.hstack( ( self.pps, tmp ) )
               
    def getPredictiveStats(self, t_):   
        self.pred_mean[t_] = np.mean(self.pps)
        
        if self.flag_lci == 1:
            self.percentile_l[t_] = np.percentile(self.pps, self.risk_level_l)        
        if self.flag_rci == 1:
            self.percentile_r[t_] = np.percentile(self.pps, 100 - self.risk_level_r)            
            
    def getRunLengthProbability(self, t_, datat):   
        pp0 = np.mean( stats.expon.pdf(datat, scale = np.exp( - self.pts[0][0,:] ) ) )
        pp = np.hstack( (pp0, np.zeros( min(t_, self.rmax-1) )) )
        for r_ in range(1, min(t_, self.rmax)): # PARALLELIZABLE!
            pp[r_] = np.mean( np.exp( - self.model[r_].getPredMinusLogLikelihood( self.pts[r_], datat ) ) )
          
        # Calculate run length posterior
        jppp = self.jp * pp                            # Joint x predictive prob
        gp = jppp * ( 1 - self.H )                     # Growth prob
        cp = np.sum( jppp * self.H )                   # Changepoint prob
        self.jp  = np.hstack( (cp, gp) )[:self.rmax]   # Joint prob
        ep = np.sum( self.jp )                         # Evidence prob
        self.rlp = self.jp / ep                        # Run length posterior
                
    def checkIfChangepoint(self, t_, datat):
        if self.flag_lci:           
            if datat < self.percentile_l[t_]: 
                risk = np.abs( ( datat - self.percentile_l[t_] ) / ( self.pred_mean[t_] - self.percentile_l[t_] ) )
                if risk > self.riskTol:
                    self.changepoints.update( {t_: risk} )   
                    print('Changepoint at time', t_, 'for drastic decrease')
                
        #if self.flag_rci: 
        #    if datat > self.percentile_r[t_]: 
        #        risk = np.abs( ( datat - self.percentile_r[t_] ) / ( self.pred_mean[t_] - self.percentile_r[t_] ) )
        #        if risk > self.riskTol:
        #            self.changepoints.update( {t_: risk} )   
        #            print('Changepoint at time', t_, 'for drastic increase')
                    
            
###########################################################################################################################################            
            
if __name__ == '__main__':
    
    pool = multiprocessing.Pool(40)
    
    # Load WannaCry data
    df = pd.read_csv(r"wannacry_SMB2")

    ttall = df.Time.values
    ttall -= ttall[0]
    tt = ttall[1:]
    
    # Run SVOCD
    bocpd = BOCPD(tt)
    bocpd.apply()
    
    print(bocpd.changepoints)
    
    plt.rcParams.update({'font.size': 24})
    fig = plt.figure(figsize = (15, 10))

    # Plot index against time process and analysis
    ax = plt.subplot(2,1,1)
    ax1 = plt.plot(tt, 'b-', label = 'data')
    ax2 = plt.plot(bocpd.pred_mean, 'r-', label = 'predicted mean')
    ax3 = plt.plot(bocpd.percentile_r, 'g-.', label = 'credible interval')
    plt.plot(bocpd.percentile_l, 'g-.')
    plt.fill_between(np.arange(len(tt)), bocpd.percentile_l, bocpd.percentile_r, alpha = 0.05,\
                    color = 'g')
    for cp in bocpd.changepoints:
        ax3 = plt.axvline(x = cp, color = 'r', linestyle = '--', label = 'changepoint')
    plt.xlabel('number of events', fontsize = 24)
    plt.ylabel('time', fontsize = 24)
    plt.rcParams.update({'font.size': 24})
    handles, labels = ax.get_legend_handles_labels();
    ax.legend(handles[:4], labels[:4], loc = 'upper left', fontsize = 18)

    # Plot counting process and changepoints
    ax = plt.subplot(2,1,2)
    plt.step(tt, np.arange(len(tt)), label = 'counting process')
    for cp in bocpd.changepoints:
        plt.axvline(x = tt[cp], color = 'r', linestyle = '--', label = 'infection time')    
    plt.xlabel('time', fontsize = 24)
    plt.ylabel('count', fontsize = 24)
    plt.rcParams.update({'font.size': 24})
    handles, labels = ax.get_legend_handles_labels();
    ax.legend(handles[:2], labels[:2], loc = 'upper left', fontsize = 18)
        
    fig.tight_layout()
    
    plt.show()
    
    pool.close()
    
