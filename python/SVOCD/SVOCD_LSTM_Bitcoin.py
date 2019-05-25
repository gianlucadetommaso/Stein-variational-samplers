import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
from IPython.display import display, Math, Latex, clear_output
import multiprocessing
from functools import partial
import time
import pandas as pd
import datetime
import pickle

import ICML19_LSTM_M2M
from ICML19_LSTM_M2M import LSTM as LSTM

###########################################################################################################################################
               
### Bayesian long short-term memory (BLSTM) neural network model
    
class BLSTM(LSTM):
    def __init__(self):
        super().__init__()
        
        # Prior setup
        self.mu0 = np.zeros((self.DoF, 1))
        self.std0 = np.ones((self.DoF,1))
        self.var0 = self.std0 ** 2
        self.Prec0 = np.eye(self.DoF)[:,:,np.newaxis] / self.var0
        
        # Noise setup
        self.stdn = 0.1
        self.varn = self.stdn ** 2
        
    def getMinusLogPrior(self, thetas):
        return 0.5 * np.sum( ( thetas - self.mu0 ) ** 2 / self.var0, 0 )
    
    def getGradientMinusLogPrior(self, thetas):
        return  (thetas - self.mu0) / self.var0
    
    def getForwardModel(self, thetas, X):       
        return self.forward_solve(X, thetas.T)
    
    def getMinusLogPredLikelihood(self, thetas, X, y, *arg):
        F = arg[0][np.newaxis, -1,:] if len(arg) > 0 \
            else self.getForwardModel(thetas, X)[np.newaxis, -1,:] 
        return 0.5 * (F - y) ** 2 / self.varn
    
    def getMinusLogLikelihood(self, thetas, Y):
        X = np.hstack( (np.zeros( (1,1) ), Y[:,:-1]) )
        F = self.getForwardModel(thetas, X)
        mllkd = 0.5 * np.sum( (Y.T - F) ** 2, 0 ) / self.varn
        gmllkd, Hmllkd = self.jacobian_forward_solve(X, thetas.T, Y)
        gmllkd /= self.varn
        Hmllkd /= self.varn
        return mllkd, gmllkd.T, np.moveaxis(Hmllkd,0,2)
    
    def getMinusLogPosterior(self, thetas, Y):
        mllkd, gmllkd, Hmllkd = self.getMinusLogLikelihood(thetas, Y)
        mlpt = self.getMinusLogPrior(thetas) + mllkd
        gmlpt = self.getGradientMinusLogPrior(thetas) + gmllkd
        Hmlpt = self.Prec0 + Hmllkd
        return mlpt, gmlpt, Hmlpt
    
    def func4MAP(self, thetas, Y):
        F = self.getForwardModel(thetas, Y)
        mllkd = 0.5 * np.sum( (Y.T - F) ** 2, 0 ) / self.varn
        return (self.getMinusLogPrior(thetas) + mllkd).squeeze()
    
    def getMAP(self, Y, *arg):
        if len(arg) > 0:
            x0 = arg[0] 
        else:
            W, b = self.initialize_weights_and_biases(1)    
            x0 = self.concatenate_weights_and_biases_all_layers(W, b)
        
        res = optimize.minimize(lambda theta: self.func4MAP(theta.reshape(self.DoF, 1), Y = Y), x0, method = 'L-BFGS-B')
        return res.x.reshape(self.DoF, 1)
               
###########################################################################################################################################      
### Stein variational Newton (SVN)

class SVN:
    def __init__(self, model, Y, *arg):
        self.model = model
        self.DoF = model.DoF
        self.nParticles = 30
        self.nIterations = 100
        self.stepsize = 1e-1        
        if len(arg) == 0:
            self.MAP = self.model.getMAP(Y = Y)
            self.resetParticles()
        else:
            self.particles = arg[0]
        
    def apply(self, Y):
        maxshiftold = np.inf
        Q = np.zeros( (self.DoF, self.nParticles) )
        for iter_ in range(self.nIterations):
            
            self.mlpt, gmlpt, Hmlpt = self.model.getMinusLogPosterior(self.particles, Y)
            M = np.mean(Hmlpt, 2) / self.DoF
            for i_ in range(self.nParticles):
                sign_diff = self.particles[:,i_,np.newaxis] - self.particles
                Msd   = np.matmul(M, sign_diff)
                kern  = np.exp( - 0.5 * np.sum( sign_diff * Msd, 0 ) )
                gkern = Msd * kern
                
                mgJ = np.mean(- gmlpt * kern + gkern , 1)
                HJ  = np.mean(Hmlpt * kern ** 2, 2) + np.matmul(gkern, gkern.T) / self.nParticles
                Q[:,i_] = np.linalg.solve(HJ, mgJ)
            self.particles += self.stepsize * Q
                        
            maxshift = np.linalg.norm(Q, np.inf)
            if np.isnan(maxshift) or (maxshift > 1e20):
                self.resetParticles(np.arange(self.nParticles))
                self.stepsize = 1
            elif maxshift < maxshiftold:
                self.stepsize *= 1.01
            else:
                self.stepsize *= 0.9
            maxshiftold = maxshift
                          
    def resetParticles(self):
        self.particles = self.MAP + np.random.normal( scale = 0.1, size = (self.DoF, self.nParticles) )
                   
########################################################################################################################################### 
### Stein variational Online Changepoint Detection (SVOCD)
##  Generalized BOCPD via SVN   
    
class BOCPD(BLSTM):
    def __init__(self, data):
        super().__init__()
        self.model = BLSTM()
        
        # Import data
        self.data = data
        self.nData = len(self.data)
        
        # Setup computational effort
        self.rmax = 30     # Max run length 
        self.npts = 30     # Number of posterior samples
        self.nrls = 100    # Number of run length samples
        
        # Setup credible interval
        self.flag_lci = 1      # Test left tail -> fire up drastic drecrease
        self.flag_rci = 1      # Test right tail -> fire up drastic increase
        self.risk_level_l = 2.5    # Left percentage of probability risk     
        self.risk_level_r = 2.5    # Right percentage of probability risk
        self.pred_mean = np.zeros(self.nData)                         # Predictive mean
        if self.flag_lci: self.percentile_l = np.zeros(self.nData)    # Left percentile (if flagged up)
        if self.flag_rci: self.percentile_r = np.zeros(self.nData)    # Right percentile (if flagged up)
                    
        # Changepoint prior
        self.rlr = 1000             # Run length rate hyper-parameter -> decrease to weigh changepoint prob more 
        self.H   = 1 / self.rlr     # Hazard rate
        
        # Initialize run length probabilities
        self.jp  = 1          # Joint 
        self.rlp = self.jp    # Posterior
        
        # Initialize posterior samples and probabilities
        self.pts = {}             
        self.pts[0] = self.mu0 + self.std0 * np.random.normal( size = (self.DoF, self.npts) )       
        self.ptp = {}             
        self.ptp[0] = np.exp( - 0.5 * np.sum( ( self.pts[0] - self.mu0 ) ** 2 / self.var0, 0 ) )
        self.ptp[0] /= np.sum(self.ptp[0])
        
        # Risk tolerance
        self.riskTol = 0
        
        # Initialize changepoints
        self.changepoints = {}
        
        self.svn = {}
    
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
        self.trmax = min(t_ + 1, self.rmax)
        if t_ != 0:
            results = pool.map( self.updateInferenceModels2Pool, range(1, self.trmax) )
            for r_ in range(1, self.trmax): 
                self.pts[r_]   = results[r_-1][0]      
                Y = self.data2t[-r_:].reshape(1,r_)
                self.ptp[r_]   = 1 / (1 + results[r_-1][1])
                sumptp = np.sum( self.ptp[r_] )
                if sumptp < 1e-20: 
                    self.ptp[r_] = np.ones(self.npts) / self.npts
                else:
                    self.ptp[r_] /= sumptp
                
    def updateInferenceModels2Pool(self, r_):
        model = BLSTM()
        Y = self.data2t[-r_:].reshape(1,r_)
        svn = SVN(model, Y, self.pts[r_ - 1])
        svn.apply(Y)
        return (svn.particles, svn.mlpt)
    
    def getPredictiveSamples(self, t_):
        self.pps = np.zeros( (1,0) )  
        r_vals, r_counts = np.unique(self.rls, return_counts = 1) 
        for k_ in range( len(r_vals) ): # PARALLELIZABLE!
            r_val = r_vals[k_]
            r_count = r_counts[k_]
            
            ptd = stats.rv_discrete( values = ( np.arange(self.npts), self.ptp[r_val] ) )
            thetas = self.pts[r_val][:,ptd.rvs(size = r_count)]
            X = np.zeros( (1,1) )
            if r_val > 0: 
                X = np.hstack( (X, self.data2t[-r_val:].reshape(1,r_val)) )
            F = self.model.getForwardModel(thetas, X)[-1, :]
            tmp = F + self.stdn * np.random.normal( size = (1, r_count) )
            self.pps = np.hstack( ( self.pps, tmp ) )
       
    def getPredictiveStats(self, t_):   
        self.pred_mean[t_] = np.mean(self.pps)
        
        if self.flag_lci == 1:
            self.percentile_l[t_] = np.percentile(self.pps, self.risk_level_l)        
        if self.flag_rci == 1:
            self.percentile_r[t_] = np.percentile(self.pps, 100 - self.risk_level_r)
                      
    def getRunLengthProbability(self, t_, datat):   
        pp = np.zeros(self.trmax)
        for r_ in range(len(pp)): # PARALLELIZABLE!
            X = np.zeros( (1,1) )    
            if r_ > 0: X = np.hstack( (X, self.data2t[-r_:].reshape(1,r_)) )
            pp[r_] = np.mean( np.exp( - self.model.getMinusLogPredLikelihood( self.pts[r_], X, datat ) ) )
          
        # Calculate run length posterior
        jppp = self.jp * pp                            # Joint x predictive prob
        gp = jppp * ( 1 - self.H )                     # Growth prob
        cp = np.sum( jppp * self.H )                   # Changepoint prob
        self.jp  = np.hstack( (cp, gp) )[:self.rmax]   # Joint prob
        ep = np.sum( self.jp )                         # Evidence prob
        if ep < 1e-20:
            self.jp = np.ones(len(self.jp)) / len(self.jp)
            ep = 1.0
        self.rlp = self.jp / ep                        # Run length posterior
               
    def checkIfChangepoint(self, t_, datat):
        if self.flag_lci:           
            if datat < self.percentile_l[t_]: 
                risk = np.abs( ( datat - self.percentile_l[t_] ) / ( self.pred_mean[t_] - self.percentile_l[t_] ) )
                if risk > self.riskTol:
                    self.changepoints.update( {t_: risk} )   
                    print('Changepoint at time', t_, 'for drastic decrease')
                
        if self.flag_rci: 
            if datat > self.percentile_r[t_]: 
                risk = np.abs( ( datat - self.percentile_r[t_] ) / ( self.pred_mean[t_] - self.percentile_r[t_] ) )
                if risk > self.riskTol:
                    self.changepoints.update( {t_: risk} )   
                    print('Changepoint at time', t_, 'for drastic increase')
                               
###########################################################################################################################################            
            
if __name__ == '__main__':
    
    pool = multiprocessing.Pool(40)
    
    # Load Bitcoin data
    data_bitcoin = pd.read_csv("bitcoin_price.txt", sep="\t")
    data_bitcoin['Timestamp2'] = pd.to_datetime(data_bitcoin['Timestamp'])
    data_bitcoin = data_bitcoin.drop(data_bitcoin.index[[22,23,24]])
    data_bitcoin["Weighted Price"] = pd.to_numeric(data_bitcoin["Weighted Price"])
    data_bitcoin['date'] = pd.to_datetime(data_bitcoin['Timestamp2'].dt.date)
    data_all = data_bitcoin[379:].reset_index(drop = True)[["date", "Weighted Price"]].set_index('date').rolling(min_periods = 1, 
                                                                   window=7).mean()["Weighted Price"].interpolate()
    data = data_all.values
    days = data_all.index

    data -= np.mean(data)
    data /= np.std(data)  
  
    # Run SVOCD
    bocpd = BOCPD(data)
    bocpd.apply()
    
#     # Save model
#     output_file = open('bocpd_model_svn.pkl', 'wb')
#     pickle.dump(bocpd, output_file)    
    
    print(bocpd.changepoints)
    
    ### Plot ###
    
    plt.rcParams.update({'font.size': 24})

    # Make dataframe
    df = pd.DataFrame({'data' : data,\
                       'predicted mean' : bocpd.pred_mean, \
                       'credible interval' : bocpd.percentile_r, \
                       'credible interval tmp': bocpd.percentile_l}, index = days)

    # Plot day against scaled price
    my_colours = ['b', 'r', 'g', 'g']
    my_styles  = ['-', '-', '--', '--']
    ax = df.plot(legend = False, figsize = (15, 10), color=my_colours, style = my_styles)

    plt.fill_between(days, bocpd.percentile_l, bocpd.percentile_r, alpha = 0.05, color = 'g')


    for cp in bocpd.changepoints:
        xvalue = days[cp]
        ax3 = plt.axvline(x = xvalue, color = 'r', linestyle = '--', label = 'changepoint')
    plt.xlabel('', fontsize = 24)
    plt.ylabel('scaled price', fontsize = 24)
    handles, labels = ax.get_legend_handles_labels();
    ax.legend(handles[:3] + [handles[4]], labels[:3] + [labels[4]], loc = 'upper left', fontsize = 24)
    axes = plt.gca()
    axes.set_ylim([-0.5,4])

    day_min = datetime.datetime(2017, 9, 1)
    day_max = datetime.datetime(2018, 5, 1)
    axes.set_xlim([day_min, day_max])
    
    ax.locator_params(axis='y', nbins=6)

    plt.show()
   
    ### End Plot ###

    pool.close()
    
