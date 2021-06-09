#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 10:37:00 2021

@author: neus
"""

import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit  
import warnings
from iminuit.util import make_func_code
from iminuit import describe #, Minuit,
from scipy import stats

def format_value(value, decimals):
    """ 
    Checks the type of a variable and formats it accordingly.
    Floats has 'decimals' number of decimals.
    """
    
    if isinstance(value, (float, np.float)):
        return f'{value:.{decimals}f}'
    elif isinstance(value, (int, np.integer)):
        return f'{value:d}'
    else:
        return f'{value}'


def values_to_string(values, decimals):
    """ 
    Loops over all elements of 'values' and returns list of strings
    with proper formating according to the function 'format_value'. 
    """
    
    res = []
    for value in values:
        if isinstance(value, list):
            tmp = [format_value(val, decimals) for val in value]
            res.append(f'{tmp[0]} +/- {tmp[1]}')
        else:
            res.append(format_value(value, decimals))
    return res


def len_of_longest_string(s):
    """ Returns the length of the longest string in a list of strings """
    return len(max(s, key=len))

def nice_string_output(d, extra_spacing=5, decimals=3):
    """ 
    Takes a dictionary d consisting of names and values to be properly formatted.
    Makes sure that the distance between the names and the values in the printed
    output has a minimum distance of 'extra_spacing'. One can change the number
    of decimals using the 'decimals' keyword.  
    """
    
    names = d.keys()
    max_names = len_of_longest_string(names)
    
    values = values_to_string(d.values(), decimals=decimals)
    max_values = len_of_longest_string(values)
    
    string = ""
    for name, value in zip(names, values):
        spacing = extra_spacing + max_values + max_names - len(name) - 1 
        string += "{name:s} {value:>{spacing}} \n".format(name=name, value=value, spacing=spacing)
    return string[:-2]


def add_text_to_ax(x_coord, y_coord, string, ax, fontsize=12, color='k'):
    """ Shortcut to add text to an ax with proper font. Relative coords."""
    ax.text(x_coord, y_coord, string, family='monospace', fontsize=fontsize,
            transform=ax.transAxes, verticalalignment='top', color=color)
    return None

def set_var_if_None(var, x):
    if var is not None:
        return np.array(var)
    else: 
        return np.ones_like(x)
    
def compute_f(f, x, *par):
    
    try:
        return f(x, *par)
    except ValueError:
        return np.array([f(xi, *par) for xi in x])
    
def simpson38(f, edges, bw, *arg):
    
    yedges = f(edges, *arg)
    left38 = f((2.*edges[1:]+edges[:-1]) / 3., *arg)
    right38 = f((edges[1:]+2.*edges[:-1]) / 3., *arg)
    
    return bw / 8.*( np.sum(yedges)*2.+np.sum(left38+right38)*3. - (yedges[0]+yedges[-1]) ) #simpson3/8

    
def integrate1d(f, bound, nint, *arg):
    """
    compute 1d integral
    """
    edges = np.linspace(bound[0], bound[1], nint+1)
    bw = edges[1] - edges[0]
    
    return simpson38(f, edges, bw, *arg)


def xlogyx(x, y):
    
    #compute x*log(y/x) to a good precision especially when y~x
    
    if x<1e-100:
        warnings.warn('x is really small return 0')
        return 0.
    
    if x<y:
        return x*np.log1p( (y-x) / x )
    else:
        return -x*np.log1p( (x-y) / y )


#compute w*log(y/x) where w < x and goes to zero faster than x
def wlogyx(w, y, x):
    if x<1e-100:
        warnings.warn('x is really small return 0')
        return 0.
    if x<y:
        return w*np.log1p( (y-x) / x )
    else:
        return -w*np.log1p( (x-y) / y )
    
class UnbinnedLH:  # override the class with a better one
    
    def __init__(self, f, data, weights=None, bound=None, badvalue=-100000, extended=False, extended_bound=None, extended_nint=100):
        
        if bound is not None:
            data = np.array(data)
            mask = (data >= bound[0]) & (data <= bound[1])
            data = data[mask]
            if (weights is not None) :
                weights = weights[mask]

        self.f = f  # model predicts PDF for given x
        self.data = np.array(data)
        self.weights = set_var_if_None(weights, self.data)
        self.bad_value = badvalue
        
        self.extended = extended
        self.extended_bound = extended_bound
        self.extended_nint = extended_nint
        if extended and extended_bound is None:
            self.extended_bound = (np.min(data), np.max(data))

        
        self.func_code = make_func_code(describe(self.f)[1:])

    def __call__(self, *par):  # par are a variable number of model parameters
        
        logf = np.zeros_like(self.data)
        
        # compute the function value
        f = compute_f(self.f, self.data, *par)
    
        # find where the PDF is 0 or negative (unphysical)        
        mask_f_positive = (f>0)

        # calculate the log of f everyhere where f is positive
        logf[mask_f_positive] = np.log(f[mask_f_positive]) * self.weights[mask_f_positive] 
        
        # set everywhere else to badvalue
        logf[~mask_f_positive] = self.bad_value
        
        # compute the sum of the log values: the LLH
        llh = -np.sum(logf)
        
        if self.extended:
            extended_term = integrate1d(self.f, self.extended_bound, self.extended_nint, *par)
            llh += extended_term
        
        return llh
    
    def default_errordef(self):
        return 0.5

def compute_bin_lh_f2(f, edges, h, w2, extended, use_sumw2, nint_subdiv, *par):
    
    N = np.sum(h)
    n = len(edges)

    ret = 0.
    
    for i in range(n-1):
        th = h[i]
        tm = integrate1d(f, (edges[i], edges[i+1]), nint_subdiv, *par)
        
        if not extended:
            if not use_sumw2:
                ret -= xlogyx(th, tm*N) + (th-tm*N)

            else:
                if w2[i]<1e-200: 
                    continue
                tw = w2[i]
                factor = th/tw
                ret -= factor*(wlogyx(th,tm*N,th)+(th-tm*N))
        else:
            if not use_sumw2:
                ret -= xlogyx(th,tm)+(th-tm)
            else:
                if w2[i]<1e-200: 
                    continue
                tw = w2[i]
                factor = th/tw
                ret -= factor*(wlogyx(th,tm,th)+(th-tm))

    return ret



class BinnedLH:  # override the class with a better one
    
    def __init__(self, f, data, bins=40, weights=None, weighterrors=None, bound=None, badvalue=1000000, extended=False, use_w2=False, nint_subdiv=1):
        
        if bound is not None:
            data = np.array(data)
            mask = (data >= bound[0]) & (data <= bound[1])
            data = data[mask]
            if (weights is not None) :
                weights = weights[mask]
            if (weighterrors is not None) :
                weighterrors = weighterrors[mask]

        self.weights = set_var_if_None(weights, data)

        self.f = f
        self.use_w2 = use_w2
        self.extended = extended

        if bound is None: 
            bound = (np.min(data), np.max(data))

        self.mymin, self.mymax = bound

        h, self.edges = np.histogram(data, bins, range=bound, weights=weights)
        
        self.bins = bins
        self.h = h
        self.N = np.sum(self.h)

        if weights is not None:
            if weighterrors is None:
                self.w2, _ = np.histogram(data, bins, range=bound, weights=weights**2)
            else:
                self.w2, _ = np.histogram(data, bins, range=bound, weights=weighterrors**2)
        else:
            self.w2, _ = np.histogram(data, bins, range=bound, weights=None)


        
        self.badvalue = badvalue
        self.nint_subdiv = nint_subdiv
        
        
        self.func_code = make_func_code(describe(self.f)[1:])
        self.ndof = np.sum(self.h > 0) - (self.func_code.co_argcount - 1)
        

    def __call__(self, *par):  # par are a variable number of model parameters

        # ret = compute_bin_lh_f(self.f, self.edges, self.h, self.w2, self.extended, self.use_w2, self.badvalue, *par)
        ret = compute_bin_lh_f2(self.f, self.edges, self.h, self.w2, self.extended, self.use_w2, self.nint_subdiv, *par)
        
        return ret


    def default_errordef(self):
        return 0.5

class Chi2Regression:  # override the class with a better one
        
    def __init__(self, f, x, y, sy=None, weights=None, bound=None):
        
        if bound is not None:
            x = np.array(x)
            y = np.array(y)
            sy = np.array(sy)
            mask = (x >= bound[0]) & (x <= bound[1])
            x  = x[mask]
            y  = y[mask]
            sy = sy[mask]

        self.f = f  # model predicts y for given x
        self.x = np.array(x)
        self.y = np.array(y)
        
        self.sy = set_var_if_None(sy, self.x)
        self.weights = set_var_if_None(weights, self.x)
        self.func_code = make_func_code(describe(self.f)[1:])

    def __call__(self, *par):  # par are a variable number of model parameters
        
        # compute the function value
        f = compute_f(self.f, self.x, *par)
        
        # compute the chi2-value
        chi2 = np.sum(self.weights*(self.y - f)**2/self.sy**2)
        
        return chi2

def create_1d_hist(ax, values, bins, x_range, title,label=None):
    ax.hist(values, bins, x_range, density=False, lw=2,label=label)  # histtype='step'       
    ax.set(xlim=x_range, title=title)
    hist_data = np.histogram(values, bins, x_range)
    return hist_data

def simple_plot(x,y,xlabel=None,ylabel=None,xrange=None,yrange=None,figpath=None,ax=None,title=None):
    
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.plot(x,y,'.')
    
    if (xlabel is not None): ax.set_xlabel(xlabel)
    if (ylabel is not None): ax.set_ylabel(ylabel)
    if (xrange is not None): ax.set_xlim(xrange)
    if (yrange is not None): ax.set_ylim(yrange)
    if (title is not None): ax.set_title(title)
    
    if (figpath is not None) and (ax is None):
        fig.savefig(figpath)
        return fig
    elif (figpath is None) and (ax is None):
        return fig
    else:
        return ax

def unit_gaussian(x,mu,sigma):
    N = ( 1/(np.sqrt(2*np.pi)*sigma) )
    return N*np.exp(-(x-mu)**2/(2*sigma**2))

def gauss_extended(x, N, binwidth, mu, sigma):
    """Non-normalized Gaussian"""
    return N * binwidth * unit_gaussian(x, mu, sigma)

def moyal_distr(x,m,s):
    lamb = (x-m)/s
    return np.exp( -(lamb+np.exp(-lamb))/2. ) / np.sqrt(2*np.pi) 

def moyal_distr_extended(x,N,binwidth,m,s):   
    return N*binwidth*stats.moyal.pdf(x, loc=m, scale=s) #moyal_distr(x,m,s)

def plot_fit_gaussian(data,Nbins,ax=None,fit_ullh=None,format_N=None,format_mu=None,format_sigma=None,format_p=None,format_binwidth=None):
    
    if (ax is None):
        fig,ax=plt.subplots(figsize=(10,5))
    
    if (format_mu is None): format_mu="{:1.2f}+/-{:1.2f}"
    if (format_N is None): format_N="{:1.0f}+/-{:1.0f}"
    if (format_sigma is None): format_sigma="{:1.2f}+/-{:1.2f}"
    if (format_p is None): format_p="{:.4f}"
    if (format_binwidth is None): format_binwidth="{:1.3f}+/-{:1.3f}"
    
    xmin,xmax=np.min(data),np.max(data)
    y, edges = np.histogram(data,bins=Nbins,range=[xmin,xmax])
    x = (edges[:-1] + edges[1:])/2
    sy = np.sqrt(y)
    bin_width = (xmax-xmin)/Nbins
    
    #Fiting with chi2
    mask = (y>0)
    chi2_object = Chi2Regression(gauss_extended,x[mask],y[mask],sy[mask])
    minuit_chi2 = Minuit(chi2_object,N=len(data),binwidth=bin_width,mu=np.mean(data),sigma=np.std(data),fix_binwidth=True,pedantic=False,print_level=0)
    minuit_chi2.migrad();
    
    chi2_value = minuit_chi2.fval
    Ndof = len(y[mask])-minuit_chi2.nfit#(len(minuit_chi2.values)-1)
    Pchi2 = stats.chi2.sf(chi2_value,Ndof)
    
    ax.errorbar(x[mask],y[mask],yerr=sy[mask],fmt='.',label='Binned data',color='gray',alpha=0.5)
    
    x_fit=np.linspace(xmin,xmax,100)
    y_fit_chi2 = gauss_extended(x_fit,*minuit_chi2.args)
    ax.plot(x_fit,y_fit_chi2,label=r'$\chi^2$ gaussian fit',color='blue')
    
    d = {'N': format_N.format(minuit_chi2.values['N'], minuit_chi2.errors['N']),
        'binwidth': format_binwidth.format(minuit_chi2.values['binwidth'],minuit_chi2.errors['binwidth']),
         r'$\mu$': format_mu.format(minuit_chi2.values['mu'],minuit_chi2.errors['mu']),
         r'$\sigma$': format_sigma.format(minuit_chi2.values['sigma'],minuit_chi2.errors['sigma']),
         r'$\chi^2$/Ndof': "{:.2f}/{:d}".format(chi2_value,Ndof),
         r'P($\chi^2$)': format_p.format(Pchi2)}    
    add_text_to_ax(0.02,0.95,nice_string_output(d,0),ax,fontsize=12,color='blue')

    
    #Fitting with ullh
    if (fit_ullh is not False):
        ullh_object = UnbinnedLH(gauss_extended,data,weights=None)
        minuit_ullh = Minuit(ullh_object,N=len(data),binwidth=1.,mu=np.mean(data),sigma=np.std(data),fix_binwidth=True,fix_N=True,pedantic=False,print_level=0)
        minuit_ullh.migrad()

        y_fit_ullh = gauss_extended(x_fit,*minuit_ullh.args)*bin_width
        ax.plot(x_fit,y_fit_ullh,label=r'ULLH gaussian fit',color='green')
        d = {'N': format_N.format(minuit_ullh.values['N'], minuit_ullh.errors['N']),
             r'$\mu$': format_mu.format(minuit_ullh.values['mu'],minuit_ullh.errors['mu']),
             r'$\sigma$': format_sigma.format(minuit_ullh.values['sigma'],minuit_ullh.errors['sigma']),}
        add_text_to_ax(0.67,0.75,nice_string_output(d,0),ax,fontsize=12,color='green')
    else:
        minuit_ullh=None
    ax.legend()
    
    return minuit_chi2,minuit_ullh,ax