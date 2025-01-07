# Collecting the functions used in the EMS Proceedings publication
# Credits go to Matteo Borgnino and Agostino Niyonkuru Meroni for setting up
# the codes initially. Then Alessandro joined the research group and expanded them.



# This is the Gaussian kernel to low-pass filter the input field
# field format : [latitude, longitude]
# sigma : Gaussian standard deviation
#
# returns : smoother (low-pass) field

def nan_gaussian_filter(field,sigma):
    """
    Function to smooth the field ignoring the NaNs.
    I follow the first answer here 
    https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    By default, the filter is truncated at 4 sigmas.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter
    
    field = np.double(field)
    
    # Take the original field and replace the NaNs with zeros.
    field0 = field.copy()
    field0[np.isnan(field)] = 0
    
    
    
#   If the sigma provided is 'inf', 
#   the function returns the average over the whole computational domain
    if sigma == 'inf':
        return np.nanmean(field, axis=(0,1))
    
    elif sigma > 0:
        ff = gaussian_filter(field0, sigma=sigma)
    
        # Create the smoothed weight field.
        weight = 0*field.copy()+1
        weight[np.isnan(field)] = 0
        ww = gaussian_filter(weight, sigma=sigma)

        zz = ff/(ww*weight) # This rescale for the actual weights used in the filter and set to NaN where the field
                            # was originally NaN.
        zz[np.isinf(zz)] = np.nan
        return zz
    
#   If the sigma provided is zero, the function just returns the input field
    elif sigma == 0:
        return field
        

        
        
# this is to reconstruct figure 2, panel (d)
# i.e. the linear regression between SST' and LHF'
# format of x and y : [time, latitude, longitude]
# arguments are
#  - x : independent variable
#  - y : dependent variable
#  - nt : no. of points to skip in time coordinate, according to autocorrelation in time
#         we used nt = 1 in the article.
#  - nskip : no. of points to skip in latitude/longitude, according to autocorrelation in space
#            we used nskip = 15 (30km of autocorrelation divided by 2km model gridspacing)
#  - ls (bool) : if True, returns results in Python list format

# make sure to have scipy among your packages

# outputs:
# linreg : fit results, contains fit slope and regression
# corr_coeff : linear correlation coefficient
# p_value : resulting p value from two-tailed T test
# sigmas : tuple with uncertainties on the regressed slope and intercept
  
def slopes_r_p_mix(x, y, nt, nskip, ls=False):
    from scipy import stats
    import numpy as np
    
    xx = x[::nt,::nskip,::nskip]
    yy = y[::nt,::nskip,::nskip]
    
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    xx = xx[~np.isnan(xx)]
    yy = yy[~np.isnan(yy)]
    
    linreg = stats.linregress(x,y)
    corr_coeff, trash = stats.spearmanr(x,y)
    
    df = np.size(xx)-2
    mean_x = np.mean(x);  mean_x2 = np.mean(x**2); lever_arm = mean_x2-mean_x**2
    
    sigma_y = np.sqrt( np.sum(  (y-linreg.slope*x-linreg.intercept)**2 )/df  )
    sigma_slope = sigma_y/( np.sqrt(np.size(xx)*(lever_arm) ) )
    sigma_intercept = sigma_y*np.sqrt(mean_x2/( np.size(xx)*(lever_arm) ))
    
    sigmas = (sigma_slope, sigma_intercept)
    
    t_value = linreg.slope/sigma_slope
    p_value = 2*(1 - stats.t.cdf(t_value_cannelli,df=df))
    
    if ls:
        return [linreg, corr_coeff, p_value, sigmas]
    else:
        return linreg, corr_coeff, p_value, sigmas


    
    
    
# In order to display the 2D density plot and regression line
# you may use function density_hexbin()
    
# x, y : fields to be regressed
# plot_fit : if True, regression metrics can be added as arguments to plot the fitting line
# fit : stats.linregress(x,y) object, contains fit slope and intercept
# corcoe : linear correlation coefficient
# title,xlabel,ylabel : set plot title and labels of axes
# pos : position [horizontal, vertical] in axis fraction of the fit characteristics in text format
# slope_units : string with fit slope units 

def density_hexbin(x,y,plot_fit,fit,corcoe,grdsz=100,title,xlabel,ylabel,colormap='inferno',pos, slope_units=None):
    
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    im = plt.hexbin(x, y, gridsize=grdsz, bins='log', cmap=colormap, mincnt=1) # 
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.title(title, fontsize=16)
    cbar = plt.colorbar(im) 
    cbar.set_label(label='counts [$log_{10}N$]', fontsize=15)
    
    if (fit is not None):
        if plot_fit:
            xx = np.linspace(x.min(), x.max(), 5)
            plt.plot(xx, fit.slope*xx+fit.intercept, '-', linewidth=3, color='cyan')
        
        if np.abs( np.log10(fit.slope) ) > 2. :
            ff2 = "{:.2e}".format
            plt.annotate(r'slope = '+ str(ff2(fit.slope))+str(slope_units), xy=(pos[0],pos[1]), \
                             xycoords='axes fraction', fontsize=14, color='k')
        else: 
             plt.annotate(r'slope = '+ str(round(fit.slope,2))+str(slope_units), xy=(pos[0],pos[1]), \
                             xycoords='axes fraction', fontsize=14, color='k')
        
            
    if corcoe is not None:
        plt.annotate('corr = '+str(round(corcoe,2)), xy=(pos[0],pos[1]-0.05), \
                                 xycoords='axes fraction', fontsize=14, color='k')
    









####################################  FUNCTIONS TO RECREATE VERTICAL SECTIONS  ######################################

# this function accepts two fields of interest
# x and y,
# x is a [time, latitude, longitude] field
# y is a [time, vertical_level, latitude, longitude] field
# 
# then it builds percentile distributions by time step and level
# with perc_step (width of percentile bins) and nbins ( 100/perc_step ) 
# SEE FUNCTION BELOW perc_distribution_pvalue_dof()

# this function also tests statistical significance within each bin with respect to
# popmean = 0
# according to autocorrelation lengths below and above the vertical level
# top = 14

# paper setting:
# nskip = 15
# nt = 1
# nskiptop = 75
# nttop = 1

# X and Y FORMAT : [time, vertical_level, latitude, longitude]

def dist_3d_subsample(x,y,perc_step, nbins, popmean=0, nt, nttop, nskip, nskiptop, top):
    
    import numpy as np
    
    dist_y = np.zeros((y.shape[1],nbins))
    std_y = np.zeros((y.shape[1],nbins))
    stderr_y = np.zeros((y.shape[1],nbins))
    pvalue_y_sub = np.zeros((y.shape[1],nbins))
    npoints_y = np.zeros_like(dist_y)
    
        for h in range(0,y.shape[1]):
        if h % 10 == 0:
            print(h)    
        xx = x.copy(); control = xx.reshape(-1)
        yy = y[:,h].copy(); variable = yy.reshape(-1)

        ##### Perc bin distribution: pvalue and stderr subsampled on Lcorr
        if h <= top:
            control_sub = x[::nt,::nskip,::nskip].copy();           control_sub = control_sub.reshape(-1)
            var_sub = y[::nt,h,::nskip,::nskip].copy();             var_sub = var_sub.reshape(-1)
        else:
            control_sub = x[::nttop,::nskiptop,::nskiptop].copy();  control_sub = control_sub.reshape(-1)
            var_sub = y[::nttop,h,::nskiptop,::nskiptop].copy();    var_sub = var_sub.reshape(-1)

        ##### Perc bin distribution: pvalue
        dist_x, dist_y[h], std_y[h], stderr_y[h], npoints_y[h], pvalue_y_sub[h] = perc_distribution_pvalue_dof(control, variable, control_sub, var_sub, nbins, perc_step, popmean)

            
    return dist_x, dist_y, std_y, stderr_y, npoints_y, pvalue_y_sub




# Percentile distributions linking two fields i.e. control and variable , 
# can be constructed with this function. This function is key in dist_3d_subsample()
# input arguments' characteristics
#
# control and variable : [latitude, longitude] fields, e.g. SST' and T'
# control_sub, var_sub : [latitude, longitude] SUBSAMPLED fields, see dist_3d_subsample()
# perc_step : width of the percentile bin
# nbins = int(100/perc_step)
# popmean : reference population mean to evaluate p value with. We want to test when anomalies are
#           different from popmean = 0

def perc_distribution_pvalue_dof(control, variable, control_sub, var_sub, nbins, perc_step, popmean):
    
    from scipy import stats
    import numpy as np
    
    # memory alloc
    distribution_control = np.zeros(nbins)
    distribution = np.zeros(nbins)
    
    std_distribution = np.zeros(nbins)
    std_err_distribution = np.zeros(nbins)
    
    number_of_points = np.zeros(nbins)
    number_of_points_sub = np.zeros(nbins)
    
    percentiles = np.zeros(nbins+1)
    percentiles_sub = np.zeros(nbins+1)
    p_value = np.zeros(nbins)

    for pp in range(0,100,perc_step):
        qq = int(pp/perc_step)
        
        lower = np.percentile(control[~np.isnan(control)],pp)
        upper = np.percentile(control[~np.isnan(control)],pp+perc_step)
        percentiles[qq] = lower
        
        #  SUBSAMPLED STATISTICS
        lower_sub = np.percentile(control_sub[~np.isnan(control_sub)],pp)
        upper_sub = np.percentile(control_sub[~np.isnan(control_sub)],pp+perc_step)
        percentiles_sub[qq] = lower_sub
        
        
        # stats computation
        cond_mean_control = np.nanmean(control[(control>=lower)&(control<upper)])
        cond_mean = np.nanmean(variable[(control>=lower)&(control<upper)])
        cond_std = np.nanstd(variable[(control>=lower)&(control<upper)])
        
        distribution_control[qq] = cond_mean_control 
        distribution[qq] = cond_mean                          
        std_distribution[qq] = cond_std
                 
        number_of_points[qq] = np.sum(~np.isnan(variable[(control>=lower)&(control<upper)]))
        number_of_points_sub[qq] = np.sum(~np.isnan(var_sub[(control_sub>=lower_sub)&(control_sub<upper_sub)]))
        
        std_err_distribution[qq] = std_distribution[qq]/np.sqrt(number_of_points_sub[qq])
        
        #################      CORRECT VERSION   ###################
        dof =  number_of_points_sub[qq]  #number_of_points[qq]
        t_stat = (distribution[qq] - popmean)/std_err_distribution[qq]
        p_value[qq] = 2*(1 - stats.t.cdf(np.abs(t_stat), df=dof))
        
    return distribution_control, distribution, std_distribution, std_err_distribution, number_of_points, p_value



####################################  FUNCTIONS TO BUILD SINGLE-LEVEL PERCENTILE BINS ######################################

# these functions were used to plot the MABLH in the vertical sections plots
# either x and y are [time, latitude, longitude]
# perc_step : width of percentile bins
# nbins = 100/perc_step
# paper setting : perc_step = 5 (each bin contains 5% of data)
# no subsampling is necessary here


def distrib_2d(x, y, perc_step, nbins, popmean=0):
    xx = x.copy(); control = xx.reshape(-1)
    yy = y.copy(); variable = yy.reshape(-1)

    ##### Perc bin distribution: pvalue
    distr_x, distr_y, std_y, stderr_y, npoints_y, pvalue_y = perc_distribution_pvalue(control, variable, nbins, perc_step, popmean)

    return distr_x, distr_y, std_y, stderr_y, npoints_y, pvalue_y




def perc_distribution_pvalue(control, variable, nbins, perc_step, popmean=0):
    from scipy import stats
    distribution = np.zeros(nbins)
    std_distribution = np.zeros(nbins)
    std_err_distribution = np.zeros(nbins)
    distribution_control = np.zeros(nbins)
    number_of_points = np.zeros(nbins)
    percentiles = np.zeros(nbins+1)
    p_value = np.zeros(nbins)

    for pp in range(0,100,perc_step):
        qq = int(pp/perc_step)
        lower = np.percentile(control[~np.isnan(control)],pp)
        upper = np.percentile(control[~np.isnan(control)],pp+perc_step)
        percentiles[qq] = lower
        cond_mean = np.nanmean(variable[(control>=lower)&(control<upper)])
        cond_std = np.nanstd(variable[(control>=lower)&(control<upper)])
        cond_mean_control = np.nanmean(control[(control>=lower)&(control<upper)])
        distribution[qq] = cond_mean#-mean
        std_distribution[qq] = cond_std
        distribution_control[qq] = cond_mean_control#-mean_control
        number_of_points[qq] = np.sum(~np.isnan(variable[(control>=lower)&(control<upper)]))
        std_err_distribution[qq] = std_distribution[qq]/np.sqrt(number_of_points[qq])
        
        t_stat, p_value[qq] = stats.ttest_1samp(variable[(control>=lower)&(control<upper)], popmean=popmean)
        
    return distribution_control, distribution, std_distribution, std_err_distribution, number_of_points, p_value




























