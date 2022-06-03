import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pathlib import Path
from scipy.optimize import curve_fit
from sympy import Eq, Symbol, solve

from .gauss import fit_gaussian_2D, fit_gaussian_1D, fit_2_gauss_sum_to_mip
from .extractor import get_mip
from .util import get_Daans_special_cmap

import tifffile

__all__ = ['get_theta',
           'save_stack',
           'eight_bit_as',
           'read_parameters',
           'get_angle_astigmatism',
           'plot_astigm_angle_fit',
           'get_zrange',
           'two_dim_gauss_fit_all_slices',
           'find_intersect_sigma',
           'plot_sigmas_vs_z',
           'huang',
           'fit_sigmas_vs_z',
           'plot_sigma_vs_z_plus_fit',
           'save_calibration_to_file',
           'solv_astigm_from_focal_line']

# No-brainer global variable
fire = get_Daans_special_cmap()
    
def get_theta(psf, fit_range=10):
    """Get theta from astigmatic PSF.

    Parameters
    ----------
    psf : array-like 
        Image stack of shape (L, M, N)

    Returns
    -------
    theta : float
        Astigmatic angle 
        
    Notes
    -----
    ...
    """
    z_sum = psf.sum(axis=(1, 2))
    z, sigma_z, A, B = fit_gaussian_1D(z_sum)
    
    z0, z1 = round(z - fit_range), round(z + fit_range)
    z = round(z)

    mip0, mip1 = get_mip(psf[z0:z]), get_mip(psf[z:z1])

    popt0 = fit_gaussian_2D(mip0, theta=(0,360))
    popt1 = fit_gaussian_2D(mip1, theta=popt0[4], epsilon=10)

    theta = np.mean([popt0[4], popt1[4]])

    return theta
    

def save_stack(psf, file_pattern,psx,psy,psz,usf):
    """Save PSF to file, along with metadata of stack.
    
    Parameters
    ----------
    psf: array-like
        Image stack of shape (L, M, N)
        
    file_pattern : list or str
        Either a list of filenames or a string that is either:
        a) the individual filename of e.g. a tiff stack or
        b) a directory from which all images will be loaded into the stack
        
    Returns
    -------
    ...
    
    Notes
    -----
    Not tested on multiple files, but should in principle work.
    
    """
   
    if isinstance(file_pattern, list):  # In case a list of files is provided
        location = str(Path(file_pattern[0]).parent) + "/_output"
        fp  = Path(location)
        fp.mkdir(exist_ok=True)
        #save psf to file
        tifffile.imwrite(location + "/psf_av.tif",eight_bit_as(psf,np.uint8),photometric='minisblack')
            
        #save meta data
        with open(location + '/parameters.txt','w') as f:
            f.write('stack parameters:\n')
            f.write('\n')
            f.write('X: '+str(psx/usf)+' nm\n')
            f.write('Y: '+str(psy/usf)+' nm\n')
            f.write('Z: '+str(psz/usf)+' nm\n')
    
    # If a single directory or multipage tiff is provided
    elif isinstance(file_pattern, str):
        if file_pattern[-1] == "/" or file_pattern[-1] == "\\": #directory!
            location = file_pattern + '_output'
            fp  = Path(location)
            fp.mkdir(exist_ok=True) #make output directory if not there
        else:    #file!
            location = str(Path(file_pattern).parent) + "/_output"
            fp  = Path(location)
            fp.mkdir(exist_ok=True) #make output directory if not there
        
        #save psf to file
        tifffile.imwrite(location + "/psf_av.tif",eight_bit_as(psf,np.uint8),photometric='minisblack')
        
        #save meta data
        with open(location + '/parameters.txt','w') as f:
            f.write('stack parameters:\n')
            f.write('\n')
            f.write('X: '+str(psx/usf)+' nm\n')
            f.write('Y: '+str(psy/usf)+' nm\n')
            f.write('Z: '+str(psz/usf)+' nm\n')
    
    print("Succesfully saved PSF and parameters to file.")
    
    return

def eight_bit_as(arr, dtype=np.float32):
    """Convert array to 8 bit integer array.
    
    Parameters
    ----------
    arr: array-like
        Array of shape (L, M, N)
        
    dtype : data type
        Data type of existing array.
        
    Returns
    -------
    arr.astype(dtype) : array-like
        Array formatted to 8 bit integer array
    
    Notes
    -----
    ...
    """
    
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        arr -= arr.min()
        arr *= 255./arr.max()
    else: 
        arr = arr.astype(np.float32)
    return arr.astype(dtype)

def read_parameters(file_pattern):
    """Read parameters from 'parameters.txt' file.
    
    Parameters
    ----------
    file_pattern : list or str
        Either a list of filenames or a string that is either:
        a) the individual filename of e.g. a tiff stack or
        b) a directory from which all images will be loaded into the stack
        
    Returns
    -------
    psx, psy, psz : float
        Values of pixel size  in X, Y and Z of associated image stack
    
    Notes
    -----
    Not tested on multiple files, but should in principle work.
    
    """
    
    if isinstance(file_pattern, str):
        if file_pattern[-1] != "/" and file_pattern[-1] != "\\": #in case file_pattern is a single file
            file_pattern = str(Path(file_pattern).parent)
    
    elif isinstance(file_pattern, list):  # In case a list of files is provided
        file_pattern = str(Path(file_pattern[0]).parent)
    #if single directory, no need to change file_pattern
    
    with open(file_pattern + '/_output/parameters.txt') as f:
        lines = f.readlines()
        
    x_line,y_line,z_line = lines[2],lines[3],lines[4]
    psx,psy,psz=float(x_line[3:-4]), float(y_line[3:-4]), float(z_line[3:-4])
    
    return psx,psy,psz

def get_angle_astigmatism(psf,first_guess_theta=45):
    ''' Obtain angle of astigmatism from stack of a single PSF.
    
    '''
    
    #make maximum intensity projection of PSF
    image = get_mip(psf)
    
    #fit
    theta, fit = fit_2_gauss_sum_to_mip(image,first_guess_theta)
    
    print('Angle of astigmatism:\t', round(theta,1), 'degrees' )
    
    return theta, image, fit

def plot_astigm_angle_fit(image, fit, psx, psy):
    sy, sx = image.shape
    plt.figure(figsize=(10,5))
    plt.subplot(131)
    plt.imshow(image, cmap=fire,extent=[-sx/2*psx*1e-3,sx/2*psx*1e-3,-sy/2*psy*1e-3,sy/2*psy*1e-3])
    plt.title('Maximum intensity projection')
    plt.xlabel('X [μm]')
    plt.ylabel('Y [μm]')
    
    plt.subplot(132)
    plt.imshow(fit, cmap=fire,extent=[-sx/2*psx*1e-3,sx/2*psx*1e-3,-sy/2*psy*1e-3,sy/2*psy*1e-3])
    plt.title('Fit')
    plt.xlabel('X [μm]')
    
    plt.subplot(133)
    plt.imshow(image - fit, cmap=fire,extent=[-sx/2*psx*1e-3,sx/2*psx*1e-3,-sy/2*psy*1e-3,sy/2*psy*1e-3])
    plt.title('MIP - Fit')
    plt.xlabel('X [μm]')
    plt.show()
    return
    
###########################

def get_zrange(psf,dz_nm,psz):
    ### Define fittable z range of PSF to do calibration on
    
    # get size of stack
    z_size, y_size, x_size = psf.shape

    # determine z position of intensity maximum by fitting gaussian to z profile in middle of PSF:
    z_max, _, _, _ = fit_gaussian_1D(psf[:,y_size//2,x_size//2])
    z_max = int(z_max)
    
    # determine z range of calibration curve based on 3*lambda_em / NA
    z_fit = int(0.5*dz_nm / (psz))
    z_range_psf = [z_max-z_fit, z_max+z_fit]

    return z_range_psf

def two_dim_gauss_fit_all_slices(psf, theta, z_range_psf):
    ### fit 2d gaussians to all slice in an image stack in a certain range
    print('Fitting all slices...')
    sigma_x, sigma_y = [], []
    for j in range(z_range_psf[1]-z_range_psf[0]): 
            _, _, xalpha, yalpha, _, _, _ = fit_gaussian_2D(psf[j+z_range_psf[0]-1], theta=theta)
            sigma_x.append(xalpha)
            sigma_y.append(yalpha)
            
    sigma_intersect, z_intersect = find_intersect_sigma(sigma_x, sigma_y)
    return sigma_x, sigma_y,sigma_intersect, z_intersect

def find_intersect_sigma(sigma_x,sigma_y): 
    ### find values of mean sigma and z at intersect of two calibration curves
    
    abs_diff_list=np.abs(np.subtract(sigma_x,sigma_y))
    z_index = abs_diff_list.argmin()
    sigma = np.mean([sigma_x[z_index],sigma_y[z_index]]) 
    return sigma, z_index

def plot_sigmas_vs_z(sigma_x,sigma_y,psx,psz,z_intersect):
    ### plot sigma vs z curves
    z_depth=len(sigma_x)  
    z = np.arange(z_depth)*psz
    z = np.subtract(z, z_intersect*psz) # put z=0 at intersect of curves
        
    plt.scatter(z,np.multiply(sigma_x,psx),label='x',c='black', s=5)
    plt.scatter(z,np.multiply(sigma_y,psx),label='y',c='r', s=5)
    plt.xlabel('Z (nm)')
    plt.ylabel(r'$\sigma_{x,y}$ (nm)')
    plt.legend()
    plt.show()
    return

def huang(z, w_0, d, c, A, B): 
    ### function for fitting calibration curve
    # see bottom p4 https://science.sciencemag.org/content/sci/suppl/2008/01/02/1153529.DC1/Huang.SOM.pdf
    return w_0 * np.sqrt(np.abs(1 + np.power(np.divide(z-c, d), 2) + A*np.power(np.divide(z-c, d), 3) + B*np.power(np.divide(z-c, d), 4)))

def fit_sigmas_vs_z(sigma_x,sigma_y,psx,psz,sigma_intersect,z_intersect,max_sigma_factor=3):
    ### fit calibration curve with huang function

    #make z values
    z_depth=len(sigma_x)  
    z = np.arange(z_depth)*psz
    z = np.subtract(z, z_intersect*psz) # put z=0 at intersect of curves
    
    #determine max_sigma as a multitude of mean minimum sigma and sigma at intersection
    #inspired by: https://opg.optica.org/boe/fulltext.cfm?uri=boe-11-2-735&id=425886
    min_sigma_x, min_sigma_y = np.min(sigma_x),np.min(sigma_y)
    mean_min_sigma = np.mean([min_sigma_x,min_sigma_y])
    max_sigma = mean_min_sigma + (sigma_intersect-mean_min_sigma)* max_sigma_factor

    #filter data to be fitted below max_sigma value:
    z_filtx, z_filty, sigmax_filt, sigmay_filt=[],[],[],[]
    for i in range(len(z)):
        if sigma_x[i] < max_sigma:
            sigmax_filt.append(sigma_x[i])
            z_filtx.append(z[i])
        if sigma_y[i] < max_sigma:
            sigmay_filt.append(sigma_y[i])
            z_filty.append(z[i])
        if i == z_intersect: z_intersect = len(z_filtx)-1 # update location of intersection in filtered z values
    
    # fit sigma vs z
    poptx_sig, pcovx_sig = curve_fit(huang,z_filtx,sigmax_filt, p0=[250, 500, z_filtx[z_intersect], 0, 0], maxfev = 8000)
    popty_sig, pcovy_sig = curve_fit(huang,z_filty,sigmay_filt, p0=[250, 500, z_filty[z_intersect], 0, 0], maxfev = 8000)

    return poptx_sig, popty_sig, max_sigma


def plot_sigma_vs_z_plus_fit(sigma_x,sigma_y,psx,psz,z_intersect,poptx_sig,popty_sig,max_sigma):
    ### plot calibration curve plus fit.

    #make z lists for exp. data and fit
    z_depth=len(sigma_x)  
    z = np.arange(z_depth)*psz
    z_plot_fit = np.linspace(np.min(z), np.max(z), 500)
    z = np.subtract(z, z_intersect*psz) # put z=0 at intersect of curves
    z_plot_fit = np.subtract(z_plot_fit, z_intersect*psz) # put z=0 at intersect of curves
    
    #draw rectangle to indicate data rejected from fit
    plt.gca().add_patch(Rectangle((np.min(z),1.2*psx*np.max([np.max(sigma_x),np.max(sigma_y)])),
                                           np.max(z)-np.min(z),
                                           -1.2*psx*np.max([np.max(sigma_x),np.max(sigma_y)])+max_sigma*psx,
                                           edgecolor='none',
                                           facecolor='gainsboro'))
    #plot exp data
    plt.scatter(z,np.multiply(sigma_x,psx),label='x',c='black', s=5)
    plt.scatter(z,np.multiply(sigma_y,psx),label='y',c='r', s=5)
    #plot fit data
    plt.plot(z_plot_fit,np.multiply(huang(z_plot_fit,*poptx_sig),psx),c='black')
    plt.plot(z_plot_fit,np.multiply(huang(z_plot_fit,*popty_sig),psx),c='red')
    #indicate focal lines
    plt.axvline(x=popty_sig[2],c='r',ls='--')
    plt.axvline(x=poptx_sig[2],c='black',ls='--')
    
    #plot layout
    plt.xlabel('Z (nm)')
    plt.ylabel(r'$\sigma_{x,y}$ (nm)')
    plt.legend()
    plt.title('Calibration curve + fit')
    plt.xlim([np.min(z),np.max(z)])
    plt.ylim([0,1.2*psx*np.max([np.max(sigma_x),np.max(sigma_y)])])
    plt.show()
    
    # get focal line distance
    dz_min = abs(poptx_sig[2]-popty_sig[2])
    
    # get mean focal depth
    d_x, d_y = poptx_sig[1], popty_sig[1]
    d = np.mean([poptx_sig[1],popty_sig[1]])
    
    # get spot size in focus
    sigma_0_x=poptx_sig[0]
    sigma_0_y=popty_sig[0]
    
    #print some stuff
    print("Grey area denotes data range left out of fit.\n")
    print("Distance between focal lines: \t \t",int(dz_min),"nm") 
    print("'A measure of focal depth': \t \t",int(d),"nm") 
    print("Spot size in x in focus (sigma): \t",int(sigma_0_x*psx),"nm") 
    print("Spot size in y in focus (sigma): \t",int(sigma_0_y*psx),"nm")
    
    return dz_min

def save_calibration_to_file(file_pattern,sigma_x,sigma_y,psx,psz,z_intersect,poptx_sig,popty_sig):
    ### Save calibration curves and fit parameters to file.
    
    location = file_pattern+'_output/'
    
    #make z list
    z_depth=len(sigma_x)  
    z = np.arange(z_depth)*psz
    z = np.subtract(z, z_intersect*psz) # put z=0 at intersect of curves
    
    #save calibration curve data in 4 columns: z, sigma_x_exp, sigma_y_exp, sigma_x_fit, sigma_y_fit
    with open(location + '/calibration_curve_plus_fit.txt','w') as f:
        f.write('z (nm) \t sigma_x_exp (nm) \t sigma_y_exp (nm) \t sigma_x_fit (nm) \t sigma_y_fit (nm)')
        for i in range(len(z)):
            line =  (str(z[i]) +'\t' + str(np.multiply(sigma_x[i],psx)) +'\t' +
                    str(np.multiply(sigma_y[i],psx)) + '\t' +
                    str(np.multiply(huang(z[i],*poptx_sig),psx)) + '\t'+
                    str(np.multiply(huang(z[i],*popty_sig),psx))
                    )
            f.write('\n'+line)
    
    #also save fit parameters in separate file
    with open(location + '/calibration_curve_fit_param.txt','w') as f:
        f.write('Fit parameters for Huang function \n' +
                'see SI (bottom page 4) of https://www.science.org/doi/full/10.1126/science.1153529 \n')
        f.write('param \t X \t Y \n')
        f.write('w_0 (nm) \t' + str(poptx_sig[0]*psx) + '\t' + str(popty_sig[0]*psx) + '\n')
        f.write('d (nm) \t' + str(poptx_sig[1]) + '\t' + str(popty_sig[1]) + '\n')
        f.write('c (nm) \t' + str(poptx_sig[2]) + '\t' + str(popty_sig[2]) + '\n')
        f.write('A \t' + str(poptx_sig[3]) + '\t' + str(popty_sig[3]) + '\n')
        f.write('B \t' + str(poptx_sig[4]) + '\t' + str(popty_sig[4]))
    return

def solv_astigm_from_focal_line(focal_line_dist,fit_a,fit_b):
    y = Symbol('y',positive=True)
    l=focal_line_dist/2
    eqn = Eq(fit_a*y**2+fit_b*y-l, y)
    answ = solve(eqn)
    astigm = float(answ[0])
    print('Degree of astigmatism: \t \t', np.round(astigm,3), "lambda")
    return 
    
