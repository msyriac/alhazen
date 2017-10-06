import alhazen.correlation_functions as corr
import numpy as np

ntheta = 100
flavor = 'cltt'
lmin = 100
lmax = 3000

ells = np.arange(lmin,lmax)

clCMB = np.ones(ells.size)
clpp = np.ones(ells.size)
clCMB_lensed = np.ones(ells.size)
accurate_lensing = True
compute_tgradt = True

l2range = range(lmin,lmax)
l2dim = len(l2range)-1

b = corr.derivative_dclcmbdclpp_corr_func(lmax,flavor,accurate_lensing,clCMB,clpp,l2range,lmin,lmax,l2dim,lmax)


a = corr.lensed_spectra_corrfunc_allsky(ntheta,flavor,accurate_lensing,compute_tgradt,clCMB,clpp,clCMB_lensed,lmin,lmax)

# subroutine lensed_spectra_corrfunc_allsky(ntheta,flavor,accurate_lensing,compute_tgradt,clCMB,clpp,clCMB_lensed,lmin,lmax)

# block='TTTT'

# ## Maximum multipole
# lmax = int(lmax)

# ## Input lensing potential power spectrum
# clpp = cls_unlensed.clpp_long


# ACCURACY_BOOST 		 = 4 # 4 is conservative. oversampling of the theta grid
# accurate_lensing 	 = True # Gauss-Legendre (true) vs simple integration (False)
# EXTRA_MULTIPOLES = 0 # If you want to add more multipole to compute dC^{CMB}/dC^{\phi \phi} (not necessary)

# 	#############################################################
# 	# Initialization of containers
# 	#############################################################
# 	cov_order0_tot = np.array([covmat(0,lmax) for i in range(len(blocks))]) ## Gaussian variance
# 	cov_order1_tot = np.array([covmat(0,lmax) for i in range(len(blocks))]) ## O(clpp)
# 	cov_order2_tot = np.array([covmat(0,lmax) for i in range(len(blocks))]) ## O(clpp^2)

# 	## We have 4 derivatives dC^{CMB}/dC^{phiphi} to compute: TTTT, EEEE, BBBB, TETE
# 	names_dCMB_over_dcpp_tot = ['TTTT','EEEE','BBBB','TETE']
# 	dCMB_over_dcpp_tot = np.array([np.zeros((lmax+1,lmax+1)) for i in range(len(names_dCMB_over_dcpp_tot))])


# 	## We approximate other derivatives (e.g. dlensedEE/dEE) by ones.
# 	dBB_over_dEE_tot = np.zeros((lmax+1,lmax+1))

# 	flavor = 'cl%s%s'%(block[0].lower(),block[2].lower())

# 		if block in ['EEBB','TTBB','TEBB']:
# 			## There is no Gaussian variance contribution for those terms
# 			continue

# 		if rank==0: print 'Gaussian Variance: doing block %s (%s)\n'%(block,flavor)

# 		## Load weights (lensed spectra, and their noisy version)
# 		if block in ['TETE', 'TTTE', 'EETE']:
# 			cl_len_XX, cl_len_YY, cl_len_XY = lib_spectra.load_weights(cls_lensed, 'clte',
# 										noise_uK_arcmin, fwhm_arcmin, 2*lmax, extra='_long',TTcorr=TTcorr)
# 		else:
# 			cl_len_XX, cl_len_YY, cl_len_XY = lib_spectra.load_weights(cls_lensed, flavor,
# 										noise_uK_arcmin, fwhm_arcmin, 2*lmax, extra='_long',TTcorr=TTcorr)

# 		## Gaussian variance
# 		if block == 'TTEE':
# 			cov_order0_tot[index_block].data = cross_gaussian_variance(cl1=cl_len_XY[1],cl2=cl_len_XY[1],ells=cls_lensed.ls)
# 		elif block == 'TTTE':
# 			cov_order0_tot[index_block].data = cross_gaussian_variance(cl1=cl_len_XX[1],cl2=cl_len_XY[1],ells=cls_lensed.ls)
# 		elif block == 'EETE':
# 			cov_order0_tot[index_block].data = cross_gaussian_variance(cl1=cl_len_YY[1],cl2=cl_len_XY[1],ells=cls_lensed.ls)
# 		else:
# 			cov_order0_tot[index_block].data = gaussian_variance(cl11=cl_len_XX[1],cl22=cl_len_YY[1],
# 												cl12=cl_len_XY[1],ells=cls_lensed.ls)


# 	#############################################################
# 	## Contribution of the trispectrum to the covariance: O(clpp)
# 	#############################################################
# 	for block in blocks:
# 		index_block = blocks.index(block)
# 		flavor = 'cl%s%s'%(block[0].lower(),block[2].lower())

# 		if block in ['BBBB','TTBB','EEBB','TETE','TTEE','TTTE','EETE','TEBB']:
# 			## We do not consider the contribution of those terms (although you can)
# 			continue
# 		cov_order1 = covmat(0,lmax)

# 		if rank==0: print 'Order O(clpp): doing block %s (%s)\n'%(block,flavor)

# 		## Load spins
# 		spinl2_x, spinl3_x, spinl2_y, spinl3_y = lib_spectra.load_spin_values_wigner('clte')

# 		## Load weights (lensed spectra, and their noisy version)
# 		cl_unlen_TT, cl_unlen_EE, cl_unlen_TE = lib_spectra.load_weights(cls_unlensed, 'clte',
# 										noise_uK_arcmin, fwhm_arcmin, 2*lmax, extra='_long',TTcorr=TTcorr)
# 		cl_unlen_vec = np.array([cl_unlen_TT[0], cl_unlen_EE[0], np.zeros_like(cl_unlen_TT[0]), cl_unlen_TE[0]])

# 		## Load weights (unlensed spectra)
# 		uup = block[0] + block[2]
# 		vvp = block[1] + block[3]
# 		uvp = block[0] + block[3]
# 		vup = block[1] + block[2]

# 		## Define range of ells, and distribute over procs.
# 		n_tot = comm.size
# 		l2range = range(lmin+rank,lmax+1,n_tot)
# 		l2dim = len(l2range)-1

# 		## Compute this term
# 		cov_order1.data = loop_lensing.covariance_cmbxcmb_order1_uvupvp(cl_unlen_vec,clpp,l2range,
# 									uup,vvp,uvp,vup,lmin,spinl2_x,spinl3_x,spinl2_y,spinl3_y,l2dim,lmax)
# 		comm.Barrier()

# 		## Reduce the results on the root
# 		comm.Reduce([cov_order1.data, MPI.DOUBLE],[cov_order1_tot[index_block].data, MPI.DOUBLE],op = MPI.SUM,root = 0)

# 		## Done for this block
# 		comm.Barrier()

# 	#############################################################
# 	## Compute dC^{CMB}/dC^{phiphi}
# 	## Here, you have two ways of computing the derivatives:
# 	## 		* Using series-expansion. Quick but less accurate.
# 	## 		* Using correlation functions. Less quick, but extra accurate.
# 	#############################################################
# 	file_manager_derivatives_CMB = util.file_manager('dCMB_over_dcpp_tot', exp, spec='v1', lmax=lmax,
# 													force_recomputation=False, folder=folder_cache,rank=rank)
# 	if file_manager_derivatives_CMB.FileExist is True:
# 		if rank==0:
# 			dCMB_over_dcpp_tot, names_dCMB_over_dcpp_tot = file_manager_derivatives_CMB.data
# 	else:
# 		for position_block,block in enumerate(names_dCMB_over_dcpp_tot):
# 			flavor = 'cl%s%s'%(block[0].lower(),block[1].lower())
# 			if rank==0: print 'Pre-compute derivatives for block %s (%s)\n'%(block,flavor)

# 			if not use_corrfunc:
# 				if rank == 0: print 'Use series-expansion to compute derivative (may not be exact)'
# 				if block == 'BBBB':
# 					## BB takes clee as unlensed weights (noiseless!)
# 					cl_unlen_XX, cl_unlen_YY, cl_unlen_XY = lib_spectra.load_weights(cls_unlensed, 'clee', 0.0,
# 																0.0, 2*lmax, extra='_long')
# 				else:
# 					## noiseless!
# 					cl_unlen_XX, cl_unlen_YY, cl_unlen_XY = lib_spectra.load_weights(cls_unlensed, flavor,
# 																0.0, 0.0, 2*lmax, extra='_long')

# 				## Define range of ells, and distribute over procs.
# 				n_tot = comm.size
# 				l2range = range(lmin+rank,lmax+1,n_tot)
# 				l2dim = len(l2range)-1

# 				## Load spins
# 				spinl2_x, spinl3_x, spinl2_y, spinl3_y = lib_spectra.load_spin_values_wigner(flavor)

# 				## Change order of spins. Why? do not know... but it works :D
# 				derivatives = loop_lensing.compute_derivatives_dcttdcpp_mpi(cl_unlen_XY[0],l2range,flavor,lmin,spinl3_x,
# 												spinl2_x,spinl3_y,spinl2_y,l2dim,lmax)

# 				## Reduce on the root
# 				comm.Reduce([derivatives, MPI.DOUBLE],[dCMB_over_dcpp_tot[position_block], MPI.DOUBLE],op = MPI.SUM,root = 0)
# 			else:
# 				if rank == 0: print 'Use correlation functions to compute derivative'
# 				if block == 'BBBB':
# 					## BB takes clee as unlensed weights
# 					## noiseless!
# 					cl_unlen_XX, cl_unlen_YY, cl_unlen_XY = lib_spectra.load_weights(cls_unlensed, 'clee',
# 																0.0, 0.0, 2*lmax, extra='_long')
# 				else:
# 					## noiseless!
# 					cl_unlen_XX, cl_unlen_YY, cl_unlen_XY = lib_spectra.load_weights(cls_unlensed, flavor,
# 																0.0, 0.0, 2*lmax, extra='_long')

# 				clpp_long = cls_unlensed.clpp_long
# 				## Define container used to reduce results on root
# 				derivatives_tot_tmp = np.zeros((lmax+1,lmax+1))

# 				## Define range of ells, and distribute over procs.
# 				n_tot = comm.size
# 				l2range = range(lmin+rank,lmax+1,n_tot)
# 				l2dim = len(l2range)-1

# 				## Compute.
# 				## Use cl_unlen_XY which is TT for TT, EE for EE, EE for BB, and TE for TE.
# 				dxim,dxip,dm,dp = correlation_functions.derivative_dclcmbdclpp_corr_func(ACCURACY_BOOST*lmax,flavor,accurate_lensing,cl_unlen_XY[0][0:lmax+EXTRA_MULTIPOLES+1],clpp_long[0:lmax+EXTRA_MULTIPOLES+1],l2range,lmin,lmax,l2dim,lmax+EXTRA_MULTIPOLES)

