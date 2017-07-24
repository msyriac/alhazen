
stamp_size_arc ~ 100
k_size_arc ~ 30 or 100
px_high ~ 0.1

kappa = NFW(M,c,z,k_size_arc)

if periodic:
    unlensed = grf(cosmology,stamp_size_arc,px_high)
else:
    unlensed = upsample(cutout(saved_map,loc_index),px_high)


beam_arc ~ 1.5
px_low = 0.5

lensed = lens(unlensed,kappa)
lensed_beam = beam_smooth(lensed,beam_arc)
lensed_beam_sampled = downsample(lensed_beam,px_low)

# SLOW PART ENDS

noise_map = gen_noise(stamp_size_arc,uKarcmin,lknee,alpha)
measured_map = lensed_beam_sampled + noise_map

# RECON STARTS

noise_psd = psd(beam,wnoise,lknee,alpha,pixel)
filters = init_filters(cosmology,noise_psd)


decon_map = low_pass(deconvolve(measured_map,beam,pixel))

kappa_recon = recon(decon_map,filters)

    


class Pipeline(object):

    def __init__(self):
        pass
