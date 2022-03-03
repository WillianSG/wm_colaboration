from __future__ import (print_function, division,
                        absolute_import, unicode_literals)

import numpy as np

"""
The Global synchrony measure is described in:

Li et al. Synchronization measurement of multiple neuronal populations. J Neurophysiol, 2007:3341
Patel et al. Dynamic changes in neural circuit topology following mild mechanical injury in vitro. Annals of
biomedical engineering, 2012:23
Patel et al. Single-neuron NMDA receptor phenotype influences neuronal rewiring and reintegration following traumatic
injury. J Neurosci, 2014:4200
"""


def get_phase_spikes( spikes, tfinal ):
    """
    calculate spike phase as defined in Patel 2012
    """
    phase = 2 * np.pi * np.random.rand( tfinal )  # np.zeros(tfinal)
    k = 0
    numspikes = len( spikes )
    for t in range( 1, tfinal + 1 ):
        while (k < numspikes) and (t >= spikes[ k ]):
            k += 1
        
        if k == 0:
            phase[ t - 1 ] = 2 * np.pi * (t - spikes[ 0 ]) / spikes[ 0 ]
        elif k == numspikes:
            # phase[t-1] = 0
            pass
        else:
            phase[ t - 1 ] = (2 * np.pi * (t - spikes[ k - 1 ]) /
                              (spikes[ k ] - spikes[ k - 1 ])) + (2 * np.pi * k)
    
    return phase


def calc_phase_sync_matrix( spike_list, tfinal ):
    """
    calculate the correlation matrix using spike phase
    """
    phase_list = list()
    for spikes in spike_list:
        phase = get_phase_spikes( spikes, tfinal )
        phase_list.append( phase )
    
    M = len( spike_list )
    C = np.zeros( (M, M) )
    for i in range( M ):
        for j in range( i, M ):
            if not ((len( spike_list[ i ] ) == 0) or (len( spike_list[ j ] ) == 0)):
                delta_phi = np.mod( phase_list[ i ] - phase_list[ j ], 2 * np.pi )
                C[ i, j ] = np.sqrt( np.square( np.mean( np.cos( delta_phi ) ) ) +
                                     np.square( np.mean( np.sin( delta_phi ) ) ) )
                C[ j, i ] = C[ i, j ]
    
    return C, phase_list


def calc_global_sync( spike_list, tfinal ):
    C, phase_list = calc_phase_sync_matrix( spike_list, tfinal )
    (D, V) = np.linalg.eig( C )
    order = np.argsort( D )
    
    eigenvals = D[ order ].copy()
    M = len( spike_list )
    index = (np.max( eigenvals ) - 1) / (M - 1)
    return index


'''

Comment by Thomas Kreuz:

This Python code (including all further comments) was written by Jeremy Fix (see http://jeremy.fix.free.fr/),
based on Matlab code written by Thomas Kreuz.

The SPIKE-distance is described in this paper:

Kreuz T, Chicharro D, Houghton C, Andrzejak RG, Mormann F:
Monitoring spike train synchrony.
J Neurophysiol 109, 1457-1472 (2013).

The Matlab codes as well as more information can be found at
http://www.fi.isc.cnr.it/users/thomas.kreuz/sourcecode.html.

'''

import numpy as np

'''
Return the times (t1,t2) of the spikes in train[ibegin:]
such that t1 < t and t2 >= t
'''


def find_corner_spikes( t, train, ibegin, ti, te ):
    if (ibegin == 0):
        tprev = ti
    else:
        tprev = train[ ibegin - 1 ]
    for idts, ts in enumerate( train[ ibegin: ] ):
        if (ts >= t):
            return np.array( [ tprev, ts ] ), idts + ibegin
        tprev = ts
    return np.array( [ train[ -1 ], te ] ), idts + ibegin


def bivariate_spike_distance( t1, t2, ti, te, N ):
    '''Computes the bivariate SPIKE distance of Kreuz et al. (2012)
       t1 and t2 are 1D arrays with the spiking times of two neurones
       It returns the array of the values of the distance
       between time ti and te with N samples.
       The arrays t1, t2 and values ti, te are unit less '''
    t = np.linspace( ti + (te - ti) / N, te, N )
    d = np.zeros( t.shape )
    
    t1 = np.insert( t1, 0, ti )
    t1 = np.append( t1, te )
    t2 = np.insert( t2, 0, ti )
    t2 = np.append( t2, te )
    
    # We compute the corner spikes for all the time instants we consider
    # corner_spikes is a 4 column matrix [t, tp1, tf1, tp2, tf2]
    corner_spikes = np.zeros( (N, 5) )
    
    ibegin_t1 = 0
    ibegin_t2 = 0
    corner_spikes[ :, 0 ] = t
    for itc, tc in enumerate( t ):
        corner_spikes[ itc, 1:3 ], ibegin_t1 = find_corner_spikes( tc, t1, ibegin_t1, ti, te )
        corner_spikes[ itc, 3:5 ], ibegin_t2 = find_corner_spikes( tc, t2, ibegin_t2, ti, te )
    
    # print corner_spikes
    xisi = np.zeros( (N, 2) )
    xisi[ :, 0 ] = corner_spikes[ :, 2 ] - corner_spikes[ :, 1 ]
    xisi[ :, 1 ] = corner_spikes[ :, 4 ] - corner_spikes[ :, 3 ]
    norm_xisi = np.sum( xisi, axis=1 )**2.0
    
    # We now compute the smallest distance between the spikes in t2 and the corner spikes of t1
    # with np.tile(t2,(N,1)) we build a matrix :
    # np.tile(t2,(N,1)) =    [   t2   ]        -   np.tile(reshape(corner_spikes,(N,1)), t2.size) = [
    # ]
    #                        [   t2   ]                                                             [  corner  corner
    #                        corner]
    #                        [   t2   ]                                                             [
    #                        ]
    dp1 = np.min( np.fabs( np.tile( t2, (N, 1) ) - np.tile( np.reshape( corner_spikes[ :, 1 ], (N, 1) ), t2.size ) ),
                  axis=1 )
    df1 = np.min( np.fabs( np.tile( t2, (N, 1) ) - np.tile( np.reshape( corner_spikes[ :, 2 ], (N, 1) ), t2.size ) ),
                  axis=1 )
    # And the smallest distance between the spikes in t1 and the corner spikes of t2
    dp2 = np.min( np.fabs( np.tile( t1, (N, 1) ) - np.tile( np.reshape( corner_spikes[ :, 3 ], (N, 1) ), t1.size ) ),
                  axis=1 )
    df2 = np.min( np.fabs( np.tile( t1, (N, 1) ) - np.tile( np.reshape( corner_spikes[ :, 4 ], (N, 1) ), t1.size ) ),
                  axis=1 )
    
    xp1 = t - corner_spikes[ :, 1 ]
    xf1 = corner_spikes[ :, 2 ] - t
    xp2 = t - corner_spikes[ :, 3 ]
    xf2 = corner_spikes[ :, 4 ] - t
    
    S1 = (dp1 * xf1 + df1 * xp1) / xisi[ :, 0 ]
    S2 = (dp2 * xf2 + df2 * xp2) / xisi[ :, 1 ]
    
    d = (S1 * xisi[ :, 1 ] + S2 * xisi[ :, 0 ]) / (norm_xisi / 2.0)
    
    return t, d


def multivariate_spike_distance( spike_trains, ti, te, N ):
    ''' t is an array of spike time arrays
    ti the initial time of the recordings
    te the end time of the recordings
    N the number of samples used to compute the distance
    spike_trains is a list of arrays of shape (N, T) with N spike trains
    The multivariate distance is the instantaneous average over all the pairwise distances
    '''
    d = np.zeros( (N,) )
    n_trains = len( spike_trains )
    t = 0
    for i, t1 in enumerate( spike_trains[ :-1 ] ):
        for t2 in spike_trains[ i + 1: ]:
            tij, dij = bivariate_spike_distance( t1, t2, ti, te, N )
            if (i == 0):
                t = tij  # The times are only dependent on ti, te, and N
            d = d + dij
    d = d / float( n_trains * (n_trains - 1) / 2 )
    return t, d
