#Decimation and interpolation
import numpy as np

def my_convolution(x, y, mode='full'):
    '''
    Perform convolution between input signal x and impulse response y.
    
    Parameters:
    x: input signal (array)
    y: impulse response (array)
    mode: 'full' or 'same' 
        'full': returns the complete convolution result
        'same': returns the central part of the convolution result with the same length as x

    Returns:
    Convolution result (array)
    '''
    n = len(x)
    m = len(y)
    # Padding the input signal x with zeros on both sides to handle border effects during convolution
    padded_x = np.pad(x, (m-1, m-1), 'constant')
    # Flip the impulse response y
    flip_y = np.flip(y)
    # Calculate the length of the convolution result
    result_length = n + m - 1
    # Initialize the result array with zeros
    result = np.zeros(result_length)
    # Perform the convolution operation
    for i in range(result_length):
        result[i] = np.sum(padded_x[i:i+m] * flip_y)
    if mode == 'full':
        return result
    elif mode == 'same':
        # Return the central part of the result to match the length of the input signal
        return result[m//2:m//2+n]
    return result

def my_interpolation_LPF(L, LPF_type):
    '''
    Create an interpolation low-pass filter based on the specified type.
    
    Parameters:
    L: interpolation factor (int)
    LPF_type: type of low-pass filter ('shanon', 'ZOH', 'FOH')

    Returns:
    Impulse response of the interpolation filter (array)
    '''
    if LPF_type == 'shanon':
        # Generate an index array for the filter
        n = np.arange(-20*L, 20*L+1)
        # Create a sinc filter for Shannon interpolation
        h = L * np.sin(np.pi/L*n) / (np.pi*n)
        # Handle the singularity at n=0
        h[n == 0] = 1
    elif LPF_type == 'ZOH':
        # Create a boxcar (rectangular) filter for Zero-Order Hold interpolation
        h = np.ones(L)
    elif LPF_type == 'FOH':
        # Create a triangular filter for First-Order Hold interpolation
        h = np.ones(L)
        h = my_convolution(h, h, mode='full')
        # Normalize the filter
        h = h / np.max(h)
    return h

def down_sample(x, M):
    '''
    Down-sample the input signal by a factor of M.
    
    Parameters:
    x: input signal (array)
    M: decimation factor (int)

    Returns:
    Down-sampled signal (array)
    '''
    return x[::M]  # Take every M-th sample from the input signal

def decimate(x, M):
    '''
    Decimate the input signal by a factor of M, including anti-aliasing filtering.
    
    Parameters:
    x: input signal (array)
    M: decimation factor (int)

    Returns:
    Decimated signal (array)
    '''
    # Generate an index array for the anti-aliasing filter
    n = np.arange(-20*M, 20*M+1)
    # Create a sinc filter for anti-aliasing
    h = np.sin(np.pi/M*n) / (np.pi*n)
    # Handle the singularity at n=0
    h[n == 0] = 1/M
    # Convolve the input signal with the anti-aliasing filter
    y = my_convolution(x, h, mode='same')
    # Down-sample the filtered signal by a factor of M
    y = down_sample(y, M)
    return y

def up_sample(x, L):
    '''
    Up-sample the input signal by a factor of L, inserting zeros between samples.
    
    Parameters:
    x: input signal (array)
    L: interpolation factor (int)

    Returns:
    Up-sampled signal (array)
    '''
    # Create an array of zeros with length L times the length of the input signal
    y = np.zeros(L*len(x))
    # Insert the input signal values into the up-sampled array at intervals of L
    y[::L] = x
    return y

def interpolate(x, L, filter_type):
    '''
    Interpolate the input signal by a factor of L using the specified filter type.
    
    Parameters:
    x: input signal (array)
    L: interpolation factor (int)
    filter_type: type of interpolation filter ('shanon', 'ZOH', 'FOH')

    Returns:
    Interpolated signal (array)
    '''
    # Up-sample the input signal by inserting zeros
    y = up_sample(x, L)
    # Create the interpolation filter based on the specified type
    h = my_interpolation_LPF(L, filter_type)
    # Convolve the up-sampled signal with the interpolation filter
    y = my_convolution(y, h, mode='same')
    return y
