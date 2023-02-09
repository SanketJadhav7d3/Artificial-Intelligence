
def deriv(func, input, delta=0.001):
    '''
        Calculates derivative of function
    '''
    return (func(input + delta) - func(input - delta)) / (2 * delta)
