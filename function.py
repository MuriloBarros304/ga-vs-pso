import numpy as np

# w4 adaptada do Scilab
def objective_function_w4(X, Y):
    Z_func = -X * np.sin(np.sqrt(np.abs(X))) - Y * np.sin(np.sqrt(np.abs(Y)))

    # Reescalonar X e Y ANTES de calcular a função de Rosenbrock
    X_scaled = X / 250.0
    Y_scaled = Y / 250.0
    
    # Função de Rosenbrock (r) é calculada com os valores reescalonados
    R_func = 100 * (Y_scaled - X_scaled**2)**2 + (1 - X_scaled)**2

    x1 = 25 * X_scaled
    x2 = 25 * Y_scaled
    a = 500
    b = 0.1
    c = 0.5 * np.pi
    
    F10 = -a * np.exp(-b * np.sqrt((x1**2 + x2**2) / 2)) - np.exp((np.cos(c * x1) + np.cos(c * x2)) / 2) + np.exp(1)

    # Cálculo do ZSH (Zhang-Shen-Huang)
    epsilon = 1e-9
    zsh_numerator = (np.sin(np.sqrt(x1**2 + x2**2)))**2 - 0.5
    zsh_denominator = (1 + 0.1 * (x1**2 + x2**2))**2
    zsh = 0.5 - zsh_numerator / (zsh_denominator + epsilon)
    
    Fobj = F10 * zsh

    w4 = np.sqrt(R_func**2 + Z_func**2) + Fobj
    
    return w4