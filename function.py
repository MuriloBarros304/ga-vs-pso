import numpy as np

class ObjectiveFunction:
    """
    Classe que representa uma função objetivo.
    Limites de entrada esperados: [-500, 500] para ambas.
    
    Suporta:
    - 'schwefel_rosenbrock': Soma completa (Original Scilab)
    - 'rastrigin': Função clássica, adaptada para receber entrada [-500, 500]
    """
    def __init__(self, target_func='schwefel_rosenbrock'):
        self.evaluations = 0
        self.multiplications = 0
        self.divisions = 0
        self.target_func = target_func

    def __call__(self, X, Y):
        """
        Calcula o valor da função.
        Args:
            X, Y: Arrays numpy com coordenadas. Esperado intervalo [-500, 500].
        """
        num_elements = np.size(X)
        self.evaluations += num_elements

        # ==============================================================================
        # ========================== FUNÇÃO RASTRIGIN ==================================
        # ==============================================================================
        if self.target_func == 'rastrigin':
            # A Rastrigin padrão opera em [-5.12, 5.12].
            # Como o GA/PSO vai enviar valores em [-500, 500], precisamos comprimir
            # a entrada para manter a geometria correta da função.
            
            # Fator de escala: 5.12 / 500 = 0.01024
            scale_factor = 5.12 / 500.0
            
            X_scaled = X * scale_factor
            Y_scaled = Y * scale_factor
            
            self.divisions += 2 * num_elements # Contabiliza a divisão do fator de escala
            
            A = 10
            # Parte X
            comp_x = X_scaled**2 - A * np.cos(2 * np.pi * X_scaled)
            self.multiplications += 3 * num_elements # x^2, 2*pi*x, A*cos
            
            # Parte Y
            comp_y = Y_scaled**2 - A * np.cos(2 * np.pi * Y_scaled)
            self.multiplications += 3 * num_elements 
            
            result = 2 * A + comp_x + comp_y
            
            return result

        # ==============================================================================
        # ========================= (Schwefel-Rosenbrock) ==============================
        # ==============================================================================
        # Esta função foi desenhada para operar nativamente em [-500, 500]
        # Devido ao componente Schwefel (Z_func).
        
        # Componente Z (Schwefel)
        Z_func = -X * np.sin(np.sqrt(np.abs(X))) - Y * np.sin(np.sqrt(np.abs(Y)))
        self.multiplications += 2 * num_elements
        self.divisions += np.size(X) # Sqrt

        # Reescalonamento interno das variáveis para Rosenbrock
        X_scaled = X / 250.0
        Y_scaled = Y / 250.0
        self.divisions += 2 * num_elements
        
        # Componente R (Rosenbrock)
        R_func = 100 * (Y_scaled - X_scaled**2)**2 + (1 - X_scaled)**2
        self.multiplications += 4 * num_elements

        # w1 = Rosenbrock + Schwefel
        w1_val = R_func + Z_func
        
        # Cálculos para W4 (Ackley/Schaffer mix)
        x1 = 25 * X_scaled
        x2 = 25 * Y_scaled
        self.multiplications += 2 * num_elements

        a = 500
        b = 0.1
        c = 0.5 * np.pi
        
        # Componente F10 (Ackley)
        F10 = -a * np.exp(-b * np.sqrt((x1**2 + x2**2) / 2)) - \
            np.exp((np.cos(c * x1) + np.cos(c * x2)) / 2) + np.exp(1)
        self.multiplications += 5 * num_elements
        self.divisions += 2 * num_elements

        # Componente zsh (Schaffer)
        epsilon = 1e-9
        zsh_numerator = (np.sin(np.sqrt(x1**2 + x2**2)))**2 - 0.5
        zsh_denominator = (1 + 0.1 * (x1**2 + x2**2))**2
        zsh = 0.5 - zsh_numerator / (zsh_denominator + epsilon)
        self.multiplications += 4 * num_elements
        self.divisions += 1 * num_elements
        
        # Fobj
        Fobj = F10 * zsh
        self.multiplications += num_elements

        # w4
        w4_val = np.sqrt(R_func**2 + Z_func**2) + Fobj
        self.multiplications += 2 * num_elements
        
        return w1_val + w4_val # Retorna a soma completa

    def reset(self):
        """ Reseta os contadores. """
        self.evaluations = 0
        self.multiplications = 0
        self.divisions = 0