import numpy as np

class ObjectiveFunction:
    """
    Classe que representa a função objetivo w1 + w4, baseada no código Scilab.
    """
    def __init__(self):
        """Inicializa o contador de avaliações."""
        self.evaluations = 0
        self.multiplications = 0
        self.divisions = 0

    def __call__(self, X, Y):
        """
        Executa o cálculo da função objetivo w1 + w4 e incrementa o contador.
        
        Função objetivo que calcula w1 + w4, baseada no código Scilab.
    
        Onde:
        - r=100*(y-x.^2).^2+(1-x).^2;
        - F10=-a*exp(-b*sqrt((x1.^2+x2.^2)/2))-exp((cos(c*x1)+cos(c*x2))/2)+exp(1);
        - zsh(i,j)=0.5-((sin(sqrt(xs(i)^2+ys(j)^2)))^2-0.5)./(1+0.1*(xs(i)^2+ys(j)^2))^2;
        - Fobj=F10.*zsh//+a*cos(x1/30);
        - a=500; b=0.1; c=0.5*pi;

        Args:
            X: Array representando as coordenadas X.
            Y: Array representando as coordenadas Y.
        Returns:
            final_result: O resultado da função objetivo w1 + w4.
        """

        # Incrementa o contador pelo número de indivíduos/partículas sendo avaliados
        num_elements = np.size(X)
        self.evaluations += num_elements

        # ==============================================================================
        # ============== 1: Calcular os componentes base (z, r, Fobj) ==================
        # ==============================================================================

        # Componente Z (Função de Schwefel), usa as coordenadas originais
        Z_func = -X * np.sin(np.sqrt(np.abs(X))) - Y * np.sin(np.sqrt(np.abs(Y)))
        self.multiplications += 2 * num_elements
        self.divisions += np.size(X)

        # Reescalonar X e Y para os cálculos de Rosenbrock e Ackley/Schaffer
        X_scaled = X / 250.0
        Y_scaled = Y / 250.0
        self.divisions += 2 * num_elements
        
        # Componente R (Função de Rosenbrock), usa as coordenadas reescalonadas
        R_func = 100 * (Y_scaled - X_scaled**2)**2 + (1 - X_scaled)**2
        self.multiplications += 4 * num_elements

        # Cálculo dos componentes para Fobj
        x1 = 25 * X_scaled
        x2 = 25 * Y_scaled
        self.multiplications += 2 * num_elements

        a = 500
        b = 0.1
        c = 0.5 * np.pi
        
        # Componente F10 (Função de Ackley)
        F10 = -a * np.exp(-b * np.sqrt((x1**2 + x2**2) / 2)) - \
            np.exp((np.cos(c * x1) + np.cos(c * x2)) / 2) + np.exp(1)
        self.multiplications += 5 * num_elements
        self.divisions += 2 * num_elements

        # Componente zsh (Função tipo Schaffer)
        epsilon = 1e-9
        zsh_numerator = (np.sin(np.sqrt(x1**2 + x2**2)))**2 - 0.5
        zsh_denominator = (1 + 0.1 * (x1**2 + x2**2))**2
        zsh = 0.5 - zsh_numerator / (zsh_denominator + epsilon)
        self.multiplications += 4 * num_elements
        self.divisions += 1 * num_elements
        
        # Componente Fobj final
        Fobj = F10 * zsh
        self.multiplications += num_elements

        # ==============================================================================
        # ===================== 2: Montar w1, w4 e somá-las ============================
        # ==============================================================================

        # Cálculo de w1
        w1_val = R_func + Z_func

        # Cálculo de w4
        w4_val = np.sqrt(R_func**2 + Z_func**2) + Fobj
        self.multiplications += 2 * num_elements
        
        # A função objetivo final é a soma das duas
        final_result = w1_val + w4_val
        
        return final_result

    def reset(self):
        """ Reseta o contador de avaliações para uma nova execução de algoritmo. """
        self.evaluations = 0
        self.multiplications = 0
        self.divisions = 0