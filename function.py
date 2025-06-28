import numpy as np

class ObjectiveFunction:
    """
    Uma classe que encapsula a função objetivo e conta suas avaliações.
    O método __call__ permite que instâncias desta classe sejam chamadas como funções.
    """
    def __init__(self):
        """Inicializa o contador de avaliações."""
        self.evaluations = 0

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
            Array 1D com os valores da função objetivo calculados para cada par (X, Y).
        """

        # Incrementa o contador pelo número de indivíduos/partículas sendo avaliados
        self.evaluations += np.size(X)

        # ==============================================================================
        # ============== 1: Calcular os componentes base (z, r, Fobj) ==================
        # ==============================================================================

        # Componente Z (Função de Schwefel), usa as coordenadas originais
        Z_func = -X * np.sin(np.sqrt(np.abs(X))) - Y * np.sin(np.sqrt(np.abs(Y)))

        # Reescalonar X e Y para os cálculos de Rosenbrock e Ackley/Schaffer
        X_scaled = X / 250.0
        Y_scaled = Y / 250.0
        
        # Componente R (Função de Rosenbrock), usa as coordenadas reescalonadas
        R_func = 100 * (Y_scaled - X_scaled**2)**2 + (1 - X_scaled)**2

        # Cálculo dos componentes para Fobj
        x1 = 25 * X_scaled
        x2 = 25 * Y_scaled
        a = 500
        b = 0.1
        c = 0.5 * np.pi
        
        # Componente F10 (Função de Ackley)
        F10 = -a * np.exp(-b * np.sqrt((x1**2 + x2**2) / 2)) - \
            np.exp((np.cos(c * x1) + np.cos(c * x2)) / 2) + np.exp(1)

        # Componente zsh (Função tipo Schaffer)
        epsilon = 1e-9
        zsh_numerator = (np.sin(np.sqrt(x1**2 + x2**2)))**2 - 0.5
        zsh_denominator = (1 + 0.1 * (x1**2 + x2**2))**2
        zsh = 0.5 - zsh_numerator / (zsh_denominator + epsilon)
        
        # Componente Fobj final
        Fobj = F10 * zsh

        # ==============================================================================
        # ===================== 2: Montar w1, w4 e somá-las ============================
        # ==============================================================================

        # Cálculo de w1
        w1_val = R_func + Z_func

        # Cálculo de w4
        w4_val = np.sqrt(R_func**2 + Z_func**2) + Fobj
        
        # A função objetivo final é a soma das duas
        final_result = w1_val + w4_val
        
        return final_result

    def reset(self):
        """ Reseta o contador de avaliações para uma nova execução de algoritmo. """
        self.evaluations = 0