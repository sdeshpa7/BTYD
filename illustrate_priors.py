import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def illustrate_priors():
    x = np.linspace(0, 100, 1000)
    
    # 1. Half-Normal Prior (Informative)
    # This is what we used for p and q
    sigma = 20
    half_normal = stats.halfnorm.pdf(x, scale=sigma)
    
    # 2. Half-Flat Prior (Uninformative/Default)
    # In math, this is a constant value from 0 to infinity
    # Because the area must sum to 1, a true flat prior over infinity is 'improper'
    # We represent it as a flat line.
    half_flat = np.ones_like(x) * 0.01 
    
    plt.figure(figsize=(12, 6))
    
    # Plotting
    plt.plot(x, half_normal, label=f'Half-Normal Prior (sigma={sigma})', lw=3, color='#2980b9')
    plt.plot(x, half_flat, label='Half-Flat Prior (Constant)', lw=3, color='#e74c3c', linestyle='--')
    
    # Aesthetics
    plt.fill_between(x, half_normal, alpha=0.2, color='#2980b9')
    plt.title('Comparison of Bayesian Priors', fontsize=16, fontweight='bold')
    plt.xlabel('Parameter Value (e.g., p, q, or v)', fontsize=12)
    plt.ylabel('Prior Probability Density', fontsize=12)
    
    # Annotations
    plt.annotate('Concentrates the search\nin a "reasonable" range', 
                 xy=(15, 0.015), xytext=(40, 0.018),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.annotate('Allows the model to drift\nto infinity if not careful', 
                 xy=(80, 0.01), xytext=(60, 0.005),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.legend()
    plt.grid(True, alpha=0.1)
    plt.ylim(0, 0.025)
    
    output_path = "prior_illustration.png"
    plt.savefig(output_path)
    print(f"Illustration saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    illustrate_priors()
