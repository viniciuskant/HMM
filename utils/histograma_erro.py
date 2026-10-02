import sys
import numpy as np
import matplotlib.pyplot as plt

def carregar(path):
    valores = []
    with open(path, 'r') as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            try:
                valores.append(float(linha))
            except ValueError:
                continue
    return np.array(valores)

def calcular_erros(ref, comp):
    n = min(len(ref), len(comp))
    ref = ref[:n]
    comp = comp[:n]
    mask = ref != 0.0
    erros = np.zeros(n)
    erros[mask] = np.abs(ref[mask] - comp[mask]) / np.abs(ref[mask])
    erros[~mask] = np.abs(comp[~mask])
    return erros

def main():
    if len(sys.argv) != 5:
        print("Uso: python histograma_erro.py <arq_ref> <arq_comp> <tolerancia> <saida.png>")
        sys.exit(1)

    path_ref  = sys.argv[1]
    path_comp = sys.argv[2]
    tol       = float(sys.argv[3])
    saida     = sys.argv[4]

    ref  = carregar(path_ref)
    comp = carregar(path_comp)

    if len(ref) == 0 or len(comp) == 0:
        print("Arquivo vazio.")
        sys.exit(1)

    erros = calcular_erros(ref, comp)

    violacoes = int(np.sum(erros > tol))
    total     = len(erros)
    frac      = violacoes / total if total else 0.0

    print(f"Total: {total}")
    print(f"Violacoes (>{tol}): {violacoes} ({frac:.4f})")
    print(f"Erro medio:  {erros.mean():.6e}")
    print(f"Erro maximo: {erros.max():.6e}")
    print(f"Erro mediano: {np.median(erros):.6e}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(erros, bins=60, color='steelblue', edgecolor='black')
    ax.axvline(tol, color='red', linestyle='--', linewidth=1.5, label=f'tol={tol}')
    ax.set_xlabel('Erro relativo')
    ax.set_ylabel('Frequencia')
    ax.set_title(f'Histograma de erro relativo (linear)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    erros_pos = erros[erros > 0]
    if len(erros_pos) > 0:
        bins = np.logspace(np.log10(erros_pos.min()), np.log10(erros_pos.max()), 60)
        ax.hist(erros_pos, bins=bins, color='coral', edgecolor='black')
        ax.set_xscale('log')
    ax.axvline(tol, color='red', linestyle='--', linewidth=1.5, label=f'tol={tol}')
    ax.set_xlabel('Erro relativo (log)')
    ax.set_ylabel('Frequencia')
    ax.set_title('Histograma de erro relativo (log)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(saida, dpi=120)
    plt.show()

if __name__ == "__main__":
    main()