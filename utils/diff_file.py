import sys

def comparar_arquivos(path_ref, path_comp, tolerancia):
    total_violacoes = 0
    violacoes_detalhes = []
    total = 0
    try:
        with open(path_ref, 'r') as f_ref, open(path_comp, 'r') as f_comp:
            for idx, (linha_ref, linha_comp) in enumerate(zip(f_ref, f_comp)):
                try:
                    val_ref = float(linha_ref.strip())
                    val_comp = float(linha_comp.strip())
                except ValueError:
                    print(f"Erro ao converter linha {idx} para float. Pulando.")
                    continue
                
                total += 1
                if val_ref == 0.0:
                    erro_relativo = abs(val_ref - val_comp) if val_comp != 0.0 else 0.0
                else:
                    erro_relativo = abs(val_ref - val_comp) / abs(val_ref)
                
                if erro_relativo > tolerancia:
                    total_violacoes += 1
                    if len(violacoes_detalhes) < 20:
                        violacoes_detalhes.append((idx, val_ref, val_comp, erro_relativo))
                        
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado - {e.filename}")
        sys.exit(1)

    print("-" * 50)
    print(f"Total de linhas que violaram a tolerancia (> {tolerancia}): {total_violacoes} / {total} ({(total_violacoes / total):.2f})")
    print("-" * 50)
    
    if total_violacoes > 0:
        print("Primeiras 20 violacoes encontradas:")
        print(f"{'Indice':<8} | {'Ref (Arq 1)':<15} | {'Comp (Arq 2)':<15} | {'Erro Relat.':<12}")
        print("-" * 55)
        for idx, ref, comp, err in violacoes_detalhes:
            print(f"{idx:<8} | {ref:<15.6f} | {comp:<15.6f} | {err:<12.6f}")
    else:
        print("Todos os numeros estao dentro da tolerancia!")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso incorreto do script.")
        print("Como usar: python compara_arquivos.py <path_arquivo1> <path_arquivo2> <tolerancia>")
        sys.exit(1)
        
    path_1 = sys.argv[1]
    path_2 = sys.argv[2]
    
    try:
        tol = float(sys.argv[3])
    except ValueError:
        print("Erro: A tolerancia deve ser um numero valido.")
        sys.exit(1)
        
    comparar_arquivos(path_1, path_2, tol)
