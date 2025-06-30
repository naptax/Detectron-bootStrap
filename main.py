import subprocess
import sys
import platform


def check_cuda():
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode()
        for line in output.splitlines():
            if "release" in line:
                cuda_version = line.strip().split("release")[-1].split(",")[0].strip()
                print(f"[CUDA] Version détectée : {cuda_version}")
                return cuda_version
        print("[CUDA] Version non détectée dans la sortie nvcc.")
        return None
    except FileNotFoundError:
        print("[CUDA] nvcc non trouvé. CUDA n'est pas installé ou n'est pas dans le PATH.")
        return None
    except Exception as e:
        print(f"[CUDA] Erreur lors de la détection de CUDA : {e}")
        return None

def check_pytorch():
    try:
        import torch
        print(f"[PyTorch] Version détectée : {torch.__version__}")
        if torch.cuda.is_available():
            print("[PyTorch] CUDA est disponible pour PyTorch.")
            print(f"[PyTorch] Version CUDA utilisée par PyTorch : {torch.version.cuda}")
            return torch.__version__, torch.version.cuda
        else:
            print("[PyTorch] CUDA n'est PAS disponible pour PyTorch.")
            return torch.__version__, None
    except ImportError:
        print("[PyTorch] PyTorch n'est pas installé.")
        return None, None
    except Exception as e:
        print(f"[PyTorch] Erreur lors de la détection de PyTorch : {e}")
        return None, None

def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            nb_gpus = torch.cuda.device_count()
            print(f"[GPU] Nombre de GPU détectés : {nb_gpus}")
            for i in range(nb_gpus):
                print(f"  [GPU {i}] Nom : {torch.cuda.get_device_name(i)}")
                print(f"  [GPU {i}] Mémoire totale : {torch.cuda.get_device_properties(i).total_memory // (1024 ** 2)} Mo")
                print(f"  [GPU {i}] Capacité de calcul : {torch.cuda.get_device_capability(i)}")
        else:
            print("[GPU] Aucun GPU compatible CUDA détecté.")
    except ImportError:
        print("[GPU] PyTorch requis pour détecter les GPU CUDA.")
    except Exception as e:
        print(f"[GPU] Erreur lors de la détection du GPU : {e}")

def check_rtx5090_compat(pytorch_cuda_version, gpu_name):
    # RTX 5090 (Ada Lovelace) nécessite CUDA >= 12.0 et PyTorch >= 2.1 pour support optimal
    print("\n[Compatibilité RTX 5090]")
    if gpu_name and "5090" in gpu_name:
        print("GPU RTX 5090 détecté.")
        cuda_ok = False
        torch_ok = False
        if pytorch_cuda_version:
            # Ex: '12.1'
            major = int(pytorch_cuda_version.split(".")[0])
            if major >= 12:
                cuda_ok = True
        if cuda_ok:
            print("  - Version CUDA OK (>= 12.0)")
        else:
            print("  - Version CUDA INSUFFISANTE (besoin >= 12.0)")
        # Vérification version PyTorch
        try:
            import torch
            torch_major = int(torch.__version__.split(".")[0])
            torch_minor = int(torch.__version__.split(".")[1])
            if torch_major > 2 or (torch_major == 2 and torch_minor >= 1):
                torch_ok = True
            if torch_ok:
                print("  - Version PyTorch OK (>= 2.1)")
            else:
                print("  - Version PyTorch INSUFFISANTE (besoin >= 2.1)")
        except:
            print("  - PyTorch non installé ou version non détectée.")
        if cuda_ok and torch_ok:
            print("[CHAÎNE COMPATIBLE : RTX 5090, CUDA >= 12.0, PyTorch >= 2.1]")
        else:
            print("[CHAÎNE NON COMPATIBLE : Mettez à jour CUDA et/ou PyTorch]")
    else:
        print("GPU RTX 5090 non détecté, vérification spécifique non effectuée.")

def main():
    print("Système :", platform.platform())
    print("Python :", sys.version)
    print("\n--- Vérification GPU ---")
    gpu_name = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU principal : {gpu_name}")
        else:
            print("Aucun GPU CUDA détecté.")
    except:
        print("PyTorch requis pour détecter le GPU.")
    check_gpu()
    print("\n--- Vérification CUDA ---")
    cuda_version = check_cuda()
    print("\n--- Vérification PyTorch ---")
    torch_version, torch_cuda_version = check_pytorch()
    check_rtx5090_compat(torch_cuda_version, gpu_name)

    # --- Vérification Detectron2 ---
    print("\n--- Vérification Detectron2 ---")
    try:
        import detectron2
        version = getattr(detectron2, '__version__', None)
        if version is not None:
            print(f"[Detectron2] Version détectée : {version}")
        else:
            print(f"[Detectron2] Module trouvé, version non détectée. Chemin : {detectron2.__file__}")
    except ImportError:
        print("[Detectron2] Detectron2 n'est pas installé.")
    except Exception as e:
        print(f"[Detectron2] Erreur lors de la détection de Detectron2 : {e}")

    print("\n--- Conseils ---")
    if not cuda_version:
        print("- Installez CUDA 12.0 ou supérieur pour la RTX 5090 : https://developer.nvidia.com/cuda-downloads")
    if not torch_version:
        print("- Installez PyTorch 2.1 ou supérieur avec support CUDA 12 : https://pytorch.org/get-started/locally/")
    print("\nFin du diagnostic.")

if __name__ == "__main__":
    main()
