"""Script para publicar o ensemble no Hugging Face Hub"""

import json
import os
from huggingface_hub import HfApi, login


def push_ensemble_to_hub(
    weights: dict,
    repo_name: str,
    token: str = None,
) -> None:
    """
    Publica os pesos e código do ensemble no Hugging Face Hub.
    
    Args:
        weights: pesos do ensemble
        repo_name: nome do repositório (formato: usuario/repo)
        token: token HF. Se None, usa variável de ambiente HF_TOKEN.
    """
    if token is None:
        token = os.environ.get("HF_TOKEN")
        if token is None:
            raise ValueError(
                "Token não encontrado. Defina HF_TOKEN ou passe como argumento."
            )
    
    login(token=token)
    api = HfApi()
    
    api.create_repo(repo_id=repo_name, repo_type="model", exist_ok=True)
    print(f"Repositório criado/acessado: {repo_name}")
    
    weights_file = "ensemble_weights.json"
    with open(weights_file, "w") as f:
        json.dump(weights, f, indent=2)
    
    api.upload_file(
        path_or_fileobj=weights_file,
        path_in_repo="ensemble_weights.json",
        repo_id=repo_name,
    )
    print(f"  weights uploaded: {weights_file}")
    
    api.upload_file(
        path_or_fileobj="src/ensemble.py",
        path_in_repo="ensemble.py",
        repo_id=repo_name,
    )
    print(f"  code uploaded: src/ensemble.py")
    
    print(f"\n✅ Publicado em: https://huggingface.co/{repo_name}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python push_to_hub.py <repo_name>")
        sys.exit(1)
    
    repo_name = sys.argv[1]
    
    default_weights = {
        "roberta_twitter": 0.40,
        "bertweet": 0.35,
        "distilbert": 0.25,
    }
    
    push_ensemble_to_hub(default_weights, repo_name)