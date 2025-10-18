# Seleção com Ações Afirmativas — V5.1 (Remanejamento entre AAs)

## Como subir no servidor (Docker + Compose)
```bash
sudo mkdir -p /opt/afirmativas-v5-1 && cd /opt/afirmativas-v5-1
# Envie estes arquivos para esta pasta:
# - afirmativas_app_v5_1.py
# - requirements.txt
# - Dockerfile
# - docker-compose.yml
mkdir -p data
docker compose build
docker compose up -d
# Acesse http://SEU-IP:8501
```

## Como rodar local (Python)
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run afirmativas_app_v5_1.py
```
