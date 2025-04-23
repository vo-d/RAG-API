FOR PRODUCTION:
First, run the docker compose
docker compose up --build

Second, download mxbai-embed-large in the ollama container
ollama pull mxbai-embed-large

Third, download Deepseek in the ollama container
ollama run deepseek-r1

FOR DEVELOPMENT ONLY:
Create python virtual env
python -m venv venv

Enter that python vitual environment
venv/Scripts/activate.ps1