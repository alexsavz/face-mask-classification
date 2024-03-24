# Utiliser une image de base Python 3.10
FROM python:3.10.2-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier des dépendances et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste des fichiers de votre application dans le conteneur
COPY . .

# Exposer le port sur lequel l'application FastAPI va s'exécuter
EXPOSE 80

# Définir la commande pour exécuter l'application
# Uvicorn sert d'ASGI server, ici on spécifie l'host sur lequel uvicorn doit écouter et le port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]