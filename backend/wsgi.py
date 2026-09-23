"""Point d'entrée WSGI pour la production : gunicorn -w 2 -b 0.0.0.0:5005 wsgi:app"""

from app import create_app

app = create_app()
