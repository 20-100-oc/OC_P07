# Projet: Détectez les Bad Buzz grâce au Deep Learning

## Contexte
L'entreprise Air Paradis cherche à améliorer sa réputation sur les réseaux sociaux.
Le premier pas est de développer un prototype permettant de prédire le sentiment associé à un texte, et plus précisément, un tweet.
Cependant, la société ne possède pas de données destinées à cet usage. 
Il faut donc se tourner vers un apport exterieur: un jeu de données Open Source recueillant 1,6 millions de tweets est sélectionné.
Le manager indique qu'il souhaite des approches avec des modèles variés, ainsi qu'un article de blog pour exposer les résultats.
Le projet reste un prototype, les tests doivent donc être effectués en limitant les coûts de mise en production.

## Objectifs
- Pipeline de prétraitement des données textuelles
- Modélisation à complexité croissante: 
  - régression linéaire
  - réseau de neuronnes récurrent avec plongement de mots
  - modèle type BERT
- Déploiement continue sur le cloud du modèle choisi (BERT)
- Création d'une application web pour interagir avec le modèle déployé
- Rédaction d'un article de blog comparant les performances des diverses approches

## Livrables
- Notebooks de modélisation
- Scripts de déploiement cloud du modèle avec Azure web app
- Application web d'interaction (Streamlit)
- Article de blog
- Présentation PowerPoint

## Outils
- Python
- Git / Github
- Jupyter notebook / Python IDE
- PowerPoint
- Streamlit
- Azure web app
- Google Storage

### Python : libraires additionnelles
- pandas
- nltk
- contractions
- numpy
- matplotlib
- seaborn
- sklearn
- tensorflow
- pytorch
- transformers
- google-auth
- streamlit
- fastapi