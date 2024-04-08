<h1 align="center">Projet de classification d'image par un modèle de deep-learning</h1>
<h1 align="center">Détecter le port du masque chirurgical</h1>

<h2 align="center">Dataset</h2>

<p align="center"><a href="https://www.kaggle.com/datasets/dhruvmak/face-mask-detection"><i>https://www.kaggle.com/datasets/dhruvmak/face-mask-detection</i></a></p>

<h2 align="center">Modèles utilisés</h2>

![Choix du modèle!](/assets/models.png "Description des modèles")

<h2 align="center">Application front-end</h2>
<table align="center">
  <tr>
    <td align="center" valign="top">
      Interface web<br/>
      <a href="https://face-mask-classification-front-alexsavina.vercel.app">https://face-mask-classification-front-alexsavina.vercel.app</a> <br/><br>
      <a href="https://face-mask-classification-front-alexsavina.vercel.app">
        <img alt="Face Mask detection" src="/assets/front_app_screen.png" width="300px" style="max-width:100%; border-radius: 10px;"/>
      </a>
      <br>
      Repo git<br/>
      <a href="https://github.com/alexsavz/FaceMaskClassification-front">github.com/alexsavz/FaceMaskClassification-front</a> <br/>
    </td>
  </tr>
</table>

## Problématique

Contexte :

- Data Scientist travaillant sur les systèmes de reconnaissance faciale
- Faciliter l’enregistrement et l’embarquement des passagers pour Paris aéroport
- Vérifier si le passager porte un masque chirurgical

Mission :

- Réaliser un premier modèle de détection
- Déployer une application

## Présentation des données

Le jeu de données est issu de kaggle :

- Deux dossiers d'images

  - **220** images de visages **sans** masques
  - **220** images de visages **avec** masques

- Données non structurées de résolutions différentes

## Structure du Notebook

1. Import des librairies, des données et des modèles
   - Import du ViT et des poids
   - Import du RegNet et des poids
2. Préparation des données
   - Création du Dataset
   - Transformation et augmentation des données
   - Division des données
3. Transfert Learning avec un Modèle Pré-entraîné
   - Construction du modèle à partir de la dernière couche
   - &Eacute;tape d'entraînement
   - &Eacute;tape de validation
   - Entraînement du modèle sur la dernière couche
   - Prédiction sur l'échantillon de test
   - Gestion des hyperparamètres
4. Sérialisation du meilleur modèle

## Compilation et entraînement

**Fonction de perte :**

- BCEWithLogitsLoss (Binary Cross Entropy)
- Fonction de perte adaptée à une classification binaire

**Algorithme d'optimisation :**

- Adam (Adaptive Moment Estimation)
- Méthode de descente de gradient
- (lr) γ de 1e-3, (betas) β1 de 0,9, β2 de 0,999
- Régularisation par pénalité (weights-decay) wd de 0,1

**Hyperparamètres**

- Méthode d'augmentation automatique : TrivialAugment
- Early stopping

Data augmentation avec TrivialAugment :

![TrivialAugment!](/assets/trivialaugment.png "TrivialAugment")

Performance du Vision Transformer et early stopping :

![Performance ViT!](/assets/vit_scores.png "Performance du modèle ViT")

## Technologies

**Manipulation des données:** numpy, scikit-learn
**Représentation graphique:** matplotlib
**Modelisation:** Pytorch
**Déploiement:** FastAPI, Docker, AWS Fargate

## Réutilisation

Commandes pour utiliser le projet:

```python
pip install -r requirements.txt
python main.py
```
