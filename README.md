<h1 align="center">Projet de classification d'image par un modèle de deep-learning</h1>
<h1 align="center">Détecter le port du masque chirurgical</h1>

<h2 align="center">Dataset</h2>

<p align="center"><a href="https://www.kaggle.com/datasets/dhruvmak/face-mask-detection"><i>https://www.kaggle.com/datasets/dhruvmak/face-mask-detection</i></a></p>

<h2 align="center">Problématique</h2>

<p align="center">
<b>Contexte :</b>
<br>- Data Scientist travaillant sur les systèmes de reconnaissance faciale
<br>- Faciliter l’enregistrement et l’embarquement des passagers pour Paris aéroport
<br>- Vérifier si le passager porte un masque chirurgical
</p>

<p align="center">
<b>Mission :</b>
<br>- Réaliser un premier modèle de détection
<br>- Déployer une application
</p>

<p align="center"><i>Modèle de machine learning : lightGBM</i></p>

<h2 align="center">Application front-end</h2>
<table align="center">
  <tr>
    <td align="center" valign="top">
      Interface web<br/>
      <a href="https://face-mask-classification-front-alexsavina.vercel.app">https://face-mask-classification-front-alexsavina.vercel.app</a> <br/><br>
      Repo github<br/>
      <a href="https://github.com/alexsavz/FaceMaskClassification-front">Application front-end</a> <br/><br>
      <a href="https://face-mask-classification-front-alexsavina.vercel.app">
        <img alt="Face Mask detection" src="/assets/front_app_screen.png" width="300px" style="max-width:100%; border-radius: 10px;"/>
      </a>
      <br>
      Repo github<br/>
      <a href="https://github.com/alexsavz/FaceMaskClassification-front">Application front-end</a> <br/>
    </td>
  </tr>
</table>

## Tech stack

**Analyse des données:** pandas, numpy, scipy
**Représentation graphique:** matplotlib, seaborn, plotly
**Modelisation:** scikit-learn, lightgbm, shap
**Mise en production:** Docker, Github Actions, mlflow, Streamlit

## Notebook

1. Exploration du jeu de données
   - Description des données
   - Contrôle de la qualité des données
   - Analyse exploratoire
   - Présélection de variables
2. Création et optimisation du modèle
   - Préprocessing des données d'évaluation
   - Pipeline de transformation
   - Optimisation du modèle
3. Evaluation et explicativité des modèles
   - Courbe de lift
   - Score de spiegelhalter
   - Courbe de calibration
   - Importance des variables
4. Sérialisation du meilleur modèle
   - Sauvegarde locale
   - Log du modèle avec MLflow
5. Prédiction sur l'échantillon de test

## Réutilisation

Commandes pour utiliser le projet:

```python
pip install -r requirements.txt
python main.py
```
