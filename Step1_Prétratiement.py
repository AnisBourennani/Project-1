import pandas as pd
import spacy

# Charger les données
df = pd.read_json('reviews.jsonl', lines=True)
print("Aperçu des données initiales :")
print(df.head(20))

# Sélection des champs pertinents
df = df[['rating', 'title', 'text']]
print("\nDonnées avec champs sélectionnés :")
print(df.head())

# Charger le modèle spaCy
nlp = spacy.load("en_core_web_sm")

# Fonction de prétraitement
def preprocess_text_spacy(text):
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        if (
            not token.is_stop  # Exclure les stop words
            and token.is_alpha  # Exclure les symboles et chiffres
            and len(token) > 2  # Exclure les mots courts
        ):
            tokens.append(token.lemma_)  # Lemmatisation
    return tokens

# Appliquer la fonction de prétraitement
df['processed_tokens'] = df['text'].apply(preprocess_text_spacy)
print("\nDonnées après prétraitement :")
print(df.head())

# Sauvegarder les données prétraitées si nécessaire
df.to_csv('processed_reviews.csv', index=False)
print("\nLes données prétraitées ont été sauvegardées dans 'processed_reviews.csv'.")

#Étape 2 : Génération des embeddings

import pandas as pd

# Charger les données prétraitées
df = pd.read_csv('processed_reviews.csv')
documents = df['processed_tokens'].apply(eval)  # Convertir les chaînes de tokens en listes Python
documents = [" ".join(tokens) for tokens in documents]  # Convertir les listes en chaînes
print(f"Nombre de documents : {len(documents)}")


