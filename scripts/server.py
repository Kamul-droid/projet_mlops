import data_loader
import pandas as pd
import quality_checks
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from preprocessing import preprocess_data

app = FastAPI()


@app.get("/")
def root() -> dict:
    """
    Point de terminaison racine de l'API.

    Returns:
        dict: Message de bienvenue de l'API.
    """
    return {"message": "Heart Attack Data Pipeline API"}


@app.post("/ingest")
def ingest_data() -> dict:
    """
    Déclenche l'ingestion des données en appelant la fonction principale du module data_loader.

    Returns:
        dict: Message confirmant la fin de l'ingestion des données.
    """
    data_loader.main()
    return {"message": "Ingestion des données terminée"}


@app.get("/quality-checks")
def run_quality_checks() -> JSONResponse:
    """
    Exécute les contrôles de qualité en utilisant le module quality_checks.

    Returns:
        JSONResponse: Résultats des contrôles de qualité ou message d'erreur en cas d'échec.
    """
    try:
        result = quality_checks.main()
        return JSONResponse(content={"message": "Contrôles de qualité terminés", "details": result})
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"message": "Erreur lors des contrôles de qualité", "error": str(e)}
        )


@app.get("/data")
def get_data() -> list:
    """
    Récupère un échantillon des données traitées depuis le fichier CSV.

    Returns:
        list: Liste des enregistrements sous forme de dictionnaires.
    """
    try:
        data = pd.read_csv("data/processed_heart.csv").head(10)
        return data.to_dict(orient="records")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Le fichier CSV est introuvable.")
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=500, detail="Le fichier CSV est vide.")
    except pd.errors.ParserError:
        raise HTTPException(status_code=500, detail="Erreur de parsing dans le fichier CSV.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {e}")


@app.post("/preprocess")
def preprocess() -> dict:
    """
    Lance le prétraitement des données en appelant la fonction preprocess_data du module preprocessing.

    Returns:
        dict: Message confirmant la fin du prétraitement.
    """
    try:
        df = pd.read_csv("data/processed_heart.csv")

        X_train_clean, X_test_clean, y_train, y_test = preprocess_data(df)

        X_train_clean.to_csv("data/X_train_clean.csv", index=False)
        X_test_clean.to_csv("data/X_test_clean.csv", index=False)
        pd.DataFrame(y_train).to_csv("data/y_train.csv", index=False)
        pd.DataFrame(y_test).to_csv("data/y_test.csv", index=False)

        return {"message": "Prétraitement terminé avec succès."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du prétraitement: {e}")
