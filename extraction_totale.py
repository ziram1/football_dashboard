import json

import numpy as np
import pandas as pd
from statsbombpy import sb
from tqdm import tqdm

SEASON_LABEL = "15_16"
COMPETITION_ID = 11   # La Liga
SEASON_ID = 27        # 2015/2016

print("📥 Connexion à StatsBomb et récupération du calendrier de La Liga 15/16...")
df_matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
df_matches.to_csv(f"liga_matches_{SEASON_LABEL}.csv", index=False)
print(f"✅ Calendrier sauvegardé : 'liga_matches_{SEASON_LABEL}.csv' ({len(df_matches)} matchs).")

# Noms de colonnes de sortie pour les coordonnées (les autres colonnes *_location
# sont nommées automatiquement : shot_end_location -> shot_end_x / shot_end_y / shot_end_z)
COORD_RENAMES = {
    "location": "",             # -> x, y
    "pass_end_location": "end", # -> end_x, end_y
}


def split_coordinates(df):
    """Transforme chaque colonne 'location' / '*_location' ([x, y] ou [x, y, z])
    en colonnes numériques. Aucune information n'est perdue."""
    loc_cols = [c for c in df.columns if c == "location" or c.endswith("_location")]
    for col in loc_cols:
        prefix = COORD_RENAMES.get(col, col[: -len("_location")])
        names = ["x", "y", "z"] if prefix == "" else [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]

        arrays = df[col].apply(
            lambda v: list(v) if isinstance(v, (list, tuple, np.ndarray)) else []
        )
        max_len = arrays.str.len().max()
        if pd.isna(max_len) or max_len == 0:
            df = df.drop(columns=[col])
            continue
        for i in range(min(int(max_len), 3)):
            df[names[i]] = arrays.apply(lambda a, i=i: a[i] if len(a) > i else np.nan).astype(float)
        df = df.drop(columns=[col])
    return df


def make_parquet_safe(df):
    """Convertit les colonnes imbriquées (listes/dicts) ou de types mixtes en texte JSON,
    pour que l'écriture Parquet ne plante pas. Les données restent exploitables
    (json.loads) si vous en avez besoin plus tard."""
    for col in df.columns:
        if df[col].dtype != object:
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        types = set(type(v) for v in non_null.head(5000))
        if types == {str}:
            continue

        def to_text(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if isinstance(v, (list, dict, tuple)):
                return json.dumps(v, default=str)
            return str(v)

        df[col] = df[col].apply(to_text)
    return df


all_events_list = []
failed_matches = []

print(f"\n🔄 Téléchargement des événements ({len(df_matches)} matchs, TOUTES les colonnes)...")

for _, row in tqdm(df_matches.iterrows(), total=len(df_matches)):
    match_id = row["match_id"]
    week = row["match_week"]

    try:
        df_events = sb.events(match_id=match_id)   # toutes les colonnes, tous les événements
        df_events = split_coordinates(df_events)
        df_events["match_week"] = week
        df_events["match_id"] = match_id
        all_events_list.append(df_events)

    except Exception as e:
        failed_matches.append(match_id)
        print(f"\n❌ Erreur match ID {match_id} (Journée {week}): {e}")

print("\n💾 Compilation de la base de données globale...")
df_final_events = pd.concat(all_events_list, ignore_index=True)
df_final_events = make_parquet_safe(df_final_events)

# Parquet : compressé et lisible colonne par colonne (pip install pyarrow)
output_file = f"liga_events_{SEASON_LABEL}.parquet"
df_final_events.to_parquet(output_file, index=False, compression="snappy")

print("=" * 50)
print("🎉 BASE DE DONNÉES COMPLÈTE GÉNÉRÉE !")
print(f"Fichier : '{output_file}'")
print(f"Total : {len(df_final_events)} événements, {df_final_events.shape[1]} colonnes, "
      f"{df_final_events['match_id'].nunique()} matchs.")
if failed_matches:
    print(f"⚠️ Matchs en échec ({len(failed_matches)}) : {failed_matches}")
print("=" * 50)