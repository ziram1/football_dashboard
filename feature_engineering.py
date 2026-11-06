import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SEASON = "15_16"
EVENTS_FILE = f"liga_events_{SEASON}.parquet"

# Repère StatsBomb (120 x 80) : l'équipe attaque vers x = 120, dernier tiers = x >= 80
FINAL_THIRD_MIN_X = 80
# 1 = on compte toutes les possessions ; 2 ou plus = on exclut les possessions très courtes
MIN_EVENTS_PER_POSSESSION = 1

# PPDA (définition StatsBomb) : passes RÉUSSIES de l'adversaire parties de x < 72 (repère de l'adversaire)
# divisées par les actions défensives de l'équipe à x >= 48 (tacles, interceptions, Dribbled Past, fautes)
PPDA_ZONE_PASS_MAX_X = 72
PPDA_ZONE_DEF_MIN_X = 48
PPDA_EXCLUDE_GOAL_KICKS = True   # exclut les coups de pied de but des passes adverses

# Analyse des passes (StatsBomb mesure les distances en yards)
PROGRESSIVE_REDUCTION = 0.25   # passe progressive = réduit d'au moins 25 % la distance au but
SHORT_PASS_MAX_YARDS = 15      # passe courte  : longueur < 15 yards (~14 m)
LONG_PASS_MIN_YARDS = 32       # passe longue  : longueur >= 32 yards (~29 m)

print("=" * 60)
print("🚀 PIPELINE DE FEATURE ENGINEERING : GÉNÉRATION DES MÉTRIQUES CLÉS")
print("=" * 60)

# 1. CHARGEMENT DE LA SOURCE BRUTE (lecture sélective des colonnes : rapide avec Parquet)
# Pour une future feature, ajoutez simplement ici les colonnes dont elle a besoin.
COLUMNS_NEEDED = [
    "id", "match_id", "match_week", "period", "minute", "second", "duration",
    "type", "team", "player", "possession", "possession_team",
    "x", "y", "shot_outcome", "shot_statsbomb_xg",
    "pass_outcome", "dribble_outcome",
    "pass_recipient", "pass_length", "under_pressure", "end_x", "end_y",
    "pass_type", "duel_type",
]

# Colonnes optionnelles (statistiques défensives) : lues si elles existent dans le fichier.
# Si l'une manque, l'indicateur correspondant sera à 0 et un avertissement s'affichera.
OPTIONAL_COLUMNS = [
    "duel_outcome", "interception_outcome", "counterpress", "ball_recovery_recovery_failure",
    "clearance_aerial_won", "pass_aerial_won", "shot_aerial_won", "miscontrol_aerial_won",
    "pass_goal_assist", "pass_shot_assist", "pass_assisted_shot_id",
    "shot_type", "carry_outcome", "carry_end_x", "carry_end_y",
    "substitution_replacement", "position",
]

print(f"\n📖 Lecture de '{EVENTS_FILE}'...")
try:
    available_columns = set(pq.ParquetFile(EVENTS_FILE).schema.names)
except FileNotFoundError:
    print(f"❌ Erreur : '{EVENTS_FILE}' est introuvable. Lancez d'abord 'extraction_totale.py'.")
    sys.exit(1)

missing_required = [c for c in COLUMNS_NEEDED if c not in available_columns]
if missing_required:
    print(f"❌ Colonnes indispensables absentes du fichier d'événements : {missing_required}")
    sys.exit(1)

optional_present = [c for c in OPTIONAL_COLUMNS if c in available_columns]
optional_absent = [c for c in OPTIONAL_COLUMNS if c not in available_columns]
if optional_absent:
    print(f"⚠️ Colonnes optionnelles absentes (indicateurs concernés calculés sans elles) : {optional_absent}")

df_events = pd.read_parquet(EVENTS_FILE, columns=COLUMNS_NEEDED + optional_present)

KEYS = ["match_id", "match_week", "team"]

# =====================================================================
# FEATURE 1 : DURÉE MOYENNE DE POSSESSION NETTE (par équipe et par match)
# =====================================================================
print("\n⏱️ [Feature 1/7] Calcul des durées de possession nettes...")

# Temps absolu en secondes (la minute StatsBomb continue en 2e mi-temps : 45, 46, ...)
df_events["temps_secondes"] = (df_events["minute"] * 60) + df_events["second"]

df_pos = df_events.dropna(subset=["possession_team", "possession", "match_id", "period"]).copy()

# Début et fin brute de chaque séquence de possession
poss_times = (
    df_pos.groupby(["match_id", "match_week", "possession_team", "possession"])["temps_secondes"]
    .agg(["min", "max"])
    .reset_index()
)
poss_times["duree_nette"] = poss_times["max"] - poss_times["min"]

# Filtrage des anomalies (0 s et arrêts de jeu de plus de 5 minutes)
poss_times = poss_times[(poss_times["duree_nette"] > 0) & (poss_times["duree_nette"] < 300)]

# Agrégation par match et par équipe ; on garde le NOMBRE de possessions pour
# pouvoir calculer ensuite une vraie moyenne pondérée sur plusieurs matchs.
df_possession = (
    poss_times.groupby(["match_id", "match_week", "possession_team"])["duree_nette"]
    .agg(duree_possession_moyenne="mean", nb_possessions="count")
    .reset_index()
    .rename(columns={"possession_team": "team"})
)

# =====================================================================
# FEATURE 2 : TIRS, BUTS ET xG (par équipe et par match)
# =====================================================================
print("\n🎯 [Feature 2/7] Calcul des tirs, buts et xG...")

df_shots = df_events[(df_events["type"] == "Shot") & (df_events["period"] < 5)].copy()  # hors tirs au but
df_shots["shot_statsbomb_xg"] = df_shots["shot_statsbomb_xg"].fillna(0)
df_shots["is_goal"] = df_shots["shot_outcome"] == "Goal"
df_shots["is_blocked"] = df_shots["shot_outcome"] == "Blocked"

df_team_shots = (
    df_shots.groupby(KEYS)
    .agg(
        tirs=("id", "count"),
        buts=("is_goal", "sum"),
        xg=("shot_statsbomb_xg", "sum"),
        tirs_bloques=("is_blocked", "sum"),
    )
    .reset_index()
)

# =====================================================================
# FEATURE 3 : POURCENTAGE DE POSSESSION (par équipe et par match)
# Méthode : part du temps de jeu où l'équipe a le ballon, mesurée en additionnant
# la durée (champ 'duration' StatsBomb) des événements de chaque possession_team.
# =====================================================================
print("\n📊 [Feature 3/7] Calcul du pourcentage de possession...")

df_dur = df_events[df_events["period"] < 5].dropna(subset=["possession_team", "duration"])

df_poss_time = (
    df_dur.groupby(["match_id", "match_week", "possession_team"])["duration"]
    .sum()
    .reset_index()
    .rename(columns={"possession_team": "team", "duration": "temps_possession"})
)
total_match_time = df_poss_time.groupby("match_id")["temps_possession"].transform("sum")
df_poss_time["possession_pct"] = (df_poss_time["temps_possession"] / total_match_time) * 100

# =====================================================================
# FEATURE 4 : ISSUE DES POSSESSIONS (par équipe et par match)
#  - possession "avec tir"       : contient au moins un tir de l'équipe en possession
#  - possession "dernier tiers"  : contient au moins un événement de l'équipe en possession
#                                  dans le dernier tiers (x >= 80), y compris si elle y démarre
# Le dénominateur (nb_possessions_total) compte toutes les possessions de l'équipe.
# =====================================================================
print("\n🔚 [Feature 4/7] Calcul de l'issue des possessions (tir / dernier tiers)...")

df_poss_ev = df_events[df_events["period"] < 5].dropna(subset=["possession", "possession_team"]).copy()
by_possessing_team = df_poss_ev["team"] == df_poss_ev["possession_team"]
df_poss_ev["is_shot"] = by_possessing_team & (df_poss_ev["type"] == "Shot")
df_poss_ev["in_final_third"] = by_possessing_team & (df_poss_ev["x"] >= FINAL_THIRD_MIN_X)

poss_flags = (
    df_poss_ev.groupby(["match_id", "match_week", "possession_team", "possession"])
    .agg(
        nb_events=("is_shot", "size"),
        has_shot=("is_shot", "any"),
        reaches_final_third=("in_final_third", "any"),
    )
    .reset_index()
)
poss_flags = poss_flags[poss_flags["nb_events"] >= MIN_EVENTS_PER_POSSESSION]

df_poss_outcomes = (
    poss_flags.groupby(["match_id", "match_week", "possession_team"])
    .agg(
        nb_possessions_total=("has_shot", "size"),
        nb_poss_with_shot=("has_shot", "sum"),
        nb_poss_final_third=("reaches_final_third", "sum"),
    )
    .reset_index()
    .rename(columns={"possession_team": "team"})
)

# =====================================================================
# FEATURE 5 : PERTES DE BALLE (un événement = une perte, avec sa position)
# Pertes comptées :
#  - passe ratée (incomplète, sortie, hors-jeu), hors dégagement pour blessure
#  - dribble raté
#  - mauvais contrôle (Miscontrol) et dépossession (Dispossessed)
# Les tirs ne sont pas des pertes. Coordonnées dans le sens d'attaque de l'équipe.
# =====================================================================
print("\n🔻 [Feature 5/7] Extraction des pertes de balle...")

df_reg = df_events[df_events["period"] < 5]

is_pass_loss = (
    (df_reg["type"] == "Pass")
    & df_reg["pass_outcome"].notna()
    & (df_reg["pass_outcome"] != "Injury Clearance")
)
is_dribble_loss = (df_reg["type"] == "Dribble") & (df_reg["dribble_outcome"] == "Incomplete")
is_miscontrol = df_reg["type"] == "Miscontrol"
is_dispossessed = df_reg["type"] == "Dispossessed"

df_losses = df_reg[is_pass_loss | is_dribble_loss | is_miscontrol | is_dispossessed].copy()
df_losses["loss_type"] = np.select(
    [
        df_losses["type"] == "Pass",
        df_losses["type"] == "Dribble",
        df_losses["type"] == "Miscontrol",
        df_losses["type"] == "Dispossessed",
    ],
    ["Passe ratée", "Dribble raté", "Mauvais contrôle", "Dépossession"],
    default="Autre",
)
df_losses = df_losses.dropna(subset=["x", "y"])

# =====================================================================
# FEATURE 6 : ANALYSE DES PASSES
#  - précision, passes progressives, passes vers l'avant, passes longues,
#    longueur moyenne, précision sous pression (compteurs par équipe et par match)
#  - table de toutes les passes (pour le réseau, les cartes de chaleur et l'histogramme)
# Direction : Avant (angle < 45° vers le but adverse), Arrière (> 135°), Latérale (entre les deux)
# Passe progressive : passe RÉUSSIE réduisant d'au moins 25 % la distance au centre du but adverse
# =====================================================================
print("\n🔄 [Feature 6/7] Analyse des passes...")

df_pass = df_events[(df_events["type"] == "Pass") & (df_events["period"] < 5)].copy()
df_pass = df_pass.dropna(subset=["x", "y", "end_x", "end_y"])

df_pass["is_complete"] = df_pass["pass_outcome"].isna()
# 'under_pressure' peut être un booléen ou du texte ("True") selon l'extraction
df_pass["under_pressure_flag"] = df_pass["under_pressure"].map(lambda v: str(v).lower() == "true")

def statsbomb_bool(series):
    """Convertit les booléens StatsBomb aplatis ou sérialisés en booléens fiables."""
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin(["true", "1", "1.0"])


if "pass_goal_assist" in df_pass.columns:
    df_pass["is_assist"] = statsbomb_bool(df_pass["pass_goal_assist"])
else:
    df_pass["is_assist"] = False

# Passe clé = passe qui amène directement un tir (tir non marqué OU but). StatsBomb distingue
# 'shot_assist' (tir) et 'goal_assist' (but) : on prend l'union pour ne pas oublier les passes décisives.
key_pass = pd.Series(False, index=df_pass.index)
if "pass_shot_assist" in df_pass.columns:
    key_pass = key_pass | statsbomb_bool(df_pass["pass_shot_assist"])
key_pass = key_pass | df_pass["is_assist"]
if "pass_assisted_shot_id" in df_pass.columns:
    key_pass = key_pass | df_pass["pass_assisted_shot_id"].notna()
df_pass["is_key_pass"] = key_pass

fallback_len = np.hypot(df_pass["end_x"] - df_pass["x"], df_pass["end_y"] - df_pass["y"])
df_pass["pass_length"] = df_pass["pass_length"].fillna(fallback_len)

# Passes progressives (but adverse au centre de la ligne de but : x = 120, y = 40)
dist_start = np.hypot(120 - df_pass["x"], 40 - df_pass["y"])
dist_end = np.hypot(120 - df_pass["end_x"], 40 - df_pass["end_y"])
df_pass["is_progressive"] = df_pass["is_complete"] & (dist_end <= (1 - PROGRESSIVE_REDUCTION) * dist_start)

# Direction de la passe
angle_abs = np.abs(np.arctan2(df_pass["end_y"] - df_pass["y"], df_pass["end_x"] - df_pass["x"]))
df_pass["direction"] = np.select(
    [angle_abs <= np.pi / 4, angle_abs >= 3 * np.pi / 4],
    ["Avant", "Arrière"],
    default="Latérale",
)

# Longueur de la passe
df_pass["length_class"] = np.select(
    [df_pass["pass_length"] < SHORT_PASS_MAX_YARDS, df_pass["pass_length"] >= LONG_PASS_MIN_YARDS],
    ["Courte", "Longue"],
    default="Moyenne",
)

df_pass["is_forward"] = df_pass["direction"] == "Avant"
df_pass["is_long"] = df_pass["length_class"] == "Longue"
df_pass["pressure_ok"] = df_pass["under_pressure_flag"] & df_pass["is_complete"]

df_team_passes = (
    df_pass.groupby(KEYS)
    .agg(
        nb_passes=("id", "count"),
        nb_passes_ok=("is_complete", "sum"),
        nb_passes_prog=("is_progressive", "sum"),
        nb_passes_forward=("is_forward", "sum"),
        nb_passes_long=("is_long", "sum"),
        sum_pass_length=("pass_length", "sum"),
        nb_passes_pressure=("under_pressure_flag", "sum"),
        nb_passes_pressure_ok=("pressure_ok", "sum"),
    )
    .reset_index()
)

# =====================================================================
# FEATURE 7 : STATISTIQUES DÉFENSIVES (par équipe et par match)
#  - Solidité : tirs / xG / buts concédés (= tirs de l'adversaire dans le même match), tirs contrés
#  - PPDA (définition StatsBomb)
#  - Pressing : pressions, pressions dans le tiers adverse, contre-pressings
#  - Récupérations (récupérations de balle réussies + interceptions réussies), hauteur, interceptions
#  - Duels : tacles gagnés, duels aériens gagnés / perdus ; dégagements
#  - Table de positions des pressions et des récupérations (cartes de chaleur)
# =====================================================================
print("\n🛡️ [Feature 7/7] Calcul des statistiques défensives...")

df_d = df_events[df_events["period"] < 5].copy()
t = df_d["type"]
x = df_d["x"]
y = df_d["y"]


def flag(col):
    """Booléen robuste : la colonne peut contenir True/False, 'True' (texte), NaN, ou être absente."""
    if col not in df_d.columns:
        return pd.Series(False, index=df_d.index)
    return df_d[col].map(lambda v: str(v).lower() == "true")


def count_by(mask, name):
    return df_d[mask].groupby(["match_id", "team"]).size().rename(name)


# Paires (équipe, adversaire) pour chaque match
match_teams = df_possession[["match_id", "team"]].drop_duplicates()
pairs = match_teams.merge(match_teams.rename(columns={"team": "opp"}), on="match_id")
pairs = pairs[pairs["team"] != pairs["opp"]]

# --- Solidité : ce que l'adversaire a obtenu dans le même match ---
opp_shots = df_team_shots.rename(
    columns={"team": "opp", "tirs": "tirs_conc", "buts": "buts_enc", "xg": "xg_conc", "tirs_bloques": "tirs_contres"}
)[["match_id", "opp", "tirs_conc", "buts_enc", "xg_conc", "tirs_contres"]]
df_conceded = pairs.merge(opp_shots, on=["match_id", "opp"], how="left")[
    ["match_id", "team", "tirs_conc", "buts_enc", "xg_conc", "tirs_contres"]
]

# --- Interceptions (événement 'Interception' réussi + passes de type interception) ---
if "interception_outcome" in df_d.columns:
    interception_failed = df_d["interception_outcome"].astype(str).str.startswith("Lost")
else:
    interception_failed = pd.Series(False, index=df_d.index)
is_interception_event = t == "Interception"
is_pass_interception = (t == "Pass") & (df_d["pass_type"] == "Interception")
interception_ok = (is_interception_event & ~interception_failed) | is_pass_interception

# --- PPDA ---
ppda_pass_mask = (t == "Pass") & df_d["pass_outcome"].isna() & (x < PPDA_ZONE_PASS_MAX_X)
if PPDA_EXCLUDE_GOAL_KICKS:
    ppda_pass_mask = ppda_pass_mask & (df_d["pass_type"] != "Goal Kick")

is_tackle = (t == "Duel") & (df_d["duel_type"] == "Tackle")
ppda_def_mask = (
    (is_tackle | is_interception_event | is_pass_interception | (t == "Dribbled Past") | (t == "Foul Committed"))
    & (x >= PPDA_ZONE_DEF_MIN_X)
)

passes_by = count_by(ppda_pass_mask, "passes_zone").reset_index().rename(columns={"team": "opp"})
df_faced = pairs.merge(passes_by, on=["match_id", "opp"], how="left")[["match_id", "team", "passes_zone"]]
df_faced = df_faced.rename(columns={"passes_zone": "ppda_passes"})

# --- Pressing ---
is_pressure = t == "Pressure"
if "counterpress" not in df_d.columns:
    print("⚠️ Colonne 'counterpress' absente : contre-pressings = 0.")

# --- Récupérations : récupérations de balle réussies + interceptions réussies ---
ball_recovery_ok = (t == "Ball Recovery") & ~flag("ball_recovery_recovery_failure")
recovery_mask = (ball_recovery_ok | interception_ok) & x.notna() & y.notna()

# --- Duels ---
if "duel_outcome" in df_d.columns:
    tackle_won = is_tackle & df_d["duel_outcome"].isin(["Won", "Success In Play", "Success Out"])
else:
    tackle_won = is_tackle
    print("⚠️ Colonne 'duel_outcome' absente : tous les tacles sont comptés comme gagnés.")

aerial_lost = (t == "Duel") & (df_d["duel_type"] == "Aerial Lost")
aerial_flag_cols = [
    c for c in ["clearance_aerial_won", "pass_aerial_won", "shot_aerial_won", "miscontrol_aerial_won"]
    if c in df_d.columns
]
aerial_won = pd.Series(False, index=df_d.index)
for c in aerial_flag_cols:
    aerial_won = aerial_won | flag(c)
if not aerial_flag_cols:
    print("⚠️ Aucune colonne '*_aerial_won' trouvée : duels aériens gagnés = 0.")

df_def_counts = pd.concat(
    [
        count_by(ppda_def_mask, "ppda_actions"),
        count_by(is_pressure, "nb_pressions"),
        count_by(is_pressure & (x >= FINAL_THIRD_MIN_X), "nb_pressions_high"),
        count_by(is_pressure & flag("counterpress"), "nb_contre_pressings"),
        count_by(recovery_mask, "nb_recuperations"),
        count_by(recovery_mask & (x >= FINAL_THIRD_MIN_X), "nb_recup_high"),
        df_d[recovery_mask].groupby(["match_id", "team"])["x"].sum().rename("sum_recup_x"),
        count_by(interception_ok, "nb_interceptions"),
        count_by(t == "Clearance", "nb_degagements"),
        count_by(is_tackle, "nb_tacles"),
        count_by(tackle_won, "nb_tacles_gagnes"),
        count_by(aerial_won, "nb_aerien_gagnes"),
        count_by(aerial_lost, "nb_aerien_perdus"),
    ],
    axis=1,
).fillna(0).reset_index()

df_defense = (
    pairs[["match_id", "team"]]
    .merge(df_faced, on=["match_id", "team"], how="left")
    .merge(df_conceded, on=["match_id", "team"], how="left")
    .merge(df_def_counts, on=["match_id", "team"], how="left")
)

# Table des positions des pressions et des récupérations (cartes de chaleur)
df_def_events = pd.concat(
    [
        df_d[is_pressure & x.notna() & y.notna()].assign(event_kind="Pression"),
        df_d[recovery_mask].assign(event_kind="Récupération"),
    ],
    ignore_index=True,
)[["id", "match_id", "match_week", "team", "player", "event_kind", "x", "y"]]

# =====================================================================
# STATISTIQUES INDIVIDUELLES : table joueur × match
# =====================================================================
print("\n👤 Construction des statistiques individuelles joueur-match...")
PLAYER_KEYS = ["match_id", "match_week", "team", "player"]

player_events = df_events[
    (df_events["period"] < 5) & df_events["player"].notna() & df_events["team"].notna()
].copy()

player_base = player_events[PLAYER_KEYS].drop_duplicates()
if "substitution_replacement" in df_events.columns:
    subs = df_events[
        (df_events["type"] == "Substitution")
        & df_events["team"].notna()
        & df_events["substitution_replacement"].notna()
    ].copy()
    if not subs.empty:
        incoming = subs[PLAYER_KEYS[:-1]].copy()
        incoming["player"] = subs["substitution_replacement"].values
        player_base = pd.concat([player_base, incoming], ignore_index=True).drop_duplicates()
else:
    subs = pd.DataFrame()

match_end = (
    df_events[df_events["period"] < 5].groupby("match_id")["temps_secondes"].max()
    .clip(lower=5400).rename("match_end").reset_index()
)
player_minutes = player_base.merge(match_end, on="match_id", how="left")
player_minutes["start_seconds"] = 0.0
player_minutes["end_seconds"] = player_minutes["match_end"]
for row in subs.itertuples(index=False):
    t_sub = float(row.temps_secondes)
    out_mask = (
        (player_minutes["match_id"] == row.match_id) & (player_minutes["match_week"] == row.match_week)
        & (player_minutes["team"] == row.team) & (player_minutes["player"] == row.player)
    )
    in_mask = (
        (player_minutes["match_id"] == row.match_id) & (player_minutes["match_week"] == row.match_week)
        & (player_minutes["team"] == row.team) & (player_minutes["player"] == row.substitution_replacement)
    )
    player_minutes.loc[out_mask, "end_seconds"] = np.minimum(player_minutes.loc[out_mask, "end_seconds"], t_sub)
    player_minutes.loc[in_mask, "start_seconds"] = np.maximum(player_minutes.loc[in_mask, "start_seconds"], t_sub)
player_minutes["minutes"] = ((player_minutes["end_seconds"] - player_minutes["start_seconds"]) / 60).clip(lower=0)

if "position" in player_events.columns:
    pos = player_events["position"].astype("string").str.lower()
    gk_keys = player_events[pos.str.contains("goalkeeper|goal keeper", na=False)][PLAYER_KEYS].drop_duplicates()
    gk_keys["is_goalkeeper"] = True
    player_minutes = player_minutes.merge(gk_keys, on=PLAYER_KEYS, how="left")
    player_minutes["is_goalkeeper"] = player_minutes["is_goalkeeper"].fillna(False)
else:
    player_minutes["is_goalkeeper"] = False
    print("⚠️ Colonne 'position' absente : aucun gardien ne peut être identifié automatiquement.")

def event_flag(df, column):
    return statsbomb_bool(df[column]) if column in df.columns else pd.Series(False, index=df.index)

player_events["interception_raw"] = (player_events["type"] == "Interception").astype(int)
player_events["tackle_raw"] = ((player_events["type"] == "Duel") & (player_events["duel_type"] == "Tackle")).astype(int)
player_events["clearance_raw"] = (player_events["type"] == "Clearance").astype(int)
player_events["aerial_won"] = (
    event_flag(player_events, "pass_aerial_won") | event_flag(player_events, "clearance_aerial_won")
    | event_flag(player_events, "shot_aerial_won") | event_flag(player_events, "miscontrol_aerial_won")
)
player_events["aerial_lost"] = (player_events["type"] == "Duel") & (player_events["duel_type"] == "Aerial Lost")
player_counts = player_events.groupby(PLAYER_KEYS).agg(
    interceptions_raw=("interception_raw", "sum"), tackles_raw=("tackle_raw", "sum"),
    clearances_raw=("clearance_raw", "sum"), aerial_won=("aerial_won", "sum"), aerial_lost=("aerial_lost", "sum"),
).reset_index()
player_counts = player_counts.merge(df_poss_time[["match_id", "team", "possession_pct"]], on=["match_id", "team"], how="left")
# Ajustement à la possession : PADJ = nombre d'événements du match × 2 / (1 + exp(-10 × (possession de l'équipe - 0,5)))
# (possession exprimée entre 0 et 1 : 60 % -> 0,6). Si la possession du match est inconnue, on prend 50 % (facteur = 1).
possession_fraction = player_counts["possession_pct"].fillna(50).clip(0, 100) / 100
padj_factor = 2 / (1 + np.exp(-10 * (possession_fraction - 0.5)))
player_counts["padj_interceptions"] = player_counts["interceptions_raw"] * padj_factor
player_counts["padj_tackles"] = player_counts["tackles_raw"] * padj_factor
player_counts["padj_clearances"] = player_counts["clearances_raw"] * padj_factor

pass_player = df_pass.copy()
pass_player["passes_into_final_third"] = pass_player["is_complete"] & (pass_player["x"] < 80) & (pass_player["end_x"] >= 80)
pass_stats = pass_player.groupby(PLAYER_KEYS).agg(
    key_passes=("is_key_pass", "sum"), progressive_passes=("is_progressive", "sum"),
    passes_into_final_third=("passes_into_final_third", "sum"),
).reset_index()

carry = player_events[player_events["type"] == "Carry"].copy()
# Carry progressif = conduite démarrant avant x = 80 et se terminant à x >= 80.
# Les événements Carry de StatsBomb n'ont pas de champ 'outcome' (une conduite est toujours terminée) :
# on ne filtre sur 'carry_outcome' que si la colonne existe dans vos données.
if "carry_end_x" in carry.columns:
    if "carry_outcome" in carry.columns:
        complete_carry = carry["carry_outcome"].astype("string").str.lower().isin(["complete", "completed", "success"])
    else:
        complete_carry = pd.Series(True, index=carry.index)
    carry["progressive_carries"] = complete_carry & (carry["x"] < 80) & (carry["carry_end_x"] >= 80)
else:
    carry["progressive_carries"] = False
    print("⚠️ Colonne 'carry_end_x' absente : carries progressifs = 0.")
carry_stats = carry.groupby(PLAYER_KEYS)["progressive_carries"].sum().rename("progressive_carries").reset_index()

dribble = player_events[player_events["type"] == "Dribble"].copy()
dribble["completed_dribbles"] = dribble["dribble_outcome"].astype("string").str.lower().isin(["complete", "completed", "success"])
dribble_stats = dribble.groupby(PLAYER_KEYS)["completed_dribbles"].sum().reset_index()

shots_player = df_shots.copy()
if "shot_type" in shots_player.columns:
    stype = shots_player["shot_type"].astype("string").str.lower()
    shots_player["xg_non_penalty"] = shots_player["shot_statsbomb_xg"].where(~stype.str.contains("penalty", na=False), 0)
else:
    shots_player["xg_non_penalty"] = shots_player["shot_statsbomb_xg"]
shots_player["goals"] = shots_player["is_goal"].astype(int)
shot_stats = shots_player.groupby(PLAYER_KEYS).agg(
    xg_non_penalty=("xg_non_penalty", "sum"), goals=("goals", "sum"), shots=("id", "count")
).reset_index()

df_player_stats = (player_minutes.merge(player_counts, on=PLAYER_KEYS, how="left")
                   .merge(pass_stats, on=PLAYER_KEYS, how="left")
                   .merge(carry_stats, on=PLAYER_KEYS, how="left")
                   .merge(dribble_stats, on=PLAYER_KEYS, how="left")
                   .merge(shot_stats, on=PLAYER_KEYS, how="left"))
player_numeric_cols = ["xg_non_penalty", "goals", "shots", "key_passes", "progressive_passes",
                       "passes_into_final_third", "progressive_carries", "completed_dribbles",
                       "padj_interceptions", "padj_tackles", "padj_clearances", "aerial_won", "aerial_lost"]
for col in player_numeric_cols:
    df_player_stats[col] = df_player_stats[col].fillna(0)

# =====================================================================
# FUTURES FEATURES : AJOUTEZ-LES ICI (PPDA, passes progressives...)
# Chaque feature = un DataFrame avec les clés ['match_id', 'match_week', 'team'],
# fusionné ci-dessous avec un merge(..., on=KEYS, how='left').
# =====================================================================

# =====================================================================
# EXPORTATION
# =====================================================================
print("\n💾 Compilation et sauvegarde des fichiers...")

# Table équipe x match (une ligne par équipe et par match joué)
df_features_final = (
    df_possession
    .merge(df_team_shots[KEYS + ["tirs", "buts", "xg"]], on=KEYS, how="left")
    .merge(df_poss_time[KEYS + ["possession_pct"]], on=KEYS, how="left")
    .merge(df_poss_outcomes, on=KEYS, how="left")
    .merge(df_team_passes, on=KEYS, how="left")
    .merge(df_defense, on=["match_id", "team"], how="left")
)
count_cols = [
    "tirs", "buts", "nb_possessions_total", "nb_poss_with_shot", "nb_poss_final_third",
    "nb_passes", "nb_passes_ok", "nb_passes_prog", "nb_passes_forward", "nb_passes_long",
    "nb_passes_pressure", "nb_passes_pressure_ok",
    # Statistiques défensives
    "tirs_conc", "buts_enc", "tirs_contres", "ppda_passes", "ppda_actions",
    "nb_pressions", "nb_pressions_high", "nb_contre_pressings",
    "nb_recuperations", "nb_recup_high", "nb_interceptions", "nb_degagements",
    "nb_tacles", "nb_tacles_gagnes", "nb_aerien_gagnes", "nb_aerien_perdus",
]
for col in count_cols + ["xg", "sum_pass_length", "xg_conc", "sum_recup_x"]:
    df_features_final[col] = df_features_final[col].fillna(0)
for col in count_cols:
    df_features_final[col] = df_features_final[col].astype(int)

features_file = "liga_features_precalculees.parquet"
df_features_final.to_parquet(features_file, index=False)

# Table des tirs individuels (carte des tirs + stats joueurs) : petite et rapide à lire
shots_cols = [
    "id", "match_id", "match_week", "team", "player", "period", "minute",
    "x", "y", "shot_outcome", "shot_statsbomb_xg", "is_goal",
]
shots_file = f"liga_shots_{SEASON}.parquet"
df_shots[shots_cols].to_parquet(shots_file, index=False)

# Table des pertes de balle (zones de pertes)
losses_cols = ["id", "match_id", "match_week", "team", "player", "period", "minute", "loss_type", "x", "y"]
losses_file = f"liga_losses_{SEASON}.parquet"
df_losses[losses_cols].to_parquet(losses_file, index=False)

# Table de toutes les passes (réseau de passes, cartes de chaleur, histogramme)
pass_cols = [
    "id", "match_id", "match_week", "team", "player", "pass_recipient",
    "x", "y", "end_x", "end_y", "pass_length",
    "is_complete", "under_pressure_flag", "is_progressive", "direction", "length_class",
    "is_assist", "is_key_pass",
]
df_passes_out = df_pass[pass_cols].copy()
for c in ["x", "y", "end_x", "end_y", "pass_length"]:
    df_passes_out[c] = df_passes_out[c].astype("float32")
passes_file = f"liga_passes_{SEASON}.parquet"
df_passes_out.to_parquet(passes_file, index=False)

# Table des statistiques individuelles joueur x match pour le radar
player_file = f"liga_player_stats_{SEASON}.parquet"
df_player_stats.to_parquet(player_file, index=False)

# Table des pressions et des récupérations (cartes de chaleur défensives)
df_def_events["x"] = df_def_events["x"].astype("float32")
df_def_events["y"] = df_def_events["y"].astype("float32")
defense_file = f"liga_defense_events_{SEASON}.parquet"
df_def_events.to_parquet(defense_file, index=False)

tot_poss = df_features_final["nb_possessions_total"].sum()
print("=" * 60)
print("🎉 SUCCÈS : fichiers générés")
print(f"   - '{features_file}' ({len(df_features_final)} lignes équipe x match)")
print(f"   - '{shots_file}' ({len(df_shots)} tirs)")
print(f"   - '{losses_file}' ({len(df_losses)} pertes de balle)")
print(f"   - '{passes_file}' ({len(df_passes_out)} passes)")
print(f"   - '{player_file}' ({len(df_player_stats)} lignes joueur-match)")
print(f"   - '{defense_file}' ({len(df_def_events)} pressions et récupérations)")
_ppda_actions = df_features_final["ppda_actions"].sum()
print(f"   - Contrôle défense : PPDA moyen Liga = {df_features_final['ppda_passes'].sum() / max(_ppda_actions, 1):.2f}, "
      f"{df_features_final['nb_pressions'].mean():.0f} pressions / équipe / match, "
      f"{df_features_final['nb_contre_pressings'].mean():.0f} contre-pressings, "
      f"{df_features_final['nb_recuperations'].mean():.0f} récupérations, "
      f"{df_features_final['nb_aerien_gagnes'].mean():.0f} duels aériens gagnés / équipe / match")
_outfield = df_player_stats[~df_player_stats["is_goalkeeper"].astype(bool)].groupby("player")["minutes"].sum()
print(f"   - Contrôle joueurs : {int((_outfield >= 450).sum())} joueurs de champ >= 450 min sur la saison ; "
      f"totaux Liga : {int(df_player_stats['key_passes'].sum())} passes clés, "
      f"{int(df_player_stats['progressive_carries'].sum())} carries progressifs, "
      f"{int(df_player_stats['completed_dribbles'].sum())} dribbles réussis")
print(f"   - Contrôle passes : précision ={df_pass['is_complete'].mean() * 100:.1f} %, "
      f"{df_pass['is_forward'].mean() * 100:.1f} % vers l'avant, "
      f"{df_pass['is_progressive'].sum() / df_pass['match_id'].nunique() / 2:.0f} progressives / équipe / match")
print(f"   - Contrôle possession : moyenne = {df_features_final['possession_pct'].mean():.1f} % (attendu ≈ 50)")
if tot_poss:
    print(f"   - Contrôle issues : {df_features_final['nb_poss_with_shot'].sum() / tot_poss * 100:.1f} % des possessions "
          f"avec tir, {df_features_final['nb_poss_final_third'].sum() / tot_poss * 100:.1f} % atteignent le dernier tiers "
          f"(sur {tot_poss} possessions, toute la ligue)")
print("=" * 60)