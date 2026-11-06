import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import textwrap
from mplsoccer import VerticalPitch, Pitch
import base64
from pathlib import Path

YARD_TO_M = 0.9144
SHORT_PASS_MAX_YARDS = 15
LONG_PASS_MIN_YARDS = 32

CMAP_LOSS = LinearSegmentedColormap.from_list("loss", ["#1e293b", "#b91c1c", "#f87171"])
CMAP_START = LinearSegmentedColormap.from_list("start", ["#1e293b", "#1d4ed8", "#60a5fa"])
CMAP_END = LinearSegmentedColormap.from_list("end", ["#1e293b", "#047857", "#34d399"])
CMAP_PRESS = LinearSegmentedColormap.from_list("press", ["#1e293b", "#b45309", "#fbbf24"])
CMAP_RECOV = LinearSegmentedColormap.from_list("recov", ["#1e293b", "#0f766e", "#2dd4bf"])

LOWER_IS_BETTER = ["xg_conc_match", "tirs_conc_match", "xg_conc_tir", "buts_enc", "ppda"]


def short_name(full_name):
    parts = str(full_name).split()
    return f"{parts[0][0]}. {parts[-1]}" if len(parts) > 1 else str(full_name)


def render_zone_heatmap(xs, ys, cmap, figsize=(7, 5), show_attack_direction=False):
    """Carte de chaleur par zones (4 x 3) avec la part de chaque zone en pourcentage."""
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1e293b", line_color="#64748b", line_zorder=2)
    fig, ax = pitch.draw(figsize=figsize)
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#1e293b")
    bs = pitch.bin_statistic(xs, ys, statistic="count", bins=(4, 3), normalize=True)
    pitch.heatmap(bs, ax=ax, cmap=cmap, edgecolors="#1e293b", alpha=0.9)
    pitch.label_heatmap(bs, ax=ax, str_format="{:.0%}", color="white", fontsize=10)
    # NOUVEAU : flèche + texte du sens de l'attaque (dans les coordonnées de l'axes, sous le terrain)
    if show_attack_direction:
        ax.annotate("", xy=(0.98, -0.04), xytext=(0.02, -0.04),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="#e2e8f0", lw=1.6,
                                    shrinkA=0, shrinkB=0),
                    annotation_clip=False)
        ax.text(0.5, -0.075, "Sens de l'attaque", ha="center", va="top",
                transform=ax.transAxes, color="#e2e8f0", fontsize=8,
                fontweight="bold", style="italic", clip_on=False)
    # Même marge basse que la heatmap de récupération, qui réserve de l'espace à sa légende.
    fig.subplots_adjust(bottom=0.27)
    st.pyplot(fig)
    plt.close(fig)


def render_recovery_thirds_heatmap(xs, ys, team_height_m, league_height_m, figsize=(4.5, 3.2)):
    """Répartition des récupérations dans les trois tiers avec deux repères de hauteur."""
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1e293b", line_color="#64748b", line_zorder=2)
    fig, ax = pitch.draw(figsize=figsize)
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#1e293b")
    bs = pitch.bin_statistic(xs, ys, statistic="count", bins=(3, 1), normalize=True)
    pitch.heatmap(bs, ax=ax, cmap=CMAP_RECOV, edgecolors="#1e293b", alpha=0.9)
    pitch.label_heatmap(bs, ax=ax, str_format="{:.0%}", color="white", fontsize=10)

    team_x = float(team_height_m) / YARD_TO_M if pd.notna(team_height_m) else np.nan
    league_x = float(league_height_m) / YARD_TO_M if pd.notna(league_height_m) else np.nan
    if np.isfinite(team_x):
        ax.axvline(team_x, color="#f43f5e", linewidth=2.2, linestyle="-", zorder=5)
    if np.isfinite(league_x):
        ax.axvline(league_x, color="#fbbf24", linewidth=2.2, linestyle="--", zorder=5)

    for xpos, label in [(20, "Propre tiers"), (60, "Tiers moyen"), (100, "Tiers adverse")]:
        ax.text(xpos, 76, label, ha="center", va="top", color="#e2e8f0", fontsize=8, fontweight="bold")
    legend_handles = [
        Line2D([0], [0], color="#f43f5e", lw=2.2,
               label=f"Hauteur moyenne de récupération — équipe : {team_height_m:.1f} m"),
        Line2D([0], [0], color="#fbbf24", lw=2.2, linestyle="--",
               label=f"Hauteur moyenne de récupération — Liga : {league_height_m:.1f} m"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.01, 0.99),
              ncol=1, frameon=False, labelcolor="#e2e8f0", fontsize=7,
              handlelength=2.2, borderaxespad=0.0)
    ax.annotate("", xy=(0.98, -0.04), xytext=(0.02, -0.04),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#e2e8f0", lw=1.6,
                                shrinkA=0, shrinkB=0),
                annotation_clip=False)
    ax.text(0.5, -0.075, "Sens de l'attaque", ha="center", va="top",
            transform=ax.transAxes, color="#e2e8f0", fontsize=8,
            fontweight="bold", style="italic", clip_on=False)
    # Même zone de dessin et mêmes marges que la heatmap de pression.
    fig.subplots_adjust(bottom=0.27)
    st.pyplot(fig)
    plt.close(fig)


MIN_MINUTES = 450

PLAYER_SUM_COLS = [
    "minutes", "xg_non_penalty", "goals", "shots", "key_passes",
    "padj_interceptions", "padj_tackles", "padj_clearances", "aerial_won", "aerial_lost",
    "progressive_passes", "progressive_carries", "passes_into_final_third", "completed_dribbles",
]

CATEGORY_COLORS = {"Attaque": "#f43f5e", "Défense": "#60a5fa", "Progression": "#34d399"}

PIZZA_METRICS = [
    ("xg_np_90", "xG non-penalty / 90", "Attaque", "{:.2f}"),
    ("buts_90", "Buts / 90", "Attaque", "{:.2f}"),
    ("tirs_90", "Tirs / 90", "Attaque", "{:.2f}"),
    ("passes_cles_90", "Passes clés / 90", "Attaque", "{:.2f}"),
    ("padj_int_90", "PADJ Interceptions / 90", "Défense", "{:.2f}"),
    ("padj_tacles_90", "PADJ Tacles / 90", "Défense", "{:.2f}"),
    ("padj_deg_90", "PADJ Dégagements / 90", "Défense", "{:.2f}"),
    ("aerien_pct", "Duels aériens gagnés (%)", "Défense", "{:.1f} %"),
    ("passes_prog_90", "Passes progressives / 90", "Progression", "{:.2f}"),
    ("carries_prog_90", "Carries progressifs / 90", "Progression", "{:.2f}"),
    ("passes_dt_90", "Passes dernier tiers / 90", "Progression", "{:.2f}"),
    ("dribbles_90", "Dribbles réussis / 90", "Progression", "{:.2f}"),
]


def compute_player_profiles(df):
    g = df.groupby("player")[PLAYER_SUM_COLS].sum()
    g = g[g["minutes"] >= MIN_MINUTES].copy()
    per90 = lambda col: g[col] / g["minutes"] * 90
    aerial_total = g["aerial_won"] + g["aerial_lost"]

    prof = pd.DataFrame(index=g.index)
    prof["minutes"] = g["minutes"]
    prof["xg_np_90"] = per90("xg_non_penalty")
    prof["buts_90"] = per90("goals")
    prof["tirs_90"] = per90("shots")
    prof["passes_cles_90"] = per90("key_passes")
    prof["padj_int_90"] = per90("padj_interceptions")
    prof["padj_tacles_90"] = per90("padj_tackles")
    prof["padj_deg_90"] = per90("padj_clearances")
    prof["aerien_pct"] = (g["aerial_won"] / aerial_total * 100).where(aerial_total > 0, 0.0)
    prof["passes_prog_90"] = per90("progressive_passes")
    prof["carries_prog_90"] = per90("progressive_carries")
    prof["passes_dt_90"] = per90("passes_into_final_third")
    prof["dribbles_90"] = per90("completed_dribbles")
    return prof


def render_pizza(player_name, team_name, minutes, raw_values, scores):
    n = len(PIZZA_METRICS)
    theta = np.arange(n) * 2 * np.pi / n
    width = 2 * np.pi / n * 0.94
    colors = [CATEGORY_COLORS[m[2]] for m in PIZZA_METRICS]
    values = [scores[m[0]] for m in PIZZA_METRICS]

    fig = plt.figure(figsize=(8.5, 8.5), facecolor="#0f172a")
    ax = fig.add_axes([0.23, 0.2, 0.54, 0.54], polar=True)
    ax.set_facecolor("#1e293b")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.bar(theta, [100] * n, width=width, color="#334155", alpha=0.35, zorder=1)
    ax.bar(theta, values, width=width, color=colors, edgecolor="#0f172a", linewidth=1.5, alpha=0.95, zorder=3)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.grid(color="#64748b", alpha=0.35, linewidth=0.8)
    ax.spines["polar"].set_visible(False)

    for angle, (key, label, category, fmt), value in zip(theta, PIZZA_METRICS, values):
        if value >= 22:
            ax.text(angle, value - 9, f"{value:.0f}", ha="center", va="center", color="white",
                    fontsize=10, fontweight="bold", zorder=5)
        else:
            ax.text(angle, value + 8, f"{value:.0f}", ha="center", va="center", color="#e2e8f0",
                    fontsize=10, fontweight="bold", zorder=5)
        sx, sy = np.sin(angle), np.cos(angle)
        ha = "left" if sx > 0.3 else "right" if sx < -0.3 else "center"
        va = "bottom" if sy > 0.3 else "top" if sy < -0.3 else "center"
        wrapped = textwrap.fill(label, width=20)
        ax.text(angle, 110, f"{wrapped}\n{fmt.format(raw_values[key])}", ha=ha, va=va, color=CATEGORY_COLORS[category],
                fontsize=9, fontweight="bold", linespacing=1.25, zorder=5)

    handles = [Patch(facecolor=c, label=k) for k, c in CATEGORY_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, labelcolor="white", fontsize=11,
               bbox_to_anchor=(0.5, 0.04))
    fig.text(0.5, 0.95, player_name, ha="center", va="center", color="white", fontsize=18, fontweight="bold")
    fig.text(0.5, 0.915, f"{team_name} · {int(minutes)} minutes jouées", ha="center", va="center",
             color="#94a3b8", fontsize=11)
    return fig


st.set_page_config(page_title="Coach Dashboard Pro", layout="wide", page_icon="⚽")

st.markdown("""
<style>
    /* Agrandit le label des expanders (= titres des 4 catégories) */
    [data-testid="stExpander"] summary {
        font-size: 1.4rem !important;
    }
    [data-testid="stExpander"] summary p {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)



@st.cache_data
def load_data():
    df_team = pd.read_parquet("liga_features_precalculees.parquet")
    df_shots = pd.read_parquet("liga_shots_15_16.parquet")
    df_losses = pd.read_parquet("liga_losses_15_16.parquet")
    df_passes = pd.read_parquet("liga_passes_15_16.parquet")
    df_defense = pd.read_parquet("liga_defense_events_15_16.parquet")
    df_player_stats = pd.read_parquet("liga_player_stats_15_16.parquet")
    return df_team, df_shots, df_losses, df_passes, df_defense, df_player_stats


try:
    df_team, df_shots, df_losses, df_passes, df_defense, df_player_stats = load_data()
except FileNotFoundError:
    st.error("❌ Fichiers manquants. Lancez d'abord extraction_totale.py puis feature_engineering.py.")
    st.stop()


def compute_league_indicators(df, max_week):
    hist = df[df["match_week"] <= max_week].copy()
    hist["poss_pondere"] = hist["duree_possession_moyenne"] * hist["nb_possessions"]

    g = hist.groupby("team").agg(
        n_matches=("match_id", "count"),
        xg=("xg", "sum"), tirs=("tirs", "sum"), buts=("buts", "sum"),
        poss_pondere=("poss_pondere", "sum"), nb_possessions=("nb_possessions", "sum"),
        possession_pct=("possession_pct", "mean"),
        nb_possessions_total=("nb_possessions_total", "sum"),
        nb_poss_with_shot=("nb_poss_with_shot", "sum"),
        nb_poss_final_third=("nb_poss_final_third", "sum"),
        nb_passes=("nb_passes", "sum"), nb_passes_ok=("nb_passes_ok", "sum"),
        nb_passes_prog=("nb_passes_prog", "sum"), nb_passes_forward=("nb_passes_forward", "sum"),
        nb_passes_long=("nb_passes_long", "sum"), sum_pass_length=("sum_pass_length", "sum"),
        nb_passes_pressure=("nb_passes_pressure", "sum"),
        nb_passes_pressure_ok=("nb_passes_pressure_ok", "sum"),
        tirs_conc=("tirs_conc", "sum"), buts_enc=("buts_enc", "sum"), xg_conc=("xg_conc", "sum"),
        tirs_contres=("tirs_contres", "sum"),
        ppda_passes=("ppda_passes", "sum"), ppda_actions=("ppda_actions", "sum"),
        nb_pressions=("nb_pressions", "sum"), nb_pressions_high=("nb_pressions_high", "sum"),
        nb_contre_pressings=("nb_contre_pressings", "sum"),
        nb_recuperations=("nb_recuperations", "sum"), nb_recup_high=("nb_recup_high", "sum"),
        sum_recup_x=("sum_recup_x", "sum"),
        nb_interceptions=("nb_interceptions", "sum"),
        nb_degagements=("nb_degagements", "sum"),
        nb_tacles=("nb_tacles", "sum"), nb_tacles_gagnes=("nb_tacles_gagnes", "sum"),
        nb_aerien_gagnes=("nb_aerien_gagnes", "sum"), nb_aerien_perdus=("nb_aerien_perdus", "sum"),
    )

    ind = pd.DataFrame(index=g.index)
    ind["xg_total"] = g["xg"]
    ind["xg_match"] = g["xg"] / g["n_matches"]
    ind["xg_tir"] = (g["xg"] / g["tirs"]).where(g["tirs"] > 0)
    ind["tirs_match"] = g["tirs"] / g["n_matches"]
    ind["buts"] = g["buts"]
    ind["duree_possession"] = (g["poss_pondere"] / g["nb_possessions"]).where(g["nb_possessions"] > 0)
    ind["possession_pct"] = g["possession_pct"]
    ind["poss_tir_pct"] = (g["nb_poss_with_shot"] / g["nb_possessions_total"] * 100).where(g["nb_possessions_total"] > 0)
    ind["poss_dernier_tiers_pct"] = (g["nb_poss_final_third"] / g["nb_possessions_total"] * 100).where(g["nb_possessions_total"] > 0)
    ind["passes_match"] = g["nb_passes"] / g["n_matches"]
    ind["precision_passes"] = (g["nb_passes_ok"] / g["nb_passes"] * 100).where(g["nb_passes"] > 0)
    ind["passes_prog_match"] = g["nb_passes_prog"] / g["n_matches"]
    ind["passes_avant_pct"] = (g["nb_passes_forward"] / g["nb_passes"] * 100).where(g["nb_passes"] > 0)
    ind["longueur_moy_passe"] = (g["sum_pass_length"] / g["nb_passes"] * YARD_TO_M).where(g["nb_passes"] > 0)
    ind["passes_longues_pct"] = (g["nb_passes_long"] / g["nb_passes"] * 100).where(g["nb_passes"] > 0)
    ind["precision_pression"] = (g["nb_passes_pressure_ok"] / g["nb_passes_pressure"] * 100).where(g["nb_passes_pressure"] > 0)
    ind["xg_conc_total"] = g["xg_conc"]
    ind["xg_conc_match"] = g["xg_conc"] / g["n_matches"]
    ind["tirs_conc_match"] = g["tirs_conc"] / g["n_matches"]
    ind["xg_conc_tir"] = (g["xg_conc"] / g["tirs_conc"]).where(g["tirs_conc"] > 0)
    ind["buts_enc"] = g["buts_enc"]
    ind["ppda"] = (g["ppda_passes"] / g["ppda_actions"]).where(g["ppda_actions"] > 0)
    ind["pressions_match"] = g["nb_pressions"] / g["n_matches"]
    ind["pressions_hautes_pct"] = (g["nb_pressions_high"] / g["nb_pressions"] * 100).where(g["nb_pressions"] > 0)
    ind["contre_pressings_match"] = g["nb_contre_pressings"] / g["n_matches"]
    ind["recuperations_match"] = g["nb_recuperations"] / g["n_matches"]
    ind["hauteur_recup"] = (g["sum_recup_x"] / g["nb_recuperations"] * YARD_TO_M).where(g["nb_recuperations"] > 0)
    ind["recup_hautes_pct"] = (g["nb_recup_high"] / g["nb_recuperations"] * 100).where(g["nb_recuperations"] > 0)
    ind["tacles_gagnes_pct"] = (g["nb_tacles_gagnes"] / g["nb_tacles"] * 100).where(g["nb_tacles"] > 0)
    total_duels_aeriens = g["nb_aerien_gagnes"] + g["nb_aerien_perdus"]
    ind["duels_aeriens_gagnes_pct"] = (g["nb_aerien_gagnes"] / total_duels_aeriens * 100).where(total_duels_aeriens > 0)
    ind["interceptions_match"] = g["nb_interceptions"] / g["n_matches"]
    ind["degagements_match"] = g["nb_degagements"] / g["n_matches"]
    ind["tirs_contres_match"] = g["tirs_contres"] / g["n_matches"]

    ranks = ind.rank(ascending=False, method="min")
    for col in LOWER_IS_BETTER:
        ranks[col] = ind[col].rank(ascending=True, method="min")
    return ind, ranks


# NOUVEAU : indicateurs transverses (calculés à part, pas dans compute_league_indicators
# car ils reposent sur df_shots / df_losses et non sur la table équipe × match)
def compute_league_shots_outside(df_shots, max_week):
    """% de tirs hors surface adverse, par équipe sur J1..max_week. Une valeur basse = meilleur."""
    shots = df_shots[df_shots["match_week"] <= max_week].dropna(subset=["x", "y"])
    if shots.empty:
        return pd.Series(dtype=float)
    in_penalty_area = (shots["x"] >= 102) & shots["y"].between(18, 62)
    return (~in_penalty_area).groupby(shots["team"]).mean() * 100


def compute_league_losses_own_third(df_losses, max_week):
    """% de pertes dans son propre tiers (x < 40), par équipe sur J1..max_week. Une valeur basse = meilleur."""
    losses = df_losses[df_losses["match_week"] <= max_week].dropna(subset=["x"])
    if losses.empty:
        return pd.Series(dtype=float)
    return (losses["x"] < 40).groupby(losses["team"]).mean() * 100


def ordinal_fr(n):
    return "1er" if n == 1 else f"{n}e"

def compute_league_standings(df, max_week):
    """Classement de la Liga sur J1..max_week.
    Critères : points (3/1/0) → différence de buts → buts marqués.
    Retourne un DataFrame indexé par équipe avec position, points, etc."""
    hist = df[df["match_week"] <= max_week].copy()

    # Attribution des points : 3 si victoire, 1 si nul, 0 sinon
    hist["points"] = 0
    hist.loc[hist["buts"] > hist["buts_enc"], "points"] = 3
    hist.loc[hist["buts"] == hist["buts_enc"], "points"] = 1

    g = hist.groupby("team").agg(
        points=("points", "sum"),
        buts=("buts", "sum"),
        buts_enc=("buts_enc", "sum"),
        n_matches=("match_id", "count"),
    )
    g["diff"] = g["buts"] - g["buts_enc"]

    # Tri puis position
    g = g.sort_values(
        ["points", "diff", "buts"],
        ascending=[False, False, False],
    )
    g["position"] = range(1, len(g) + 1)
    return g

LOGO_DIR = Path("logo")


@st.cache_data(show_spinner=False)
def load_logo_b64(team_name: str) -> str | None:
    """Charge l'écusson PNG de l'équipe et le retourne en base64 (mis en cache)."""
    path = LOGO_DIR / f"{team_name}.png"
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode()


def team_logo_html(team_name: str, height: int = 56) -> str:
    """HTML de l'écusson : vrai logo si présent, sinon cercle avec initiales."""
    b64 = load_logo_b64(team_name)
    if b64:
        return (
            f'<img src="data:image/png;base64,{b64}" '
            f'alt="{team_name}" '
            f'style="height:{height}px; width:auto; object-fit:contain; '
            f'filter: drop-shadow(0 2px 6px rgba(0,0,0,0.35));" />'
        )
    # Fallback : initiales dans un cercle sombre
    initials = "".join(w[0] for w in team_name.split()[:2]).upper() or "?"
    return (
        f'<div style="width:{height}px; height:{height}px; border-radius:50%; '
        f'background:linear-gradient(135deg,#475569,#94a3b8); '
        f'display:inline-flex; align-items:center; justify-content:center; '
        f'color:white; font-weight:700; font-size:{height // 3}px; '
        f'border:2px solid rgba(255,255,255,0.25);">{initials}</div>'
    )

def _help_marker(help_text):
    """Petit '?' gris circulaire qui imite l'icône d'aide native de Streamlit
    (celle affichée à côté du label des st.metric avec help=...).
    Utilise l'attribut HTML `title` pour le tooltip au survol."""
    if not help_text:
        return ""
    escaped = (help_text.replace('"', "&quot;")
                       .replace("<", "&lt;")
                       .replace(">", "&gt;"))
    return (
        f'<span title="{escaped}" style="'
        f'display: inline-flex; align-items: center; justify-content: center; '
        f'width: 16px; height: 16px; border-radius: 50%; '
        f'border: 1.5px solid #a3a8b8; color: #a3a8b8; '
        f'font-size: 10px; font-weight: 600; line-height: 1; '
        f'cursor: help; vertical-align: middle; margin-left: 6px; '
        f'user-select: none;">?</span>'
    )


st.sidebar.title("🎮 Panneau Tactique Coach")

st.sidebar.markdown("""
<div style="text-align:center; margin:-6px 0 20px 0;">
    <span style="
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(244,63,94,0.18), rgba(96,165,250,0.18));
        border: 1px solid rgba(244,63,94,0.35);
        color: #f8fafc;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        box-shadow: 0 2px 12px rgba(244,63,94,0.15);
    ">
        <span style="color:#f43f5e; font-size:0.9rem;">●</span>
        La Liga · Saison 2015/2016
    </span>
</div>
""", unsafe_allow_html=True)

all_teams = sorted(df_team["team"].dropna().unique())
team_selected = st.sidebar.selectbox(
    "1. Équipe à étudier", all_teams,
    index=all_teams.index("Real Madrid") if "Real Madrid" in all_teams else 0,
)

all_weeks = sorted(df_team["match_week"].unique())
week_selected = st.sidebar.selectbox("2. Journée à préparer", all_weeks, index=min(15, len(all_weeks) - 1))
max_history_week = week_selected - 1

st.sidebar.markdown("---")
st.sidebar.markdown("""
<style>
@keyframes shimmer {
    0%   { background-position:   0% 50%; }
    100% { background-position: 200% 50%; }
}
.sidebar-signature {
    text-align: center;
    padding: 4px 0 10px 0;
}
.sidebar-signature .role {
    font-size: 0.66rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    font-weight: 600;
    line-height: 1.5;
    margin-bottom: 10px;
}
.sidebar-signature .role .accent {
    color: #f43f5e;
}
.sidebar-signature .divider {
    width: 40px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #475569, transparent);
    margin: 0 auto 10px auto;
}
.sidebar-signature .name {
    font-family: 'Georgia', 'Cambria', serif;
    font-style: italic;
    font-size: 1.35rem;
    font-weight: 600;
    background: linear-gradient(90deg, #60a5fa, #f43f5e, #fbbf24, #60a5fa);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 5s linear infinite;
    letter-spacing: 1px;
}
</style>
<div class="sidebar-signature">
    <div class="name">Ramzi Abdelhafidh</div>
    <div class="role">Football Data <span class="accent">Analyst</span> / <span class="accent">Scientist</span></div>
    <div class="divider"></div>
    
</div>
""", unsafe_allow_html=True)

# === CHANGEMENT 1 : noms des onglets ===
tab_team, tab_player = st.tabs(["Performance de l'équipe", "Statistiques du joueur"])

team_history = df_team[(df_team["team"] == team_selected) & (df_team["match_week"] <= max_history_week)]
team_shots = df_shots[(df_shots["team"] == team_selected) & (df_shots["match_week"] <= max_history_week)]
team_losses = df_losses[(df_losses["team"] == team_selected) & (df_losses["match_week"] <= max_history_week)]
team_passes = df_passes[(df_passes["team"] == team_selected) & (df_passes["match_week"] <= max_history_week)]
team_defense = df_defense[(df_defense["team"] == team_selected) & (df_defense["match_week"] <= max_history_week)]

# ==========================================
# ONGLET 1 : PERFORMANCE ÉQUIPE
# ==========================================
with tab_team:
    required_cols = {
        "possession_pct", "nb_possessions_total", "nb_poss_with_shot", "nb_poss_final_third",
        "nb_passes", "nb_passes_ok", "nb_passes_prog", "nb_passes_forward", "nb_passes_long",
        "sum_pass_length", "nb_passes_pressure", "nb_passes_pressure_ok",
        "tirs_conc", "buts_enc", "xg_conc", "tirs_contres", "ppda_passes", "ppda_actions",
        "nb_pressions", "nb_pressions_high", "nb_contre_pressings", "nb_recuperations", "nb_recup_high",
        "sum_recup_x", "nb_interceptions", "nb_degagements", "nb_tacles", "nb_tacles_gagnes",
        "nb_aerien_gagnes", "nb_aerien_perdus",
    }

    if max_history_week < 1:
        st.warning("⚠️ Sélectionnez la Journée 2 ou plus.")
    elif team_history.empty:
        st.warning("Aucune donnée disponible.")
    elif not required_cols.issubset(team_history.columns):
        st.error("❌ Colonnes de possession absentes : relancez feature_engineering.py.")
    else:
        n_matches = len(team_history)
        ind, ranks = compute_league_indicators(df_team, max_history_week)
        n_teams = len(ind)
        me = ind.loc[team_selected]

        # NOUVEAU : classement et points à la fin de la journée J(X-1)
        standings = compute_league_standings(df_team, max_history_week)
        my_pos = int(standings.loc[team_selected, "position"])
        my_pts = int(standings.loc[team_selected, "points"])
        my_played = int(standings.loc[team_selected, "n_matches"])
        my_diff = int(standings.loc[team_selected, "diff"])

        def kpi(label, key, value_text, help_text=None):
            rank = ranks.loc[team_selected, key]
            if key in LOWER_IS_BETTER:
                rank_explanation = "Une valeur plus basse correspond à un meilleur classement pour cet indicateur."
            else:
                rank_explanation = "Une valeur plus élevée correspond à un meilleur classement pour cet indicateur."
            rank_help = (
                f"Le classement compare l'équipe sélectionnée aux {n_teams} équipes de la Liga "
                f"sur la même période historique. {rank_explanation}"
            )
            complete_help = f"{help_text} {rank_help}" if help_text else rank_help
            st.metric(label=label, value=value_text, help=complete_help)
            if pd.notna(rank):
                st.caption(f"🏅 {ordinal_fr(int(rank))} sur {n_teams} en Liga")

        # === CHANGEMENT 2 : le "?" est maintenant à côté du label (tooltip natif HTML) ===
        def indicator_bar(label, key, value_format=".1f", help_text=None):
            value = me[key]
            league_mean = ind[key].dropna().mean()
            rank = ranks.loc[team_selected, key]

            if pd.isna(value) or pd.isna(league_mean):
                st.info(f"{label} : donnée indisponible")
                return

            # En-tête : label + point d'interrogation avec tooltip (HTML title)
            st.markdown(f"**{label}** {_help_marker(help_text)}", unsafe_allow_html=True)

            chart_df = pd.DataFrame({"Équipe": [team_selected], "Valeur": [float(value)]})
            max_value = max(float(value), float(league_mean), 0.0)
            domain_max = max(max_value * 1.22, 1.0)

            bar = (
                alt.Chart(chart_df)
                .mark_bar(color="#f43f5e", size=52, cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                .encode(
                    x=alt.X("Équipe:N", title=None, axis=alt.Axis(labels=False, ticks=False, domain=False)),
                    y=alt.Y("Valeur:Q", title=None, scale=alt.Scale(domain=[0, domain_max]),
                            axis=alt.Axis(format="~f", tickCount=4)),
                    tooltip=[alt.Tooltip("Équipe:N", title="Équipe"),
                             alt.Tooltip("Valeur:Q", title=label, format=value_format)],
                )
            )
            bar_value = (
                alt.Chart(chart_df)
                .mark_text(dy=-8, color="white", fontWeight="bold", fontSize=12)
                .encode(
                    x=alt.X("Équipe:N", title=None, axis=alt.Axis(labels=False, ticks=False, domain=False)),
                    y=alt.Y("Valeur:Q", scale=alt.Scale(domain=[0, domain_max])),
                    text=alt.Text("Valeur:Q", format=value_format),
                )
            )
            mean_df = pd.DataFrame({"Équipe": [team_selected], "Moyenne Liga": [float(league_mean)]})
            mean_line = (
                alt.Chart(mean_df)
                .mark_tick(orient="horizontal", color="#fbbf24", thickness=2, size=52)
                .encode(
                    x=alt.X("Équipe:N", title=None, axis=alt.Axis(labels=False, ticks=False, domain=False)),
                    y=alt.Y("Moyenne Liga:Q", scale=alt.Scale(domain=[0, domain_max])),
                )
            )
            mean_value = (
                alt.Chart(mean_df)
                .mark_text(align="left", dx=5, dy=-7, color="#fbbf24", fontWeight="bold", fontSize=11)
                .encode(
                    x=alt.value(5),
                    y=alt.Y("Moyenne Liga:Q", scale=alt.Scale(domain=[0, domain_max])),
                    text=alt.Text("Moyenne Liga:Q", format=value_format),
                    tooltip=[alt.Tooltip("Moyenne Liga:Q", title="Moyenne Liga", format=value_format)],
                )
            )
            legend_inside = (
                alt.Chart(pd.DataFrame({"texte": ["━ Moyenne Liga"]}))
                .mark_text(align="left", baseline="top", color="#fbbf24", fontWeight="bold", fontSize=10)
                .encode(x=alt.value(5), y=alt.value(8), text="texte:N")
            )
            st.altair_chart(
                (bar + bar_value + mean_line + mean_value + legend_inside).properties(width=130, height=180),
                use_container_width=True,
            )
            if pd.notna(rank):
                st.caption(f"🏅 {ordinal_fr(int(rank))} sur {n_teams} en Liga")

        st.markdown(
            f'<div style="display:flex; align-items:center; gap:22px; margin:4px 0 10px 0;">'
            f'{team_logo_html(team_selected, height=90)}'
            f'<h2 style="margin:0; padding:0; font-size:2.4rem; font-weight:700; '
            f'color:#f8fafc; line-height:1.15; letter-spacing:-0.5px;">'
            f'📊 Analyse Collective : {team_selected}'
            f'</h2>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Couleur du badge selon la zone du classement (contexte La Liga)
        if my_pos <= 4:
            zone_color, zone_label = "#10b981", "Ligue des champions"   # vert
        elif my_pos <= 6:
            zone_color, zone_label = "#3b82f6", "Europa League"         # bleu
        elif my_pos >= 18:
            zone_color, zone_label = "#ef4444", "Zone de relégation"    # rouge
        else:
            zone_color, zone_label = "#94a3b8", "Milieu de tableau"     # gris

        # Médaille pour le top 3
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(my_pos, "")

        diff_sign = "+" if my_diff > 0 else ("−" if my_diff < 0 else "")
        diff_color = "#10b981" if my_diff > 0 else ("#ef4444" if my_diff < 0 else "#94a3b8")

        st.markdown(
            f'<div style="display:flex; align-items:center; gap:14px; flex-wrap:wrap; '
            f'margin:-6px 0 18px 6px; font-size:0.95rem;">'

            # Badge de position (cercle coloré)
            f'<span style="display:inline-flex; align-items:center; justify-content:center; '
            f'min-width:44px; height:36px; padding:0 10px; border-radius:8px; '
            f'background:linear-gradient(135deg,{zone_color},{zone_color}cc); '
            f'color:white; font-weight:800; font-size:1.15rem; letter-spacing:0.5px; '
            f'box-shadow:0 2px 8px {zone_color}55;">'
            f'{medal} {ordinal_fr(my_pos)}'
            f'</span>'

            # Points
            f'<span style="color:#f8fafc; font-weight:700; font-size:1.1rem;">'
            f'{my_pts} <span style="font-weight:500; font-size:0.9rem; color:#94a3b8;">points</span>'
            f'</span>'

            # Séparateur
            f'<span style="color:#475569;">·</span>'

            # Matchs joués
            f'<span style="color:#94a3b8; font-size:0.9rem;">'
            f'{my_played} match{"s" if my_played > 1 else ""} joué{"s" if my_played > 1 else ""}'
            f'</span>'

            # Différence de buts
            f'<span style="color:#475569;">·</span>'
            f'<span style="color:{diff_color}; font-weight:600; font-size:0.9rem;">'
            f'{diff_sign}{my_diff} buts'
            f'</span>'

            # Zone (badge discret)
            f'<span style="padding:3px 10px; border-radius:12px; '
            f'background:{zone_color}22; color:{zone_color}; '
            f'font-size:0.72rem; font-weight:700; text-transform:uppercase; '
            f'letter-spacing:1.2px; border:1px solid {zone_color}55;">'
            f'{zone_label}'
            f'</span>'

            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(f"📈 Tendance sur les **{n_matches} premiers matchs**")
        st.caption(
            "Classement : 1er = valeur la plus élevée de la Liga, "
            f"calculée sur la même période (journées 1 à {max_history_week}) pour les {n_teams} équipes."
        )

        # ==========================================
        # === CHANGEMENT 3 : EXPANDER + ORDRE ===
        # ==========================================
        with st.expander("⚽ Statistiques offensives", expanded=True):

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                indicator_bar("Moyenne xG / match", "xg_match", ".2f",
                              "xG moyen produit par match sur la période historique.")
            with c2:
                indicator_bar("Qualité de tir moyenne", "xg_tir", ".2f",
                              "xG moyen par tir. Une valeur élevée indique des occasions moyennement plus dangereuses.")
            with c3:
                indicator_bar("Volume de tirs / match", "tirs_match", ".1f",
                              "Nombre moyen de tirs tentés par match.")
            with c4:
                indicator_bar("Buts inscrits (cumul)", "buts", ".0f",
                              "Nombre total de buts inscrits sur la période historique.")

            st.write("### 👤 Principales contributions offensives")
            table_shots, table_creation = st.columns(2)

            with table_shots:
                st.write("**Tirs, buts et xG — Top 3**")
                if team_shots.empty:
                    st.info("Aucun tir enregistré pour cette période.")
                else:
                    player_stats_off = (
                        team_shots.groupby("player")
                        .agg(Tirs=("id", "count"), Buts=("is_goal", "sum"), xG=("shot_statsbomb_xg", "sum"))
                        .reset_index()
                        .sort_values(["Buts", "Tirs", "xG"], ascending=[False, False, False])
                        .head(3)
                        .rename(columns={"player": "Joueur"})
                    )
                    player_stats_off["xG"] = player_stats_off["xG"].round(2)
                    st.dataframe(player_stats_off, use_container_width=True, hide_index=True)

            with table_creation:
                st.write("**Passes décisives et passes clés — Top 3**")
                required_creation_cols = {"player", "is_assist", "is_key_pass"}
                if team_passes.empty:
                    st.info("Aucune passe enregistrée pour cette période.")
                elif not required_creation_cols.issubset(team_passes.columns):
                    st.warning("Relancez feature_engineering.py pour ajouter les assists et les passes clés.")
                else:
                    creation_stats = (
                        team_passes.assign(
                            Assist=team_passes["is_assist"].astype(int),
                            Passe_cle=team_passes["is_key_pass"].astype(int),
                        )
                        .groupby("player")
                        .agg(Assists=("Assist", "sum"),
                             **{"Passes clés": ("Passe_cle", "sum")},
                             Passes_tentées=("player", "size"))
                        .reset_index()
                        .sort_values(["Assists", "Passes clés", "Passes_tentées"], ascending=[False, False, False])
                        .head(3)
                        .rename(columns={"player": "Joueur"})
                    )
                    st.dataframe(creation_stats, use_container_width=True, hide_index=True)

            col_map, col_eff = st.columns(2)
            with col_map:
                st.write("### 🎯 Carte des tirs réels cumulés")
                pitch = VerticalPitch(half=True, pitch_type="statsbomb", pitch_color="#1e293b", line_color="#64748b")
                fig, ax = pitch.draw(figsize=(6, 5))
                shots_xy = team_shots.dropna(subset=["x", "y"])
                buts = shots_xy[shots_xy["is_goal"]]
                rates = shots_xy[~shots_xy["is_goal"]]
                if not rates.empty:
                    pitch.scatter(rates["x"], rates["y"], s=rates["shot_statsbomb_xg"] * 500,
                                  color="#60a5fa", alpha=0.5, ax=ax)
                if not buts.empty:
                    pitch.scatter(buts["x"], buts["y"], s=buts["shot_statsbomb_xg"] * 700,
                                  color="#f43f5e", marker="*", ax=ax)
                legend_handles = [
                    Line2D([0], [0], marker="*", color="none", markerfacecolor="#f43f5e",
                           markeredgecolor="#f43f5e", markersize=13, label="But"),
                    Line2D([0], [0], marker="o", color="none", markerfacecolor="#60a5fa",
                           markeredgecolor="#60a5fa", alpha=0.6, markersize=9, label="Tir sans but"),
                ]
                ax.legend(handles=legend_handles, loc="upper left", frameon=True,
                          facecolor="#1e293b", edgecolor="#64748b", labelcolor="white")
                st.pyplot(fig)
                plt.close(fig)

                # === CHANGEMENT 4 : %Tirs hors surface + classement Liga ===
                outside_pct_series = compute_league_shots_outside(df_shots, max_history_week)
                my_outside = outside_pct_series.get(team_selected, np.nan)
                help_outside = (
                    "Part des tirs réalisés en dehors de la surface adverse. "
                    "La surface est approximée par x ≥ 102 et 18 ≤ y ≤ 62 dans le repère StatsBomb. "
                    "Une valeur basse est préférable (meilleure sélection de tir)."
                )
                if pd.notna(my_outside):
                    my_rank_outside = int(outside_pct_series.rank(ascending=True, method="min")[team_selected])
                    st.metric("%Tirs en dehors de la surface", f"{my_outside:.1f} %", help=help_outside)
                    st.caption(f"🏅 {ordinal_fr(my_rank_outside)} sur {n_teams} en Liga")
                else:
                    st.metric("%Tirs en dehors de la surface", "—", help=help_outside)

            with col_eff:
                st.write("### ⚖️ Efficacité devant le but : buts vs xG cumulé")
                df_eff = ind[["xg_total", "buts"]].reset_index()
                df_eff["Groupe"] = df_eff["team"].apply(
                    lambda t: "Équipe sélectionnée" if t == team_selected else "Autres équipes"
                )
                df_eff["ordre"] = (df_eff["team"] == team_selected).astype(int)
                max_val = float(max(df_eff["xg_total"].max(), df_eff["buts"].max())) * 1.08
                axis_scale = alt.Scale(domain=[0, max_val])
                diagonale = (
                    alt.Chart(pd.DataFrame({"x": [0, max_val], "y": [0, max_val]}))
                    .mark_line(strokeDash=[6, 4], color="#94a3b8")
                    .encode(x=alt.X("x:Q", title="xG cumulé", scale=axis_scale),
                            y=alt.Y("y:Q", title="Buts marqués", scale=axis_scale))
                )
                bulles = (
                    alt.Chart(df_eff)
                    .mark_circle(opacity=0.85)
                    .encode(
                        x=alt.X("xg_total:Q", title="xG cumulé", scale=axis_scale),
                        y=alt.Y("buts:Q", title="Buts marqués", scale=axis_scale),
                        color=alt.Color(
                            "Groupe:N",
                            scale=alt.Scale(domain=["Autres équipes", "Équipe sélectionnée"],
                                            range=["#60a5fa", "#f43f5e"]),
                            legend=alt.Legend(orient="bottom", title=None),
                        ),
                        size=alt.Size("ordre:Q", scale=alt.Scale(domain=[0, 1], range=[150, 400]), legend=None),
                        order=alt.Order("ordre:Q"),
                        tooltip=[alt.Tooltip("team:N", title="Équipe"),
                                 alt.Tooltip("xg_total:Q", title="xG cumulé", format=".1f"),
                                 alt.Tooltip("buts:Q", title="Buts marqués")],
                    )
                )
                st.altair_chart((diagonale + bulles).properties(height=420), use_container_width=True)
                st.caption("Au-dessus de la ligne pointillée (x = y) : l'équipe marque plus que ce que ses tirs laissaient prévoir. En dessous : elle en marque moins.")

        # ==========================================
        # STATISTIQUES DÉFENSIVES (2e catégorie)
        # ==========================================
        with st.expander("🛡️ Statistiques défensives", expanded=True):

            st.write("### Solidité défensive")
            d1, d2, d3 = st.columns(3)
            with d1:
                indicator_bar("xG concédé / match", "xg_conc_match", ".2f",
                              "Qualité cumulée des occasions concédées à l'adversaire, rapportée à un match. Une valeur basse est préférable.")
            with d2:
                indicator_bar("Tirs concédés / match", "tirs_conc_match", ".1f",
                              "Nombre moyen de tirs accordés à l'adversaire par match. Un classement 2e signifie que l'équipe concède le deuxième plus petit volume de tirs.")
            with d3:
                indicator_bar("Buts encaissés (cumul)", "buts_enc", ".0f",
                              "Nombre total de buts encaissés sur la période. Un classement élevé correspond à moins de buts encaissés.")

            st.write("### 🚨 Pressing")
            pr1, pr2, pr3, pr4 = st.columns(4)
            with pr1:
                kpi("PPDA", "ppda", f"{me['ppda']:.2f}",
                    "Passes réussies de l'adversaire autorisées par action défensive dans la zone PPDA. Un PPDA bas indique un pressing plus intense ; un classement 2e signifie le deuxième PPDA le plus bas.")
            with pr2:
                kpi("Pressions / match", "pressions_match", f"{me['pressions_match']:.1f}",
                    "Nombre moyen de pressions réalisées par match. Une valeur élevée traduit davantage d'actions de pression, sans mesurer à elle seule leur réussite.")
            with pr3:
                kpi("Pressions dans le tiers adverse", "pressions_hautes_pct", f"{me['pressions_hautes_pct']:.1f} %",
                    "Part des pressions réalisées dans le tiers le plus proche du but adverse. Une valeur élevée indique un bloc de pressing plus haut.")
            with pr4:
                kpi("Contre-pressings / match", "contre_pressings_match", f"{me['contre_pressings_match']:.1f}",
                    "Nombre moyen de pressions portant le marqueur StatsBomb counterpress.")

            st.write("### 🔁 Récupérations et duels")
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                indicator_bar("% de récupérations dans le tiers adverse", "recup_hautes_pct", ".1f",
                              "Part des récupérations situées dans le tiers adverse.")
            with rc2:
                indicator_bar("% de tacles gagnés", "tacles_gagnes_pct", ".1f",
                              "Part des tacles remportés parmi tous les tacles disputés sur la période.")
            with rc3:
                indicator_bar("% de duels aériens gagnés", "duels_aeriens_gagnes_pct", ".1f",
                              "Part des duels aériens remportés parmi les duels aériens disputés.")

            st.write("### 🗺️ Zones de pressing et de récupération")
            pressure_xy = team_defense[team_defense["event_kind"] == "Pression"].dropna(subset=["x", "y"])
            recovery_xy = team_defense[team_defense["event_kind"] == "Récupération"].dropna(subset=["x", "y"])
            heat_press, heat_recov = st.columns(2)
            with heat_press:
                st.write("**Zones de pression**")
                if pressure_xy.empty:
                    st.info("Aucune pression enregistrée.")
                else:
                    # === CHANGEMENT 5 : flèche de sens d'attaque dans le graphique ===
                    render_zone_heatmap(pressure_xy["x"], pressure_xy["y"], CMAP_PRESS,
                                        figsize=(6.2, 4.5), show_attack_direction=True)
                st.caption("Sens de l'attaque : de gauche à droite. Répartition spatiale des pressions sur la période.")
            with heat_recov:
                st.write("**Zones de récupération**")
                if recovery_xy.empty:
                    st.info("Aucune récupération enregistrée.")
                else:
                    league_recovery_height = ind["hauteur_recup"].dropna().mean()
                    render_recovery_thirds_heatmap(
                        recovery_xy["x"], recovery_xy["y"],
                        me["hauteur_recup"], league_recovery_height,
                        figsize=(6.2, 4.5),
                    )
                st.caption("Répartition des récupérations dans les trois tiers. Les deux lignes verticales représentent des hauteurs moyennes de récupération : rose pour l'équipe sélectionnée et jaune en tirets pour la moyenne de la Liga.")

        # ==========================================
        # ANALYSE DES PASSES (3e catégorie)
        # ==========================================
        with st.expander("🔄 Analyse des passes", expanded=True):

            nb_pressure = int(team_history["nb_passes_pressure"].sum())
            nb_pressure_ok = int(team_history["nb_passes_pressure_ok"].sum())

            p1, p2, p3 = st.columns(3)
            with p1:
                indicator_bar("Passes / match", "passes_match", ".0f",
                              "Nombre moyen de passes tentées par match.")
            with p2:
                indicator_bar("Précision des passes", "precision_passes", ".1f",
                              "Part des passes réussies parmi les passes tentées.")
            with p3:
                indicator_bar("Précision sous pression", "precision_pression", ".1f",
                              f"{nb_pressure_ok} passes réussies sur {nb_pressure} passes effectuées sous pression.")

            st.write("### 🎯 Style de jeu : longueur et verticalité des passes")
            df_style = ind[["longueur_moy_passe", "passes_avant_pct"]].reset_index()
            df_style["Groupe"] = df_style["team"].apply(
                lambda t: "Équipe sélectionnée" if t == team_selected else "Autres équipes"
            )
            df_style["ordre"] = (df_style["team"] == team_selected).astype(int)

            mean_length = float(df_style["longueur_moy_passe"].mean())
            mean_forward = float(df_style["passes_avant_pct"].mean())
            x_min = float(df_style["longueur_moy_passe"].min())
            x_max = float(df_style["longueur_moy_passe"].max())
            y_min = float(df_style["passes_avant_pct"].min())
            y_max = float(df_style["passes_avant_pct"].max())
            x_margin = max((x_max - x_min) * 0.08, 0.5)
            y_margin = max((y_max - y_min) * 0.08, 1.0)
            x_scale = alt.Scale(domain=[x_min - x_margin, x_max + x_margin])
            y_scale = alt.Scale(domain=[max(0, y_min - y_margin), min(100, y_max + y_margin)])

            vertical_mean = (
                alt.Chart(pd.DataFrame({"x": [mean_length]}))
                .mark_rule(strokeDash=[6, 4], color="#fbbf24")
                .encode(x=alt.X("x:Q", scale=x_scale))
            )
            horizontal_mean = (
                alt.Chart(pd.DataFrame({"y": [mean_forward]}))
                .mark_rule(strokeDash=[6, 4], color="#fbbf24")
                .encode(y=alt.Y("y:Q", scale=y_scale))
            )
            bubbles = (
                alt.Chart(df_style)
                .mark_circle(opacity=0.85)
                .encode(
                    x=alt.X("longueur_moy_passe:Q", title="Longueur moyenne des passes (m)", scale=x_scale),
                    y=alt.Y("passes_avant_pct:Q", title="Proportion de passes vers l'avant (%)", scale=y_scale),
                    color=alt.Color(
                        "Groupe:N",
                        scale=alt.Scale(domain=["Autres équipes", "Équipe sélectionnée"],
                                        range=["#60a5fa", "#f43f5e"]),
                        legend=alt.Legend(orient="bottom", title=None),
                    ),
                    size=alt.Size("ordre:Q", scale=alt.Scale(domain=[0, 1], range=[150, 450]), legend=None),
                    order=alt.Order("ordre:Q"),
                    tooltip=[alt.Tooltip("team:N", title="Équipe"),
                             alt.Tooltip("longueur_moy_passe:Q", title="Longueur moyenne (m)", format=".1f"),
                             alt.Tooltip("passes_avant_pct:Q", title="Passes vers l'avant (%)", format=".1f")],
                )
            )

            # === CHANGEMENT 6 : labels de quadrants dans les coins, derrière les bulles, multi-lignes ===
            pad_x = (x_max - x_min) * 0.04
            pad_y = (y_max - y_min) * 0.04

            def _quadrant_label(x_val, y_val, text, align, baseline):
                return (
                    alt.Chart(pd.DataFrame({"x": [x_val], "y": [y_val], "label": [text]}))
                    .mark_text(color="#cbd5e1", fontSize=10, fontWeight="bold", opacity=0.85,
                               align=align, baseline=baseline)
                    .encode(
                        x=alt.X("x:Q", scale=x_scale),
                        y=alt.Y("y:Q", scale=y_scale),
                        text="label:N",
                    )
                )

            label_bl = _quadrant_label(x_min + pad_x, y_min + pad_y,
                                       "Jeu court\net conservateur", "left", "bottom")
            label_tl = _quadrant_label(x_min + pad_x, y_max - pad_y,
                                       "Jeu court\net progressif", "left", "top")
            label_br = _quadrant_label(x_max - pad_x, y_min + pad_y,
                                       "Jeu long de\nconservation ou\nde renversement", "right", "bottom")
            label_tr = _quadrant_label(x_max - pad_x, y_max - pad_y,
                                       "Jeu direct\net vertical", "right", "top")

            # Ordre de layer : labels en dessous des bulles
            st.altair_chart(
                (vertical_mean + horizontal_mean
                 + label_bl + label_tl + label_br + label_tr
                 + bubbles).properties(height=520),
                use_container_width=True,
            )
            st.caption(
                f"Les lignes indiquent les moyennes de la Liga : {mean_length:.1f} m pour la longueur des passes et "
                f"{mean_forward:.1f} % pour la proportion de passes vers l'avant."
            )

            if team_passes.empty:
                st.info("Aucune passe enregistrée pour cette période.")
            else:
                st.write("### 🕸️ Réseau de passes")
                top_players = team_passes["player"].value_counts().head(11).index
                tp = team_passes[team_passes["player"].isin(top_players)]
                nodes = (
                    tp.groupby("player")
                    .agg(x=("x", "mean"), y=("y", "mean"), nb_passes=("x", "size"))
                    .reset_index()
                )
                links = team_passes[
                    team_passes["is_complete"]
                    & team_passes["player"].isin(top_players)
                    & team_passes["pass_recipient"].isin(top_players)
                ].copy()
                pl = links["player"].astype(str)
                rc = links["pass_recipient"].astype(str)
                links["a"] = np.where(pl < rc, pl, rc)
                links["b"] = np.where(pl < rc, rc, pl)
                link_counts = links.groupby(["a", "b"]).size().reset_index(name="n")

                max_link = int(link_counts["n"].max()) if not link_counts.empty else 1
                slider_max = max(max_link, 2)
                default_min = min(max(1, round(max_link * 0.25)), slider_max)
                min_link = st.slider(
                    "Nombre minimum de passes entre deux joueurs pour afficher un lien",
                    min_value=1, max_value=slider_max, value=default_min,
                )

                col_net, col_tbl = st.columns([2, 1])
                with col_net:
                    shown = link_counts[link_counts["n"] >= min_link]
                    pitch_n = Pitch(pitch_type="statsbomb", pitch_color="#1e293b", line_color="#64748b", line_zorder=1)
                    fig_n, ax_n = pitch_n.draw(figsize=(8, 5.5))
                    pos = nodes.set_index("player")[["x", "y"]]
                    if not shown.empty:
                        widths = 1 + 8 * shown["n"] / shown["n"].max()
                        pitch_n.lines(
                            shown["a"].map(pos["x"]), shown["a"].map(pos["y"]),
                            shown["b"].map(pos["x"]), shown["b"].map(pos["y"]),
                            lw=widths.values, color="#60a5fa", alpha=0.55, zorder=2, ax=ax_n,
                        )
                    sizes = 250 + 1500 * nodes["nb_passes"] / nodes["nb_passes"].max()
                    pitch_n.scatter(nodes["x"], nodes["y"], s=sizes, color="#f43f5e", edgecolors="white",
                                    linewidth=1.5, alpha=0.95, zorder=3, ax=ax_n)
                    for r in nodes.itertuples():
                        ax_n.text(r.x, r.y, short_name(r.player), ha="center", va="center", color="white",
                                  fontsize=8, fontweight="bold", zorder=4,
                                  bbox=dict(boxstyle="round,pad=0.15", fc="#0f172a", ec="none", alpha=0.7))
                    st.pyplot(fig_n)
                    plt.close(fig_n)
                    st.caption(
                        "Les 11 joueurs ayant le plus de passes sur la période, placés à la position moyenne de leurs passes "
                        "(sens d'attaque de gauche à droite). Taille du cercle = nombre de passes, épaisseur du lien = passes "
                        "réussies échangées. C'est une vue moyenne, pas un onze réel."
                    )

                with col_tbl:
                    st.write("**Organisateurs du jeu — Top 3**")
                    tbl_organizers = (
                        team_passes.groupby("player")
                        .agg(Passes=("id", "count"), Precision=("is_complete", "mean"))
                        .reset_index()
                        .sort_values(["Passes", "Precision"], ascending=[False, False])
                        .head(3)
                    )
                    tbl_organizers["Precision"] = (tbl_organizers["Precision"] * 100).round(1)
                    tbl_organizers = tbl_organizers.rename(columns={"player": "Joueur", "Precision": "Précision (%)"})
                    st.dataframe(tbl_organizers, use_container_width=True, hide_index=True)

                    st.write("**Passes progressives — Top 3**")
                    if "is_progressive" not in team_passes.columns:
                        st.warning("Relancez feature_engineering.py pour ajouter les passes progressives.")
                    else:
                        tbl_progressive = (
                            team_passes.assign(Passe_progressive=team_passes["is_progressive"].astype(int))
                            .groupby("player")
                            .agg(**{"Passes progressives": ("Passe_progressive", "sum")},
                                 Passes=("id", "count"))
                            .reset_index()
                            .sort_values(["Passes progressives", "Passes"], ascending=[False, False])
                            .head(3)
                            .rename(columns={"player": "Joueur"})
                        )
                        st.dataframe(tbl_progressive, use_container_width=True, hide_index=True)

        # ==========================================
        # PERFORMANCE DE POSSESSION (4e catégorie)
        # ==========================================
        with st.expander("🎮 Performance de possession", expanded=True):

            total_poss_all = int(team_history["nb_possessions_total"].sum())
            poss_with_shot = int(team_history["nb_poss_with_shot"].sum())
            poss_final_third = int(team_history["nb_poss_final_third"].sum())

            s1, s2, s3, s4 = st.columns(4)
            with s1:
                indicator_bar("⏱️ Durée moyenne d'une possession", "duree_possession", ".1f",
                              "Durée moyenne d'une séquence de possession, en secondes.")
            with s2:
                indicator_bar("Pourcentage moyen de possession", "possession_pct", ".1f",
                              "Part moyenne du temps de jeu pendant laquelle l'équipe conserve le ballon.")
            with s3:
                indicator_bar("🥅 % possessions avec tir", "poss_tir_pct", ".1f",
                              f"{poss_with_shot} possessions contenant un tir sur {total_poss_all} possessions.")
            with s4:
                indicator_bar("🚩 % possessions atteignant le dernier tiers", "poss_dernier_tiers_pct", ".1f",
                              f"{poss_final_third} possessions atteignant le dernier tiers sur {total_poss_all} possessions.")

            st.write("### 🔻 Zones de pertes de balle")
            col_loss, col_loss_info = st.columns(2)
            losses_xy = team_losses.dropna(subset=["x", "y"])

            with col_loss:
                if losses_xy.empty:
                    st.info("Aucune perte de balle enregistrée pour cette période.")
                else:
                    pitch_l = Pitch(pitch_type="statsbomb", pitch_color="#1e293b", line_color="#64748b", line_zorder=2)
                    fig_l, ax_l = pitch_l.draw(figsize=(7, 5))
                    bs = pitch_l.bin_statistic(losses_xy["x"], losses_xy["y"], statistic="count", bins=(6, 5), normalize=True)
                    pitch_l.heatmap(bs, ax=ax_l, cmap=CMAP_LOSS, edgecolors="#1e293b", alpha=0.9)
                    pitch_l.label_heatmap(bs, ax=ax_l, str_format="{:.0%}", color="white", fontsize=11)
                    st.pyplot(fig_l)
                    plt.close(fig_l)
                    st.caption("Sens de l'attaque : de gauche à droite. Chaque case indique la part des pertes de balle de l'équipe.")

            with col_loss_info:
                if not losses_xy.empty:
                    st.write("**Répartition des pertes de balle**")
                    breakdown = losses_xy["loss_type"].value_counts().rename_axis("Type de perte").reset_index(name="Nombre")
                    breakdown["Par match"] = (breakdown["Nombre"] / n_matches).round(1)
                    st.dataframe(breakdown, use_container_width=True, hide_index=True)

                    # === CHANGEMENT 7 : %Pertes dans son propre tiers + classement Liga ===
                    losses_own_series = compute_league_losses_own_third(df_losses, max_history_week)
                    my_own = losses_own_series.get(team_selected, np.nan)
                    help_own = (
                        "Part des pertes de balle survenues dans le premier tiers du terrain (x < 40), "
                        "zone la plus dangereuse. Une valeur basse est préférable."
                    )
                    if pd.notna(my_own):
                        my_rank_own = int(losses_own_series.rank(ascending=True, method="min")[team_selected])
                        st.metric("%Pertes dans son propre tiers", f"{my_own:.1f} %", help=help_own)
                        st.caption(f"🏅 {ordinal_fr(my_rank_own)} sur {n_teams} en Liga")
                    else:
                        st.metric("%Pertes dans son propre tiers", "—", help=help_own)

# ==========================================
# ONGLET 2 : STATISTIQUES JOUEURS
# ==========================================
with tab_player:
    st.subheader("🎯 Profil individuel des joueurs")

    if max_history_week < 1:
        st.warning("⚠️ Sélectionnez la Journée 2 ou plus.")
    elif df_player_stats.empty or not set(PLAYER_SUM_COLS).issubset(df_player_stats.columns):
        st.info("Aucune statistique individuelle disponible. Relancez feature_engineering.py.")
    else:
        player_history = df_player_stats[
            (df_player_stats["match_week"] <= max_history_week)
            & (~df_player_stats["is_goalkeeper"].astype(bool))
        ]
        league_profiles = compute_player_profiles(player_history)

        team_names = player_history.loc[player_history["team"] == team_selected, "player"].unique()
        eligible_players = sorted(p for p in team_names if p in league_profiles.index)

        if not eligible_players:
            st.info(f"Aucun joueur de champ de {team_selected} n'a atteint {MIN_MINUTES} minutes jouées sur la période sélectionnée.")
        else:
            selected_player = st.selectbox(
                "Sélectionner un joueur",
                eligible_players,
                format_func=lambda p: f"{p} ({int(league_profiles.loc[p, 'minutes'])} min)",
            )
            metric_keys = [m[0] for m in PIZZA_METRICS]
            raw_values = league_profiles.loc[selected_player, metric_keys]
            best_values = league_profiles[metric_keys].max()
            scores = {
                k: float(min(100.0, raw_values[k] / best_values[k] * 100)) if best_values[k] > 0 else 0.0
                for k in metric_keys
            }
            _, col_pizza, _ = st.columns([1, 3, 1])
            with col_pizza:
                fig_pizza = render_pizza(
                    selected_player, team_selected, league_profiles.loc[selected_player, "minutes"], raw_values, scores
                )
                st.pyplot(fig_pizza)
                plt.close(fig_pizza)

            st.caption(
                f"Chaque indicateur est noté de 0 à 100 par rapport au meilleur joueur de champ de la Liga ayant joué "
                f"au moins {MIN_MINUTES} minutes (journées 1 à {max_history_week}) : 100 = meilleure valeur de la Liga. "
                f"Sous chaque libellé : la valeur réelle du joueur. {len(league_profiles)} joueurs admissibles dans la Liga."
            )