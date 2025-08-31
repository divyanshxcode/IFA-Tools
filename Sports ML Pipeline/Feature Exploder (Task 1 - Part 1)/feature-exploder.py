import io
import pandas as pd
import numpy as np
import streamlit as st

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Universal Sports Data Feature Engineer", layout="wide")
st.title("Universal Sports Feature Engineering")
st.caption(
    "Upload sports match data from any sport (football, cricket, basketball, esports, etc.), map columns in 4 steps, and transform each match into two team-rows with ML-ready features."
)

# ---------------------------
# Helpers
# ---------------------------

def try_convert_numeric(series):
    """Best-effort convert to numeric without failing the app."""
    return pd.to_numeric(series, errors="coerce")


def normalized_probs(odds_teamx, odds_teamy, odds_draw=None):
    """Return normalized probabilities from decimal odds (removes the overround)."""
    inv_teamx = 1.0 / odds_teamx if odds_teamx and odds_teamx > 0 else np.nan
    inv_teamy = 1.0 / odds_teamy if odds_teamy and odds_teamy > 0 else np.nan
    inv_draw = 1.0 / odds_draw if odds_draw and odds_draw > 0 else 0.0
    
    total = np.nansum([inv_teamx, inv_teamy, inv_draw])
    if total == 0 or np.isnan(total):
        return np.nan, np.nan, (np.nan if odds_draw is not None else None)
    
    return (inv_teamx / total, inv_teamy / total, 
            (inv_draw / total if odds_draw is not None else None))


def transform_matches_to_team_rows(
    df,
    *,
    # Step 1: Match Details
    match_id_col=None,
    date_col=None,
    league_col=None,
    teamx_col=None,
    teamy_col=None,
    match_shared_columns=None,
    
    # Step 2: Result Columns
    ftr_col=None,
    ftr_teamx_code="H",
    ftr_teamy_code="A", 
    ftr_draw_code="D",
    teamx_score_col=None,
    teamy_score_col=None,
    
    # Step 3: Match Statistics (Team-specific + Both)
    team_specific_columns=None,
    
    # Step 4: Pre-Match Info (Odds + Other)
    odds_teamx_col=None,
    odds_teamy_col=None,
    odds_draw_col=None,
    prematch_specific_columns=None,
):
    """
    Universal function: transform match-level data to team-level rows with engineered features.
    
    Args:
        df: Input DataFrame with match-level data
        team_specific_columns: Dict with keys 'TeamX', 'TeamY', 'Both' containing column lists
        prematch_specific_columns: Dict with keys 'TeamX', 'TeamY', 'Both' for pre-match data
        
    Returns: (long_df, info_dict)
    """
    work = df.copy()
    
    # Initialize column mappings
    team_specific_columns = team_specific_columns or {"TeamX": [], "TeamY": [], "Both": []}
    prematch_specific_columns = prematch_specific_columns or {"TeamX": [], "TeamY": [], "Both": []}
    match_shared_columns = match_shared_columns or []

    # PRESERVE ORIGINAL TEAM NAMES BEFORE ANY CONVERSIONS
    if teamx_col and teamx_col in work.columns:
        work['_original_teamx'] = work[teamx_col].astype(str)
    if teamy_col and teamy_col in work.columns:
        work['_original_teamy'] = work[teamy_col].astype(str)

    # Create a Match ID if missing
    if match_id_col and match_id_col in work.columns:
        match_id = work[match_id_col]
    else:
        match_id = pd.Series(range(1, len(work) + 1), index=work.index, name="MatchID")
        work["MatchID"] = match_id
        match_id_col = "MatchID"

    # Convert odds to numeric
    for c in [odds_teamx_col, odds_teamy_col, odds_draw_col]:
        if c and c in work.columns:
            work[c] = try_convert_numeric(work[c])

    # Convert scores to numeric
    for c in [teamx_score_col, teamy_score_col]:
        if c and c in work.columns:
            work[c] = try_convert_numeric(work[c])

    # Convert all team-specific numeric columns - BUT EXCLUDE TEAM NAME COLUMNS
    all_stat_columns = (team_specific_columns["TeamX"] + team_specific_columns["TeamY"] + 
                       team_specific_columns["Both"] + prematch_specific_columns["TeamX"] + 
                       prematch_specific_columns["TeamY"] + prematch_specific_columns["Both"])
    
    # Exclude team name columns from numeric conversion
    team_name_columns = set(filter(None, [teamx_col, teamy_col]))
    
    for col in all_stat_columns:
        if col in work.columns and col not in team_name_columns:
            work[col] = try_convert_numeric(work[col])

    # Build per-row records for Team_X and Team_Y perspectives
    records = []
    for idx, row in work.iterrows():
        mid = row[match_id_col]
        date_val = row[date_col] if date_col and date_col in work.columns else None
        league_val = row[league_col] if league_col and league_col in work.columns else None

        # USE PRESERVED ORIGINAL TEAM NAMES
        teamx = row['_original_teamx'] if '_original_teamx' in work.columns else (row[teamx_col] if teamx_col else None)
        teamy = row['_original_teamy'] if '_original_teamy' in work.columns else (row[teamy_col] if teamy_col else None)

        ftr_val = row[ftr_col] if ftr_col and ftr_col in work.columns else None
        ftr_str = str(ftr_val).strip() if (ftr_val is not None and not pd.isna(ftr_val)) else None

        teamx_score = row[teamx_score_col] if teamx_score_col and teamx_score_col in work.columns else None
        teamy_score = row[teamy_score_col] if teamy_score_col and teamy_score_col in work.columns else None

        ox = row[odds_teamx_col] if odds_teamx_col and odds_teamx_col in work.columns else None
        oy = row[odds_teamy_col] if odds_teamy_col and odds_teamy_col in work.columns else None
        od = row[odds_draw_col] if odds_draw_col and odds_draw_col in work.columns else None

        # Determine favorite/underdog based on lower odds
        fav_team = None
        udog_team = None
        if pd.notna(ox) and pd.notna(oy):
            if ox < oy:
                fav_team, udog_team = "TeamX", "TeamY"
            elif oy < ox:
                fav_team, udog_team = "TeamY", "TeamX"
            else:
                fav_team, udog_team = "Co-favorites", "Co-favorites"

        # Normalized probabilities (remove bookmaker overround)
        p_teamx, p_teamy, p_draw = normalized_probs(ox, oy, od if odds_draw_col else None)

        # Build two rows: Team_X perspective and Team_Y perspective
        for role, team, opp, odds_team, odds_opp, score_for, score_against in [
            ("TeamX", teamx, teamy, ox, oy, teamx_score, teamy_score),
            ("TeamY", teamy, teamx, oy, ox, teamy_score, teamx_score),
        ]:
            if team is None or opp is None:
                continue

            # Determine team result from team perspective
            team_result = None
            if ftr_str:
                if ftr_str == ftr_draw_code:
                    team_result = "Draw"
                elif (role == "TeamX" and ftr_str == ftr_teamx_code) or (role == "TeamY" and ftr_str == ftr_teamy_code):
                    team_result = "Win"
                else:
                    team_result = "Loss"

            # Set favorite flags
            is_favorite = None
            is_underdog = None
            if fav_team:
                if fav_team == "Co-favorites":
                    is_favorite = True
                    is_underdog = True
                else:
                    is_favorite = (role == fav_team)
                    is_underdog = (role == udog_team)

            # Team-level implied probabilities, normalized
            prob_team = p_teamx if role == "TeamX" else p_teamy
            prob_opp = p_teamy if role == "TeamX" else p_teamx

            # Base record
            rec = {
                "MatchID": mid,
                "Date": date_val,
                "League": league_val,
                "Team": team,
                "Role": role,
                "Opponent": opp,
                "FTR_raw": ftr_str,
                "Team_Result": team_result,
                "Team_Score": score_for,
                "Opponent_Score": score_against,
                "ScoreDiff": (score_for - score_against) if pd.notna(score_for) and pd.notna(score_against) else np.nan,
                "Team_Odds": odds_team,
                "Opponent_Odds": odds_opp,
                "Team_ImpliedProb": (1.0 / odds_team) if odds_team and odds_team > 0 else np.nan,
                "Opponent_ImpliedProb": (1.0 / odds_opp) if odds_opp and odds_opp > 0 else np.nan,
                "Team_NormProb": prob_team,
                "Opponent_NormProb": prob_opp,
                "FavoriteTeam": fav_team,
                "IsFavorite": is_favorite,
                "IsUnderdog": is_underdog,
            }
            
            if odds_draw_col:
                rec["Odds_Draw"] = od
                rec["NormProb_Draw"] = p_draw

            # Add match shared columns (same for both perspectives)
            for col in match_shared_columns:
                if col in work.columns:
                    rec[col] = row[col]

            # Add team-specific match statistics - KEEP ORIGINAL COLUMN NAMES, SWAP VALUES
            if role == "TeamX":
                # For TeamX row: use original values as-is
                for col in team_specific_columns["TeamX"]:
                    if col in work.columns:
                        rec[col] = row[col]
                for col in team_specific_columns["TeamY"]:
                    if col in work.columns:
                        rec[col] = row[col]
            else:
                # For TeamY row: SWAP the values between corresponding TeamX and TeamY columns
                # First, add all columns with original values
                for col in team_specific_columns["TeamX"]:
                    if col in work.columns:
                        rec[col] = row[col]
                for col in team_specific_columns["TeamY"]:
                    if col in work.columns:
                        rec[col] = row[col]
                
                # Then swap values between paired columns
                teamx_cols = team_specific_columns["TeamX"]
                teamy_cols = team_specific_columns["TeamY"]
                
                # Create pairs based on position or naming convention
                min_len = min(len(teamx_cols), len(teamy_cols))
                for i in range(min_len):
                    teamx_col = teamx_cols[i]
                    teamy_col = teamy_cols[i]
                    if teamx_col in work.columns and teamy_col in work.columns:
                        # Swap the values
                        rec[teamx_col] = row[teamy_col]
                        rec[teamy_col] = row[teamx_col]
            
            # Add shared match statistics (same for both roles)
            for col in team_specific_columns["Both"]:
                if col in work.columns:
                    rec[col] = row[col]

            # Add pre-match team-specific columns - KEEP ORIGINAL COLUMN NAMES, SWAP VALUES
            if role == "TeamX":
                # For TeamX row: use original values as-is
                for col in prematch_specific_columns["TeamX"]:
                    if col in work.columns:
                        rec[col] = row[col]
                for col in prematch_specific_columns["TeamY"]:
                    if col in work.columns:
                        rec[col] = row[col]
            else:
                # For TeamY row: SWAP the values between corresponding TeamX and TeamY columns
                # First, add all columns with original values
                for col in prematch_specific_columns["TeamX"]:
                    if col in work.columns:
                        rec[col] = row[col]
                for col in prematch_specific_columns["TeamY"]:
                    if col in work.columns:
                        rec[col] = row[col]
                
                # Then swap values between paired columns
                prematch_teamx_cols = prematch_specific_columns["TeamX"]
                prematch_teamy_cols = prematch_specific_columns["TeamY"]
                
                # Create pairs based on position
                min_len = min(len(prematch_teamx_cols), len(prematch_teamy_cols))
                for i in range(min_len):
                    teamx_col = prematch_teamx_cols[i]
                    teamy_col = prematch_teamy_cols[i]
                    if teamx_col in work.columns and teamy_col in work.columns:
                        # Swap the values
                        rec[teamx_col] = row[teamy_col]
                        rec[teamy_col] = row[teamx_col]
                        
            # Add shared pre-match columns (same for both roles)
            for col in prematch_specific_columns["Both"]:
                if col in work.columns:
                    rec[col] = row[col]

            records.append(rec)

    long_df = pd.DataFrame.from_records(records)

    info = {
        "n_matches": len(work),
        "n_team_rows": len(long_df),
        "has_draw_odds": bool(odds_draw_col),
        "teamx_stats_count": len(team_specific_columns["TeamX"]),
        "teamy_stats_count": len(team_specific_columns["TeamY"]),
        "shared_stats_count": len(team_specific_columns["Both"]) + len(match_shared_columns),
        "prematch_teamx_count": len(prematch_specific_columns["TeamX"]),
        "prematch_teamy_count": len(prematch_specific_columns["TeamY"]),
        "prematch_shared_count": len(prematch_specific_columns["Both"]),
    }
    return long_df, info


# ---------------------------
# Sidebar – 4-Step Column Mapping Pipeline
# ---------------------------
st.sidebar.header("📁 Upload your data")
file = st.sidebar.file_uploader("Excel (.xlsx/.xls) or CSV", type=["xlsx", "xls", "csv"]) 

sheet_name = None
raw_df = None
if file is not None:
    # Header row selection
    header_row = st.sidebar.number_input(
        "Header row (0-indexed)", 
        min_value=0, 
        max_value=10, 
        value=0,
        help="Which row contains the column names? Usually 0 (first row)"
    )
    
    if file.name.lower().endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(file)
        sheet_name = st.sidebar.selectbox("Sheet", options=xls.sheet_names, index=0)
        raw_df = xls.parse(sheet_name, header=header_row)
    else:
        # CSV
        sep = st.sidebar.selectbox("CSV delimiter", [",", ";", "\t", "|"], index=0)
        raw_df = pd.read_csv(file, sep=sep, header=header_row)

if raw_df is not None:
    st.subheader("Preview — Raw Data")
    st.dataframe(raw_df.head(25), use_container_width=True)

    cols = list(raw_df.columns)
    
    # ---------------------------
    # STEP 1: Match Details
    # ---------------------------
    st.sidebar.header("🏆 Step 1: Match Details")
    match_id_col = st.sidebar.selectbox("Match ID (optional)", options=[None] + cols, index=0)
    date_col = st.sidebar.selectbox("Date Column (optional)", options=[None] + cols, index=0)
    league_col = st.sidebar.selectbox("League/Division/Category (optional)", options=[None] + cols, index=0)
    
    teamx_col = st.sidebar.selectbox("Team/Player X (required)", options=cols)
    teamy_col = st.sidebar.selectbox("Team/Player Y (required)", options=cols)
    
    # Get already mapped columns for Step 1
    step1_mapped = set(filter(None, [match_id_col, date_col, league_col, teamx_col, teamy_col]))
    step1_remaining = [col for col in cols if col not in step1_mapped]
    
    match_shared_columns = []
    if step1_remaining:
        with st.sidebar.expander("➕ Additional Match Metadata (optional)"):
            match_shared_columns = st.sidebar.multiselect(
                "Shared match info (Referee, Venue, Weather, etc.)",
                options=step1_remaining,
                help="Columns that apply to the entire match (not team-specific)"
            )

    # ---------------------------
    # STEP 2: Match Result Columns
    # ---------------------------
    st.sidebar.header("📊 Step 2: Match Results")
    
    # Update remaining columns after Step 1
    step2_mapped = step1_mapped | set(match_shared_columns)
    step2_remaining = [col for col in cols if col not in step2_mapped]
    
    ftr_col = st.sidebar.selectbox("Result Column (optional)", options=[None] + step2_remaining, index=0)
    
    with st.sidebar.expander("Result Code Mapping"):
        ftr_teamx_code = st.text_input("Code for Team/Player X win", value="H")
        ftr_teamy_code = st.text_input("Code for Team/Player Y win", value="A")
        ftr_draw_code = st.text_input("Code for Draw", value="D")
    
    # Update remaining after result column
    step2_mapped = step2_mapped | set(filter(None, [ftr_col]))
    step2_remaining = [col for col in cols if col not in step2_mapped]
    
    teamx_score_col = st.sidebar.selectbox("Team/Player X Score (optional)", options=[None] + step2_remaining, index=0)
    # Update remaining after teamx score
    step2_mapped = step2_mapped | set(filter(None, [teamx_score_col]))
    step2_remaining = [col for col in cols if col not in step2_mapped]
    
    teamy_score_col = st.sidebar.selectbox("Team/Player Y Score (optional)", options=[None] + step2_remaining, index=0)

    # ---------------------------
    # STEP 3: Match Statistics (Post-Match)
    # ---------------------------
    st.sidebar.header("📈 Step 3: Match Statistics")
    
    # Update remaining columns after Step 2
    step3_mapped = step2_mapped | set(filter(None, [teamy_score_col]))
    step3_remaining = [col for col in cols if col not in step3_mapped]
    
    team_specific_columns = {"TeamX": [], "TeamY": [], "Both": []}
    
    if step3_remaining:
        with st.sidebar.expander("Team/Player X Statistics"):
            team_specific_columns["TeamX"] = st.sidebar.multiselect(
                "Select Team/Player X specific stats (shots, fouls, etc.)",
                options=step3_remaining,
                help="Stats that belong specifically to Team/Player X"
            )
        
        # Update remaining after TeamX selection
        step3_remaining_after_x = [col for col in step3_remaining if col not in team_specific_columns["TeamX"]]
        
        with st.sidebar.expander("Team/Player Y Statistics"):
            team_specific_columns["TeamY"] = st.sidebar.multiselect(
                "Select Team/Player Y specific stats (shots, fouls, etc.)",
                options=step3_remaining_after_x,
                help="Stats that belong specifically to Team/Player Y"
            )
            
        # Update remaining after TeamY selection
        step3_remaining_after_y = [col for col in step3_remaining_after_x if col not in team_specific_columns["TeamY"]]
        
        with st.sidebar.expander("Shared Match Statistics"):
            team_specific_columns["Both"] = st.sidebar.multiselect(
                "Select shared match stats (attendance, cards, etc.)",
                options=step3_remaining_after_y,
                help="Stats that apply to both teams or the entire match"
            )

    # ---------------------------
    # STEP 4: Pre-Match Info (Odds + Other)
    # ---------------------------
    st.sidebar.header("🎯 Step 4: Pre-Match Information")
    
    # Update remaining columns after Step 3
    step4_mapped = (step3_mapped | set(team_specific_columns["TeamX"]) | 
                   set(team_specific_columns["TeamY"]) | set(team_specific_columns["Both"]))
    step4_remaining = [col for col in cols if col not in step4_mapped]
    
    st.sidebar.subheader("Odds (Decimal Format)")
    odds_teamx_col = st.sidebar.selectbox("Team/Player X Odds (optional)", options=[None] + step4_remaining, index=0)
    
    # Update remaining after teamx odds
    step4_remaining = [col for col in step4_remaining if col != odds_teamx_col]
    odds_teamy_col = st.sidebar.selectbox("Team/Player Y Odds (optional)", options=[None] + step4_remaining, index=0)
    
    # Update remaining after teamy odds
    step4_remaining = [col for col in step4_remaining if col != odds_teamy_col]
    odds_draw_col = st.sidebar.selectbox("Draw Odds (optional)", options=[None] + step4_remaining, index=0)
    
    # Final remaining columns for other pre-match info
    step4_remaining = [col for col in step4_remaining if col != odds_draw_col]
    
    prematch_specific_columns = {"TeamX": [], "TeamY": [], "Both": []}
    
    if step4_remaining:
        with st.sidebar.expander("➕ Other Pre-Match Info"):
            st.sidebar.write("Map additional pre-match columns (ratings, power indices, etc.)")
            
            with st.sidebar.expander("Team/Player X Pre-Match"):
                prematch_specific_columns["TeamX"] = st.sidebar.multiselect(
                    "Team/Player X pre-match data",
                    options=step4_remaining,
                    help="Pre-match stats specific to Team/Player X"
                )
            
            step4_remaining_after_x = [col for col in step4_remaining if col not in prematch_specific_columns["TeamX"]]
            
            with st.sidebar.expander("Team/Player Y Pre-Match"):
                prematch_specific_columns["TeamY"] = st.sidebar.multiselect(
                    "Team/Player Y pre-match data",
                    options=step4_remaining_after_x,
                    help="Pre-match stats specific to Team/Player Y"
                )
                
            step4_remaining_after_y = [col for col in step4_remaining_after_x if col not in prematch_specific_columns["TeamY"]]
            
            with st.sidebar.expander("Shared Pre-Match"):
                prematch_specific_columns["Both"] = st.sidebar.multiselect(
                    "Shared pre-match data",
                    options=step4_remaining_after_y,
                    help="Pre-match data that applies to both teams"
                )

    # ---------------------------
    # Transform Button
    # ---------------------------
    st.sidebar.header("🚀 Transform")
    go = st.sidebar.button("Transform to Team-Level Rows", type="primary")

    if go:
        long_df, info = transform_matches_to_team_rows(
            raw_df,
            # Step 1
            match_id_col=match_id_col,
            date_col=date_col,
            league_col=league_col,
            teamx_col=teamx_col,
            teamy_col=teamy_col,
            match_shared_columns=match_shared_columns,
            
            # Step 2
            ftr_col=ftr_col,
            ftr_teamx_code=ftr_teamx_code,
            ftr_teamy_code=ftr_teamy_code,
            ftr_draw_code=ftr_draw_code,
            teamx_score_col=teamx_score_col,
            teamy_score_col=teamy_score_col,
            
            # Step 3
            team_specific_columns=team_specific_columns,
            
            # Step 4
            odds_teamx_col=odds_teamx_col,
            odds_teamy_col=odds_teamy_col,
            odds_draw_col=odds_draw_col,
            prematch_specific_columns=prematch_specific_columns,
        )

        st.success(
            f"✅ Transformed {info['n_matches']} matches into {info['n_team_rows']} team-level rows!"
        )

        # Construct feature column order for display
        feature_cols = [
            "MatchID", "Date", "League", "Team", "Role", "Opponent",
            "FTR_raw", "Team_Result", "Team_Score", "Opponent_Score", "ScoreDiff",
            "Team_Odds", "Opponent_Odds", "Team_ImpliedProb", "Opponent_ImpliedProb",
            "Team_NormProb", "Opponent_NormProb", "FavoriteTeam", "IsFavorite", "IsUnderdog",
        ]
        
        if odds_draw_col:
            feature_cols += ["Odds_Draw", "NormProb_Draw"]
            
        # Add match shared columns
        feature_cols.extend(match_shared_columns)
        
        # Add original stat columns (no Team_/Opponent_ prefixes)
        feature_cols.extend(team_specific_columns["TeamX"])
        feature_cols.extend(team_specific_columns["TeamY"])
        feature_cols.extend(team_specific_columns["Both"])
        
        # Add original pre-match columns (no Team_/Opponent_ prefixes)
        feature_cols.extend(prematch_specific_columns["TeamX"])
        feature_cols.extend(prematch_specific_columns["TeamY"])
        feature_cols.extend(prematch_specific_columns["Both"])

        # Remove duplicates while preserving order
        seen = set()
        feature_cols = [x for x in feature_cols if not (x in seen or seen.add(x))]
        
        # Display engineered dataset
        show_cols = [c for c in feature_cols if c in long_df.columns]
        st.subheader("🎯 Engineered Team-Level Dataset")
        st.dataframe(long_df[show_cols].head(50), use_container_width=True)

        # Summary metrics
        st.subheader("📊 Transformation Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Original Matches", info["n_matches"])
        with col2:
            st.metric("Team Rows Generated", info["n_team_rows"])
        with col3:
            st.metric("Has Draw Odds", "Yes" if info["has_draw_odds"] else "No")
        with col4:
            total_features = (info["teamx_stats_count"] + info["teamy_stats_count"] + 
                            info["shared_stats_count"] + info["prematch_teamx_count"] + 
                            info["prematch_teamy_count"] + info["prematch_shared_count"])
            st.metric("Additional Features", total_features)

        # Feature breakdown
        if any([info["teamx_stats_count"], info["teamy_stats_count"], info["shared_stats_count"]]):
            st.info(
                f"📈 **Match Stats:** {info['teamx_stats_count']} Team X, "
                f"{info['teamy_stats_count']} Team Y, {info['shared_stats_count']} Shared"
            )
            
        if any([info["prematch_teamx_count"], info["prematch_teamy_count"], info["prematch_shared_count"]]):
            st.info(
                f"🎯 **Pre-Match Info:** {info['prematch_teamx_count']} Team X, "
                f"{info['prematch_teamy_count']} Team Y, {info['prematch_shared_count']} Shared"
            )

        if "Team_Result" in long_df.columns:
            result_counts = long_df["Team_Result"].value_counts()
            st.info(f"🏆 **Results:** {result_counts.to_dict()}")

        # Download button
        csv_bytes = long_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Engineered Dataset (CSV)",
            data=csv_bytes,
            file_name="sports_team_level_data.csv",
            mime="text/csv",
        )

else:
    st.info(
        """
        🚀 **Welcome to Universal Sports Feature Engineering!**
        
        Upload your sports data (any sport) and follow the 4-step process:
        
        1. **Match Details** → Map teams, date, league, etc.
        2. **Match Results** → Map outcome codes and scores  
        3. **Match Statistics** → Map team-specific and shared stats
        4. **Pre-Match Info** → Map odds and other predictive features
        
        The tool will transform your match-level data into ML-ready team-level rows with engineered features like favorite/underdog flags, normalized probabilities, and more!
        """
    )


