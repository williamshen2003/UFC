"""
UFC Fight Analysis Module

This module contains classes and functions for processing and analyzing UFC fight data.
It handles data loading, preprocessing, feature engineering, and dataset preparation
for machine learning. Includes integrated data leakage verification.
"""

import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import os
from datetime import datetime
# Calculate Elo ratings (imported from Elo module)
from src.data_processing.features.Elo import calculate_elo_ratings
from src.data_processing.features.helper import DataUtils, OddsUtils, FighterUtils, DateUtils

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class FightDataProcessor:
    """Process and transform UFC fight data for analysis."""

    def __init__(self, data_dir: str = "../../../data", enable_verification: bool = True):
        """
        Initialize the processor with data directory.

        Args:
            data_dir: Directory containing data files
            enable_verification: Whether to enable leakage verification checks
        """
        module_dir = Path(__file__).resolve().parent
        repo_root = module_dir.parents[2]

        candidate_dir = Path(data_dir).expanduser()
        if not candidate_dir.is_absolute():
            module_relative = (module_dir / candidate_dir).resolve(strict=False)
            repo_relative = (repo_root / candidate_dir).resolve(strict=False)
            if module_relative.exists():
                candidate_dir = module_relative
            elif repo_relative.exists():
                candidate_dir = repo_relative
            else:
                candidate_dir = module_relative

        self.data_dir = candidate_dir
        self.utils = DataUtils()
        self.odds_utils = OddsUtils(data_dir=self.data_dir)
        self.fighter_utils = FighterUtils(enable_verification=enable_verification)
        self.enable_verification = enable_verification

    def _load_csv(self, filepath: str) -> pd.DataFrame:
        """Load a CSV file into a DataFrame."""
        # Handle forward/backward slashes
        filepath = Path(filepath.replace('/', os.sep))

        if not filepath.is_absolute():
            filepath = self.data_dir / filepath

        filepath = filepath.expanduser()
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found at {filepath}")

        return pd.read_csv(filepath)

    def _save_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """Save DataFrame to CSV file."""
        # Handle forward/backward slashes
        filepath = Path(filepath.replace('/', os.sep))

        if not filepath.is_absolute():
            filepath = self.data_dir / filepath

        filepath = filepath.expanduser()
        filepath.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(filepath, index=False)
        print(f"Saved to {filepath}")

    def combine_rounds_stats(self, file_path: str) -> pd.DataFrame:
        """
        Process round-level fight data into fighter career statistics.

        Args:
            file_path: Path to the UFC stats CSV file

        Returns:
            DataFrame with processed fighter statistics
        """
        print("Loading and preprocessing data...")
        ufc_stats = self._load_csv(file_path)
        fighter_stats = self._load_csv('raw/ufc_fighter_tott.csv')
        ufc_stats = self.utils.preprocess_data(ufc_stats, fighter_stats)

        # Get numeric columns for aggregation
        numeric_columns = self._get_numeric_columns(ufc_stats)

        print("Aggregating stats...")
        # Get maximum round information
        max_round_data = ufc_stats.groupby('id').agg({
            'last_round': 'max',
            'time': 'max'
        }).reset_index()

        # Aggregate numeric stats by fighter and fight
        aggregated_stats = ufc_stats.groupby(['id', 'fighter'])[numeric_columns].sum().reset_index()

        # Calculate basic rates
        aggregated_stats = self._calculate_basic_rates(aggregated_stats)

        # Get non-numeric data
        non_numeric_data = self._extract_non_numeric_data(ufc_stats)

        print("Merging aggregated stats with non-numeric data...")
        # Merge all components
        merged_stats = pd.merge(aggregated_stats, non_numeric_data, on=['id', 'fighter'], how='left')
        merged_stats = pd.merge(merged_stats, max_round_data, on='id', how='left')

        print("Calculating career stats...")
        # Calculate career-level statistics
        final_stats = merged_stats.groupby('fighter', group_keys=False).apply(
            lambda x: self.fighter_utils.aggregate_fighter_stats(x, numeric_columns)
        )

        # Calculate per-minute stats
        final_stats = self._calculate_per_minute_stats(final_stats)

        # Calculate additional rates
        final_stats = self._calculate_additional_rates(final_stats)

        # Filter and process data
        final_stats = self._filter_unwanted_results(final_stats)
        final_stats = self._factorize_categorical_columns(final_stats)

        # Process odds data
        final_stats = self.odds_utils.process_odds_data(final_stats)

        # Clean up columns
        columns_to_drop = ['new_Open', 'new_Closing Range Start', 'new_Closing Range End', 'new_Movement', 'dob']
        final_stats = final_stats.drop(columns=columns_to_drop, errors='ignore')

        # Remove duplicate columns
        duplicate_columns = final_stats.columns[final_stats.columns.duplicated()]
        final_stats = final_stats.loc[:, ~final_stats.columns.duplicated()]
        if len(duplicate_columns) > 0:
            print(f"Dropped duplicate columns: {list(duplicate_columns)}")

        print("Calculating additional stats...")
        # Sort by fighter and date
        final_stats = final_stats.sort_values(['fighter', 'fight_date'])

        # Calculate experience, streaks, and time-based stats
        final_stats = final_stats.groupby('fighter', group_keys=False).apply(
            self.fighter_utils.calculate_experience_and_days
        )
        final_stats = final_stats.groupby('fighter', group_keys=False).apply(
            self.fighter_utils.update_streaks
        )
        final_stats['days_since_last_fight'] = final_stats['days_since_last_fight'].fillna(0)

        print("Calculating takedowns and knockdowns per 15 minutes...")
        final_stats = self.fighter_utils.calculate_time_based_stats(final_stats)

        print("Calculating total fights, wins, and losses...")
        final_stats = final_stats.groupby('fighter', group_keys=False).apply(
            self.fighter_utils.calculate_total_fight_stats
        )

        # Print verification summary if enabled
        if self.enable_verification:
            self.fighter_utils.print_verification_summary()

        print("Saving processed data...")
        self._save_csv(final_stats, 'processed/combined_rounds.csv')

        return final_stats

    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Extract relevant numeric columns for aggregation."""
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col not in ['id', 'last_round', 'age']]
        if 'time' not in numeric_columns:
            numeric_columns.append('time')
        return numeric_columns

    def _calculate_basic_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic strike and takedown rates."""
        df['significant_strikes_rate'] = self.utils.safe_divide(
            df['significant_strikes_landed'],
            df['significant_strikes_attempted']
        )
        df['takedown_rate'] = self.utils.safe_divide(
            df['takedown_successful'],
            df['takedown_attempted']
        )
        return df

    def _extract_non_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract non-numeric columns from the DataFrame."""
        non_numeric_columns = df.select_dtypes(exclude=['int64', 'float64']).columns.difference(
            ['id', 'fighter']
        )
        return df.drop_duplicates(subset=['id', 'fighter'])[
            ['id', 'fighter', 'age'] + list(non_numeric_columns)
            ]

    def _calculate_per_minute_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate per-minute statistics."""
        df['fight_duration_minutes'] = df['time'] / 60
        for col in ['significant_strikes_landed', 'significant_strikes_attempted',
                    'total_strikes_landed', 'total_strikes_attempted']:
            df[f'{col}_per_min'] = self.utils.safe_divide(df[col], df['fight_duration_minutes'])
        return df

    def _calculate_additional_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional rate statistics."""
        df["total_strikes_rate"] = self.utils.safe_divide(
            df["total_strikes_landed"],
            df["total_strikes_attempted"]
        )
        df["combined_success_rate"] = (df["takedown_rate"] + df["total_strikes_rate"]) / 2
        return df

    def _filter_unwanted_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out unwanted fight results."""
        df = df[~df['winner'].isin(['NC/NC', 'D/D'])]
        df = df[~df['result'].isin(['DQ', 'DQ ', 'Could Not Continue ', 'Overturned ', 'Other '])]
        return df

    def _factorize_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical columns to numeric codes and print the mapping."""
        for column in ['result', 'winner', 'scheduled_rounds']:
            df[column], unique = pd.factorize(df[column])
            mapping = {index: label for index, label in enumerate(unique)}
            print(f"Mapping for {column}: {mapping}")
        return df

    def combine_fighters_stats(self, file_path: str) -> pd.DataFrame:
        """
        Create pairwise fighter statistics for all fights.

        Args:
            file_path: Path to the combined rounds CSV file

        Returns:
            DataFrame with paired fighter statistics
        """
        df = self._load_csv(file_path)

        # Remove event columns and sort
        df = df.drop(columns=[col for col in df.columns if 'event' in col.lower()])
        df = df.sort_values(by=['id', 'fighter'])

        # Create mirrored fight pairs
        fights_dict = {}
        for _, row in df.iterrows():
            fight_id = row['id']
            fights_dict.setdefault(fight_id, []).append(row)

        # Combine original and mirrored rows
        combined_fights = []
        skipped_fights = 0

        for fight_id, fighters in fights_dict.items():
            if len(fighters) == 2:
                fighter_1, fighter_2 = fighters
                # Original pairing (fighter 1 vs fighter 2)
                original = pd.concat([pd.Series(fighter_1), pd.Series(fighter_2).add_suffix('_b')])
                # Mirrored pairing (fighter 2 vs fighter 1)
                mirrored = pd.concat([pd.Series(fighter_2), pd.Series(fighter_1).add_suffix('_b')])
                combined_fights.extend([original, mirrored])
            else:
                skipped_fights += 1

        if skipped_fights > 0:
            print(f"Skipped {skipped_fights} fights with missing fighter data")

        # Create and process combined DataFrame
        final_combined_df = pd.DataFrame(combined_fights).reset_index(drop=True)

        # Define columns for processing
        final_combined_df = self._calculate_differential_and_ratio_features(final_combined_df)

        # Filter and sort
        final_combined_df = final_combined_df[~final_combined_df['winner'].isin(['NC', 'D'])]
        final_combined_df['fight_date'] = pd.to_datetime(final_combined_df['fight_date'])
        final_combined_df = final_combined_df.sort_values(
            by=['fighter', 'fight_date'],
            ascending=[True, True]
        )

        # Save the result
        self._save_csv(final_combined_df, 'processed/combined_sorted_fighter_stats.csv')

        return final_combined_df

    def _calculate_differential_and_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate differential and ratio features between fighter pairs."""
        # Define columns to process
        base_columns = [
            'knockdowns', 'significant_strikes_landed', 'significant_strikes_attempted',
            'significant_strikes_rate', 'total_strikes_landed', 'total_strikes_attempted',
            'takedown_successful', 'takedown_attempted', 'takedown_rate', 'submission_attempt',
            'reversals', 'head_landed', 'head_attempted', 'body_landed', 'body_attempted',
            'leg_landed', 'leg_attempted', 'distance_landed', 'distance_attempted',
            'clinch_landed', 'clinch_attempted', 'ground_landed', 'ground_attempted'
        ]
        other_columns = [
            'open_odds', 'closing_range_start', 'closing_range_end', 'pre_fight_elo',
            'years_of_experience', 'win_streak', 'loss_streak', 'days_since_last_fight',
            'significant_strikes_landed_per_min', 'significant_strikes_attempted_per_min',
            'total_strikes_landed_per_min', 'total_strikes_attempted_per_min', 'takedowns_per_15min',
            'knockdowns_per_15min', 'total_fights', 'total_wins', 'total_losses',
            'wins_by_ko', 'losses_by_ko', 'wins_by_submission', 'losses_by_submission', 'wins_by_decision',
            'losses_by_decision', 'win_rate_by_ko', 'loss_rate_by_ko', 'win_rate_by_submission',
            'loss_rate_by_submission', 'win_rate_by_decision', 'loss_rate_by_decision'
        ]
        columns_to_process = (
                base_columns +
                [f"{col}_career" for col in base_columns] +
                [f"{col}_career_avg" for col in base_columns] +
                other_columns
        )

        # Calculate differential features
        diff_features = {}
        for col in columns_to_process:
            if col in df.columns and f"{col}_b" in df.columns:
                diff_features[f"{col}_diff"] = df[col] - df[f"{col}_b"]

        # Calculate ratio features
        ratio_features = {}
        for col in columns_to_process:
            if col in df.columns and f"{col}_b" in df.columns:
                ratio_features[f"{col}_ratio"] = self.utils.safe_divide(df[col], df[f"{col}_b"])

        # Combine all features
        return pd.concat([df, pd.DataFrame(diff_features), pd.DataFrame(ratio_features)], axis=1)


class MatchupProcessor:
    """Process and prepare matchup data for predictive modeling."""

    def __init__(self, data_dir: str = "../../../data", enable_verification: bool = True):
        """
        Initialize the processor with data directory.

        Args:
            data_dir: Directory containing data files
            enable_verification: Whether to enable leakage verification checks
        """
        self.fight_processor = FightDataProcessor(data_dir, enable_verification=enable_verification)
        self.data_dir = self.fight_processor.data_dir
        self.utils = DataUtils()
        self.odds_utils = OddsUtils(data_dir=self.data_dir)
        self.enable_verification = enable_verification
        self.leakage_warnings = []

    def create_matchup_data(self, file_path: str, tester: int, include_names: bool = False) -> pd.DataFrame:
        """
        Create matchup data for predictive modeling.

        Args:
            file_path: Path to the fighter stats CSV
            tester: Determines the number of most recent fights to use
            include_names: Whether to include fighter names in output

        Returns:
            DataFrame with matchup features
        """
        print(f"Creating matchup data with {tester} recent fights...")
        df = self.fight_processor._load_csv(file_path)
        n_past_fights = 6 - tester

        # Define columns to exclude from features
        columns_to_exclude = [
            'fighter', 'id', 'fighter_b', 'fight_date', 'fight_date_b',
            'result', 'winner', 'weight_class', 'scheduled_rounds',
            'result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b'
        ]

        # Define features to include
        features_to_include = [
            col for col in df.columns if col not in columns_to_exclude and
                                         col != 'age' and not col.endswith('_age')
        ]

        # Method columns (target variables)
        method_columns = ['winner']

        # Process matchups
        matchup_data = self._process_matchups(
            df, features_to_include, method_columns, n_past_fights, tester, include_names
        )

        # Create DataFrame
        column_names = self._generate_column_names(
            features_to_include, method_columns, n_past_fights, tester, include_names
        )
        matchup_df = pd.DataFrame(matchup_data, columns=column_names)

        # Drop fight_date column if present
        matchup_df = matchup_df.drop(columns=['fight_date'], errors='ignore')

        # Standardize column names
        matchup_df.columns = [self.utils.rename_columns_general(col) for col in matchup_df.columns]

        # Calculate additional differential and ratio columns
        matchup_df = self._calculate_matchup_features(matchup_df, features_to_include, n_past_fights)

        # Print leakage summary if enabled
        if self.enable_verification and self.leakage_warnings:
            print("\n" + "="*60)
            print("MATCHUP DATA LEAKAGE WARNINGS")
            print("="*60)
            for warning in self.leakage_warnings[:10]:  # Show first 10 warnings
                print(warning)
            if len(self.leakage_warnings) > 10:
                print(f"... and {len(self.leakage_warnings) - 10} more warnings")
            print("="*60)

        # Save output
        output_filename = f'matchup data/matchup_data_{n_past_fights}_avg{"_name" if include_names else ""}.csv'
        self.fight_processor._save_csv(matchup_df, output_filename)

        return matchup_df

    def _process_matchups(
            self,
            df: pd.DataFrame,
            features_to_include: List[str],
            method_columns: List[str],
            n_past_fights: int,
            tester: int,
            include_names: bool
    ) -> List[List]:
        """Process each fight to create matchup feature vectors with support for fighters with fewer fights."""
        matchup_data = []
        skipped_count = 0
        processed_count = 0
        partial_data_count = 0

        # ========== LEAKAGE CHECK #4: Setup ==========
        verification_sample_size = 5
        verification_counter = 0
        # ========================================

        # Process each current fight
        for idx, current_fight in df.iterrows():
            fighter_a_name = current_fight['fighter']
            fighter_b_name = current_fight['fighter_b']

            # Get past fights for each fighter
            fighter_a_df = df[
                (df['fighter'] == fighter_a_name) &
                (df['fight_date'] < current_fight['fight_date'])
                ].sort_values(by='fight_date', ascending=False).head(n_past_fights)

            fighter_b_df = df[
                (df['fighter'] == fighter_b_name) &
                (df['fight_date'] < current_fight['fight_date'])
                ].sort_values(by='fight_date', ascending=False).head(n_past_fights)

            # ========== LEAKAGE CHECK #4: Matchup Data Creation ==========
            if self.enable_verification and verification_counter < verification_sample_size:
                # Check Fighter A's data
                if len(fighter_a_df) > 0:
                    latest_past_fight_a = fighter_a_df.iloc[0]

                    # Critical check: Is this date BEFORE current fight?
                    if latest_past_fight_a['fight_date'] >= current_fight['fight_date']:
                        warning = f"CRITICAL LEAKAGE: Fighter {fighter_a_name} using future fight data!"
                        self.leakage_warnings.append(warning)
                        print(warning)

                    # Check if career stats make sense
                    all_fighter_a_fights = df[df['fighter'] == fighter_a_name].sort_values('fight_date')
                    actual_fight_number = len(all_fighter_a_fights[
                        all_fighter_a_fights['fight_date'] <= latest_past_fight_a['fight_date']
                    ])

                    if 'total_fights' in latest_past_fight_a:
                        if latest_past_fight_a['total_fights'] > actual_fight_number:
                            warning = (f"LEAKAGE: {fighter_a_name} has total_fights="
                                     f"{latest_past_fight_a['total_fights']} but only {actual_fight_number} "
                                     f"fights up to {latest_past_fight_a['fight_date']}")
                            self.leakage_warnings.append(warning)

                # Similar check for Fighter B
                if len(fighter_b_df) > 0:
                    latest_past_fight_b = fighter_b_df.iloc[0]

                    if latest_past_fight_b['fight_date'] >= current_fight['fight_date']:
                        warning = f"CRITICAL LEAKAGE: Fighter {fighter_b_name} using future fight data!"
                        self.leakage_warnings.append(warning)
                        print(warning)

                verification_counter += 1
            # ========== END LEAKAGE CHECK #4 ==========

            # Skip if either fighter has no past fights
            if len(fighter_a_df) == 0 or len(fighter_b_df) == 0:
                skipped_count += 1
                continue

            # Flag if we have partial data (at least one fighter with fewer than n_past_fights)
            has_partial_data = len(fighter_a_df) < n_past_fights or len(fighter_b_df) < n_past_fights
            if has_partial_data:
                partial_data_count += 1

            # Extract features from available past fights
            fighter_a_features = fighter_a_df[features_to_include].mean().values
            fighter_b_features = fighter_b_df[features_to_include].mean().values

            # Extract recent fight results
            # Only extract the available fight results, up to tester number
            num_a_results = min(len(fighter_a_df), tester)
            num_b_results = min(len(fighter_b_df), tester)

            results_fighter_a = fighter_a_df[['result', 'winner', 'weight_class', 'scheduled_rounds']].head(
                num_a_results).values.flatten() if num_a_results > 0 else np.array([])

            results_fighter_b = fighter_b_df[['result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']].head(
                num_b_results).values.flatten() if num_b_results > 0 else np.array([])

            # Pad results with None values to ensure consistent length
            results_fighter_a = np.pad(
                results_fighter_a,
                (0, tester * 4 - len(results_fighter_a)),
                'constant',
                constant_values=np.nan
            )
            results_fighter_b = np.pad(
                results_fighter_b,
                (0, tester * 4 - len(results_fighter_b)),
                'constant',
                constant_values=np.nan
            )

            # Get target labels
            labels = current_fight[method_columns].values

            # Process odds and age data
            current_fight_odds, current_fight_odds_diff, current_fight_odds_ratio = self._process_fight_odds(
                current_fight['open_odds'], current_fight['open_odds_b']
            )

            current_fight_closing_odds, current_fight_closing_odds_diff, current_fight_closing_odds_ratio = self._process_fight_odds(
                current_fight['closing_range_end'], current_fight['closing_range_end_b']
            )

            # Calculate the difference between closing and opening odds for each fighter
            current_fight_closing_open_diff_a = current_fight['closing_range_end'] - current_fight['open_odds']
            current_fight_closing_open_diff_b = current_fight['closing_range_end_b'] - current_fight['open_odds_b']

            current_fight_ages = [current_fight['age'], current_fight['age_b']]
            current_fight_age_diff = current_fight['age'] - current_fight['age_b']
            current_fight_age_ratio = self.utils.safe_divide(current_fight['age'], current_fight['age_b'])

            # Process Elo and other stats
            elo_stats, elo_ratio = self._process_elo_stats(current_fight)
            other_stats = self._process_other_stats(current_fight)

            # Combine all features
            combined_features = np.concatenate([
                fighter_a_features, fighter_b_features, results_fighter_a, results_fighter_b,
                current_fight_odds, [current_fight_odds_diff, current_fight_odds_ratio],
                current_fight_closing_odds, [current_fight_closing_odds_diff, current_fight_closing_odds_ratio,
                                             current_fight_closing_open_diff_a, current_fight_closing_open_diff_b],
                current_fight_ages, [current_fight_age_diff, current_fight_age_ratio],
                elo_stats, [elo_ratio], other_stats
            ])
            combined_row = np.concatenate([combined_features, labels])

            # Get most recent date and current fight date
            most_recent_date_a = fighter_a_df['fight_date'].max() if len(fighter_a_df) > 0 else None
            most_recent_date_b = fighter_b_df['fight_date'].max() if len(fighter_b_df) > 0 else None
            most_recent_date = max(most_recent_date_a,
                                   most_recent_date_b) if most_recent_date_a and most_recent_date_b else most_recent_date_a or most_recent_date_b
            current_fight_date = current_fight['fight_date']

            # Add to matchup data
            if not include_names:
                matchup_data.append([most_recent_date] + combined_row.tolist() + [current_fight_date])
            else:
                matchup_data.append(
                    [fighter_a_name, fighter_b_name, most_recent_date] + combined_row.tolist() + [current_fight_date]
                )

            processed_count += 1

        print(f"Processed {processed_count} matchups (including {partial_data_count} with partial fight history)")
        print(f"Skipped {skipped_count} matchups where at least one fighter had no previous fights")

        return matchup_data

    def _process_fight_odds(self, odds_a: float, odds_b: float) -> Tuple[List[float], float, float]:
        """Process betting odds for a fight."""
        return self.odds_utils.process_odds_pair(odds_a, odds_b)

    def _process_elo_stats(self, current_fight: pd.Series) -> Tuple[List[float], float]:
        """Process Elo rating statistics."""
        elo_a = current_fight['pre_fight_elo']
        elo_b = current_fight['pre_fight_elo_b']
        elo_diff = current_fight['pre_fight_elo_diff']

        # Calculate win probabilities based on Elo
        a_win_prob = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        b_win_prob = 1 / (1 + 10 ** ((elo_a - elo_b) / 400))

        elo_stats = [elo_a, elo_b, elo_diff, a_win_prob, b_win_prob]
        elo_ratio = self.utils.safe_divide(elo_a, elo_b)

        return elo_stats, elo_ratio

    def _process_other_stats(self, current_fight: pd.Series) -> List[float]:
        """Process other fighter statistics."""
        # Win/loss streak stats
        win_streak_a = current_fight['win_streak']
        win_streak_b = current_fight['win_streak_b']
        win_streak_diff = win_streak_a - win_streak_b
        win_streak_ratio = self.utils.safe_divide(win_streak_a, win_streak_b)

        loss_streak_a = current_fight['loss_streak']
        loss_streak_b = current_fight['loss_streak_b']
        loss_streak_diff = loss_streak_a - loss_streak_b
        loss_streak_ratio = self.utils.safe_divide(loss_streak_a, loss_streak_b)

        # Experience stats
        exp_a = current_fight['years_of_experience']
        exp_b = current_fight['years_of_experience_b']
        exp_diff = exp_a - exp_b
        exp_ratio = self.utils.safe_divide(exp_a, exp_b)

        # Last fight stats
        days_since_a = current_fight['days_since_last_fight']
        days_since_b = current_fight['days_since_last_fight_b']
        days_since_diff = days_since_a - days_since_b
        days_since_ratio = self.utils.safe_divide(days_since_a, days_since_b)

        return [
            win_streak_a, win_streak_b, win_streak_diff, win_streak_ratio,
            loss_streak_a, loss_streak_b, loss_streak_diff, loss_streak_ratio,
            exp_a, exp_b, exp_diff, exp_ratio,
            days_since_a, days_since_b, days_since_diff, days_since_ratio
        ]

    def _generate_column_names(
            self,
            features_to_include: List[str],
            method_columns: List[str],
            n_past_fights: int,
            tester: int,
            include_names: bool
    ) -> List[str]:
        """Generate column names for the matchup DataFrame."""
        # Results columns
        results_columns = []
        for i in range(1, tester + 1):
            results_columns += [
                f"result_fight_{i}", f"winner_fight_{i}", f"weight_class_fight_{i}", f"scheduled_rounds_fight_{i}",
                f"result_b_fight_{i}", f"winner_b_fight_{i}", f"weight_class_b_fight_{i}",
                f"scheduled_rounds_b_fight_{i}"
            ]

        # New feature columns
        new_columns = [
            'current_fight_pre_fight_elo_a', 'current_fight_pre_fight_elo_b', 'current_fight_pre_fight_elo_diff',
            'current_fight_pre_fight_elo_a_win_chance', 'current_fight_pre_fight_elo_b_win_chance',
            'current_fight_pre_fight_elo_ratio', 'current_fight_win_streak_a', 'current_fight_win_streak_b',
            'current_fight_win_streak_diff', 'current_fight_win_streak_ratio', 'current_fight_loss_streak_a',
            'current_fight_loss_streak_b', 'current_fight_loss_streak_diff', 'current_fight_loss_streak_ratio',
            'current_fight_years_experience_a', 'current_fight_years_experience_b',
            'current_fight_years_experience_diff',
            'current_fight_years_experience_ratio', 'current_fight_days_since_last_a',
            'current_fight_days_since_last_b', 'current_fight_days_since_last_diff',
            'current_fight_days_since_last_ratio'
        ]

        # Base columns
        base_columns = ['fight_date'] if not include_names else ['fighter_a', 'fighter_b', 'fight_date']

        # Feature columns
        feature_columns = (
                [f"{feature}_fighter_avg_last_{n_past_fights}" for feature in features_to_include] +
                [f"{feature}_fighter_b_avg_last_{n_past_fights}" for feature in features_to_include]
        )

        # Odds and age columns
        odds_age_columns = [
            'current_fight_open_odds', 'current_fight_open_odds_b', 'current_fight_open_odds_diff',
            'current_fight_open_odds_ratio',
            'current_fight_closing_odds', 'current_fight_closing_odds_b', 'current_fight_closing_odds_diff',
            'current_fight_closing_odds_ratio', 'current_fight_closing_open_diff_a',
            'current_fight_closing_open_diff_b',
            'current_fight_age', 'current_fight_age_b', 'current_fight_age_diff', 'current_fight_age_ratio'
        ]

        # Combine all column names
        return (
                base_columns + feature_columns + results_columns + odds_age_columns + new_columns +
                [f"{method}" for method in method_columns] + ['current_fight_date']
        )

    def _calculate_matchup_features(
            self,
            df: pd.DataFrame,
            features_to_include: List[str],
            n_past_fights: int
    ) -> pd.DataFrame:
        """Calculate additional differential and ratio features."""
        diff_columns = {}
        ratio_columns = {}

        for feature in features_to_include:
            col_a = f"{feature}_fighter_a_avg_last_{n_past_fights}"
            col_b = f"{feature}_fighter_b_avg_last_{n_past_fights}"

            if col_a in df.columns and col_b in df.columns:
                diff_columns[f"matchup_{feature}_diff_avg_last_{n_past_fights}"] = df[col_a] - df[col_b]
                ratio_columns[f"matchup_{feature}_ratio_avg_last_{n_past_fights}"] = self.utils.safe_divide(
                    df[col_a], df[col_b]
                )

        return pd.concat([df, pd.DataFrame(diff_columns), pd.DataFrame(ratio_columns)], axis=1)

    def split_train_val_test(
            self,
            matchup_data_file: str,
            start_date: str,
            end_date: str,
            years_back: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split matchup data into training, validation, and test sets with random fighter ordering.
        FIXED: Ensures no date overlap between splits.
        """
        print(f"Splitting data from {start_date} to {end_date} with {years_back} years history...")
        matchup_df = self.fight_processor._load_csv(matchup_data_file)

        # Convert dates
        matchup_df['current_fight_date'] = pd.to_datetime(matchup_df['current_fight_date'])
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        years_before = start_date - pd.DateOffset(years=years_back)

        # Split data - first extract test set
        test_data = matchup_df[
            (matchup_df['current_fight_date'] >= start_date) &
            (matchup_df['current_fight_date'] <= end_date)
            ].copy()

        remaining_data = matchup_df[
            (matchup_df['current_fight_date'] >= years_before) &
            (matchup_df['current_fight_date'] < start_date)
            ].copy()

        # Sort remaining data by date
        remaining_data = remaining_data.sort_values(by='current_fight_date', ascending=True)

        # ========== FIX: Ensure no date overlap between train and val ==========
        # Get unique dates in remaining data
        unique_dates = sorted(remaining_data['current_fight_date'].unique())

        # Find the split point by dates (not rows) to ensure 80/20 split
        n_dates = len(unique_dates)
        split_date_idx = int(n_dates * 0.8)

        # Get the cutoff date
        if split_date_idx < n_dates:
            cutoff_date = unique_dates[split_date_idx]

            # All fights before cutoff date go to train
            train_data = remaining_data[remaining_data['current_fight_date'] < cutoff_date].copy()

            # All fights from cutoff date onwards go to validation
            val_data = remaining_data[remaining_data['current_fight_date'] >= cutoff_date].copy()
        else:
            # Edge case: if not enough dates, use the last date for split
            train_data = remaining_data.copy()
            val_data = pd.DataFrame()  # Empty validation set

        print(f"Split using cutoff date: {cutoff_date if split_date_idx < n_dates else 'N/A'}")
        # ========== END FIX ==========

        # Remove duplicate fights with random ordering (no alphabetical enforcement)
        test_data = self._remove_duplicate_fights(test_data, random=False)

        # Sort datasets by date only
        train_data = train_data.sort_values(by='current_fight_date', ascending=True)
        val_data = val_data.sort_values(by='current_fight_date', ascending=True) if not val_data.empty else val_data
        test_data = test_data.sort_values(by=['current_fight_date', 'fighter_a'], ascending=[True, True])

        # Remove highly correlated features **after** the temporal split to avoid leakage
        removed_features: List[str] = []
        if not train_data.empty:
            train_data, removed_features = self.utils.remove_correlated_features(
                train_data,
                correlation_threshold=0.95,
                protected_columns=[
                    'winner',
                    'current_fight_open_odds',
                    'current_fight_open_odds_b',
                    'current_fight_open_odds_diff',
                    'current_fight_open_odds_ratio',
                    'current_fight_closing_odds',
                    'current_fight_closing_odds_b',
                    'current_fight_closing_odds_diff',
                    'current_fight_closing_odds_ratio',
                    'current_fight_closing_range_end',
                    'current_fight_closing_range_end_b',
                    'current_fight_closing_open_diff_a',
                    'current_fight_closing_open_diff_b'
                ]
            )

            if removed_features:
                val_data = val_data.drop(columns=removed_features, errors='ignore')
                test_data = test_data.drop(columns=removed_features, errors='ignore')

        # ========== LEAKAGE CHECK #5: Train/Test Split ==========
        if self.enable_verification:
            print("\n" + "=" * 60)
            print("LEAKAGE CHECK #5: Train/Test Split Verification")
            print("=" * 60)

            if not train_data.empty:
                print(
                    f"Train date range: {train_data['current_fight_date'].min()} to {train_data['current_fight_date'].max()}")
            if not val_data.empty:
                print(
                    f"Val date range: {val_data['current_fight_date'].min()} to {val_data['current_fight_date'].max()}")
            if not test_data.empty:
                print(
                    f"Test date range: {test_data['current_fight_date'].min()} to {test_data['current_fight_date'].max()}")

            # Check for date overlap
            overlap_issues = []
            if not train_data.empty and not val_data.empty:
                if train_data['current_fight_date'].max() >= val_data['current_fight_date'].min():
                    overlap_issues.append("Train and validation dates overlap")
                    print("LEAKAGE: Train and validation dates overlap!")

            if not val_data.empty and not test_data.empty:
                if val_data['current_fight_date'].max() >= test_data['current_fight_date'].min():
                    overlap_issues.append("Validation and test dates overlap")
                    print("LEAKAGE: Validation and test dates overlap!")

            if not overlap_issues:
                print("No date overlap between train/val/test sets")

            # Additional check: verify no duplicate dates across sets
            if not train_data.empty and not val_data.empty:
                train_dates = set(train_data['current_fight_date'].unique())
                val_dates = set(val_data['current_fight_date'].unique())
                common_dates = train_dates.intersection(val_dates)
                if common_dates:
                    print(f"WARNING: {len(common_dates)} dates appear in both train and val sets")
                    print(f"   Common dates: {sorted(list(common_dates))[:5]}")  # Show first 5

            print("=" * 60)
        # ========== END LEAKAGE CHECK #5 ==========

        # Save datasets
        self.fight_processor._save_csv(train_data, 'train_test/train_data.csv')
        self.fight_processor._save_csv(val_data, 'train_test/val_data.csv')
        self.fight_processor._save_csv(test_data, 'train_test/test_data.csv')

        # Save removed features
        with open(os.path.join(self.fight_processor.data_dir, 'train_test/removed_features.txt'), 'w') as file:
            file.write(','.join(removed_features))

        print(
            f"Train set size: {len(train_data)}, Validation set size: {len(val_data)}, Test set size: {len(test_data)}")
        print(f"Removed {len(removed_features)} correlated features")

        return train_data, val_data, test_data

    def _remove_duplicate_fights(self, df: pd.DataFrame, random=True) -> pd.DataFrame:
        """
        Remove duplicate fights, with option to enforce alphabetical ordering or keep random duplicates.

        Args:
            df: DataFrame with fighter data
            random: If True, keep random duplicates. If False, enforce alphabetical ordering.

        Returns:
            DataFrame with duplicates removed
        """
        df = df.copy()

        # Create a temporary fight_pair column based on sorted fighter names
        df['fight_pair'] = df.apply(
            lambda row: tuple(sorted([row['fighter_a'], row['fighter_b']])) + (row['current_fight_date'],),
            axis=1
        )

        if random:
            # Shuffle the data to randomize which duplicate is kept
            df = df.sample(frac=1, random_state=42)  # Set random_state for reproducibility

            # Drop duplicates based on the fight_pair column
            df = df.drop_duplicates(subset=['fight_pair'], keep='first')
        else:

            result_rows = []

            # Process each unique fight pair
            for pair, group in df.groupby('fight_pair'):
                # Check if any row has fighter_a alphabetically before fighter_b
                alpha_rows = group[group['fighter_a'] <= group['fighter_b']]

                if len(alpha_rows) > 0:
                    # Add the first alphabetically ordered row
                    result_rows.append(alpha_rows.iloc[0])
                else:
                    # No alphabetically ordered row exists, take the first row
                    result_rows.append(group.iloc[0])

            # Create a new DataFrame from the selected rows
            df = pd.DataFrame(result_rows)

            # Sort by date and then alphabetically by fighter_a
            df = df.sort_values(by=['current_fight_date', 'fighter_a'], ascending=[True, True])

        # Drop the temporary fight_pair column and reset the index
        return df.drop(columns=['fight_pair']).reset_index(drop=True)


# =============================================================================
# Comprehensive Data Integrity Verification
# =============================================================================

def verify_data_integrity(data_dir: str = "../../../data", sample_size: int = 5):
    """
    Comprehensive data leakage verification
    Run this after your data processing pipeline

    Args:
        data_dir: Base data directory
        sample_size: Number of samples to check in detail
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE DATA LEAKAGE VERIFICATION")
    print("="*80)

    issues_found = []

    try:
        # Load the processed data
        combined_rounds = pd.read_csv(f"{data_dir}/processed/combined_rounds.csv")
        combined_sorted = pd.read_csv(f"{data_dir}/processed/combined_sorted_fighter_stats.csv")

        # Test 1: Check a specific fighter's progression
        test_fighter = combined_rounds['fighter'].value_counts().index[0]  # Most common fighter
        fighter_data = combined_rounds[combined_rounds['fighter'] == test_fighter].sort_values('fight_date')

        print(f"\n1. Checking fighter: {test_fighter}")
        print(f"   Total fights in dataset: {len(fighter_data)}")

        # Check first 3 fights
        for i in range(min(3, len(fighter_data))):
            fight = fighter_data.iloc[i]
            print(f"\n   Fight {i+1} ({fight['fight_date']}):")
            print(f"     total_fights: {fight.get('total_fights', 'N/A')} (expected: {i+1})")
            print(f"     total_wins: {fight.get('total_wins', 'N/A')}")
            print(f"     win_streak: {fight.get('win_streak', 'N/A')}")

            # Verify
            if fight.get('total_fights', 0) != i+1:
                issues_found.append(f"Fighter {test_fighter}: total_fights mismatch in fight {i+1}")
                print(f"LEAKAGE DETECTED!")

        # Test 2: Check date ordering
        print("\n2. Checking date ordering in career stats:")
        date_issues = 0
        for fighter, fighter_group in combined_rounds.groupby('fighter'):
            dates = pd.to_datetime(fighter_group['fight_date']).values
            if not all(dates[i] <= dates[i+1] for i in range(len(dates)-1)):
                date_issues += 1
                if date_issues <= 3:  # Show first 3 issues
                    print(f"LEAKAGE: Fighter {fighter} has unordered dates!")
                    issues_found.append(f"Fighter {fighter}: unordered dates")

        if date_issues > 3:
            print(f"   ... and {date_issues - 3} more fighters with date issues")
        elif date_issues == 0:
            print("All fighters have properly ordered dates")

        # Test 3: Check train/test split
        train_data = pd.read_csv(f"{data_dir}/train_test/train_data.csv")
        val_data = pd.read_csv(f"{data_dir}/train_test/val_data.csv")
        test_data = pd.read_csv(f"{data_dir}/train_test/test_data.csv")

        print("\n3. Checking train/val/test split:")
        print(f"   Train: {train_data['current_fight_date'].min()} to {train_data['current_fight_date'].max()}")
        print(f"   Val:   {val_data['current_fight_date'].min()} to {val_data['current_fight_date'].max()}")
        print(f"   Test:  {test_data['current_fight_date'].min()} to {test_data['current_fight_date'].max()}")

        if train_data['current_fight_date'].max() >= val_data['current_fight_date'].min():
            print("LEAKAGE: Train and validation dates overlap!")
            issues_found.append("Train/val date overlap")
        elif val_data['current_fight_date'].max() >= test_data['current_fight_date'].min():
            print("LEAKAGE: Validation and test dates overlap!")
            issues_found.append("Val/test date overlap")
        else:
            print("No date overlap between train/val/test")

        # Test 4: Check for future data in features
        print("\n4. Checking for future data in features (sample):")
        for i in range(min(sample_size, len(test_data))):
            row = test_data.iloc[i]
            if 'total_fights' in row:
                # This is a simplified check - you'd need more context to verify thoroughly
                print(f"   Sample {i+1}: Fighter A has {row.get('total_fights_fighter_a_avg_last_3', 'N/A')} total fights")

    except FileNotFoundError as e:
        print(f"\nError: Required file not found - {e}")
        issues_found.append(f"Missing file: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        issues_found.append(f"Unexpected error: {e}")

    # Final summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    if not issues_found:
        print("ALL CHECKS PASSED - No data leakage detected!")
    else:
        print(f"ISSUES FOUND ({len(issues_found)} total):")
        for issue in issues_found[:10]:  # Show first 10 issues
            print(f"   - {issue}")
        if len(issues_found) > 10:
            print(f"   ... and {len(issues_found) - 10} more issues")

    print("="*80)

    return len(issues_found) == 0


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    # Initialize processors with verification enabled
    fight_processor = FightDataProcessor(enable_verification=True)
    matchup_processor = MatchupProcessor(data_dir=str(fight_processor.data_dir), enable_verification=True)

    # Uncomment the functions you want to run
    print("Starting UFC data processing pipeline with leakage verification...")

    # Process fight data
    fight_processor.combine_rounds_stats('processed/ufc_fight_processed.csv')

    # Calculate Elo ratings
    combined_rounds_path = fight_processor.data_dir / 'processed' / 'combined_rounds.csv'
    calculate_elo_ratings(str(combined_rounds_path))

    # Combine fighter stats
    fight_processor.combine_fighters_stats('processed/combined_rounds.csv')

    # Create matchup data
    matchup_processor.create_matchup_data('processed/combined_sorted_fighter_stats.csv', 3, True)

    # Split into train/val/test
    matchup_processor.split_train_val_test(
        'matchup data/matchup_data_3_avg_name.csv',
        '2025-01-01',
        '2025-12-31',
        20
    )

    # Run comprehensive verification
    print("\nRunning comprehensive data integrity check...")
    integrity_passed = verify_data_integrity()

    if integrity_passed:
        print("\nData processing completed successfully with no leakage detected!")
    else:
        print("\n⚠Data processing completed but potential leakage issues were detected.")
        print("Please review the verification output above.")


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"\nTotal runtime: {end_time - start_time}")