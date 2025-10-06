import re
import time
import random
import pandas as pd
import os
import traceback
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from datetime import datetime
from rapidfuzz.fuzz import ratio
import concurrent.futures
from tqdm import tqdm
import threading
import queue


class BestFightOddsScraperSelenium:
    def __init__(self, num_workers=3):
        self.num_workers = num_workers
        self.unprocessed_fighters = []
        self.progress_queue = queue.Queue()
        self.lock = threading.Lock()
        self.total_fighters = 0
        self.processed_count = 0

    def initialize_driver(self):
        options = Options()
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--log-level=3')
        return webdriver.Chrome(options=options)

    @staticmethod
    def clean_movement(movement):
        if pd.isna(movement) or movement == "":
            return None
        match = re.search(r'([+-]?\d+(\.\d+)?)', str(movement))
        if match:
            try:
                return float(match.group(1)) / 100
            except ValueError:
                return None
        return None

    def scrape_fighter(self, fighter, driver):
        odds_data = []
        similarity_threshold = 85

        for attempt in range(1, 6):
            try:
                print(f"\nAttempting to scrape {fighter} (Attempt {attempt})")
                driver.get("https://www.bestfightodds.com/search")

                # Search for fighter
                search_input = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.NAME, "query"))
                )
                search_input.clear()
                search_input.send_keys(fighter)

                search_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//input[@type='submit']"))
                )
                search_button.click()
                time.sleep(1)

                # Find odds table
                try:
                    odds_table = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.XPATH, "//table[@class='team-stats-table']"))
                    )
                except TimeoutException:
                    try:
                        # Search results page
                        search_results = WebDriverWait(driver, 5).until(
                            EC.presence_of_element_located((By.XPATH, "//table[@class='content-list']"))
                        )
                        fighter_links = search_results.find_elements(By.XPATH, ".//a[contains(@href, '/fighters/')]")

                        # Find matching fighter
                        for link in fighter_links:
                            link_text = link.text.strip()
                            similarity = ratio(link_text.lower(), fighter.lower())
                            if similarity >= similarity_threshold:
                                link.click()
                                break
                        else:
                            print(f"No match found for {fighter}")
                            continue

                        odds_table = WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.XPATH, "//table[@class='team-stats-table']"))
                        )
                    except TimeoutException:
                        print(f"No odds table found for {fighter}")
                        continue

                # Process odds table
                rows = odds_table.find_elements(By.XPATH, ".//tr")[1:]
                for row in rows:
                    try:
                        fight_data = {
                            "Matchup": row.find_element(By.XPATH, ".//th[@class='oppcell']/a").text,
                            "Event": "",
                            "Open": "",
                            "Closing Range Start": "",
                            "Closing Range End": "",
                            "Movement": None,
                            "Date": ""
                        }

                        # Get event
                        try:
                            event_cell = row.find_element(By.XPATH, ".//td[@class='item-non-mobile'][1]")
                            fight_data["Event"] = event_cell.find_element(By.XPATH, ".//a").text
                        except NoSuchElementException:
                            pass

                        # Get odds
                        try:
                            fight_data["Open"] = row.find_element(By.XPATH, ".//td[@class='moneyline']/span").text
                            fight_data["Closing Range Start"] = row.find_element(By.XPATH,
                                                                                 ".//td[@class='moneyline'][2]/span").text
                            fight_data["Closing Range End"] = row.find_element(By.XPATH,
                                                                               ".//td[@class='moneyline'][3]/span").text
                        except NoSuchElementException:
                            pass

                        # Get movement
                        try:
                            movement = row.find_element(By.XPATH, ".//td[@class='change-cell']/span").text
                            fight_data["Movement"] = self.clean_movement(movement)
                        except NoSuchElementException:
                            pass

                        # Get date
                        try:
                            date_cell = row.find_element(
                                By.XPATH,
                                ".//td[@class='item-non-mobile'][@style='padding-left: 20px; color: #767676']"
                            )
                            fight_data["Date"] = date_cell.text.strip()
                        except NoSuchElementException:
                            pass

                        odds_data.append(fight_data)

                    except NoSuchElementException:
                        continue

                # Update progress
                with self.lock:
                    self.processed_count += 1
                    self.progress_queue.put(1)

                print(f"Successfully scraped {fighter}")
                return odds_data

            except Exception as e:
                print(f"Error processing {fighter}: {str(e)}")
                if attempt == 5:
                    self.unprocessed_fighters.append(fighter)
                time.sleep(random.uniform(0.5, 1))

        return []

    def process_batch(self, batch):
        driver = self.initialize_driver()
        batch_data = []
        try:
            for fighter in batch:
                fighter_data = self.scrape_fighter(fighter, driver)
                batch_data.extend(fighter_data)
                time.sleep(random.uniform(0.5, 1))
        finally:
            driver.quit()
        return batch_data

    def scrape_all(self, fighters):
        self.total_fighters = len(fighters)
        print(f"\nStarting scrape of {self.total_fighters} fighters using {self.num_workers} workers")

        # Create batches
        batch_size = max(1, len(fighters) // self.num_workers)
        batches = [fighters[i:i + batch_size] for i in range(0, len(fighters), batch_size)]

        # Progress bar setup
        pbar = tqdm(total=self.total_fighters, desc="Processing fighters")

        # Process batches
        all_data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_batch = {executor.submit(self.process_batch, batch): batch for batch in batches}

            # Monitor progress
            while future_to_batch:
                done, _ = concurrent.futures.wait(future_to_batch, timeout=0.1)

                # Update progress bar
                while True:
                    try:
                        progress = self.progress_queue.get_nowait()
                        pbar.update(progress)
                    except queue.Empty:
                        break

                # Process completed futures
                for future in done:
                    try:
                        batch_data = future.result()
                        all_data.extend(batch_data)
                    except Exception as e:
                        print(f"\nBatch processing error: {str(e)}")
                    future_to_batch.pop(future)

        pbar.close()
        return pd.DataFrame(all_data)

    @staticmethod
    def clean_fight_odds_from_csv(input_csv_path, output_csv_path):
        fight_odds_df = pd.read_csv(input_csv_path)

        if 'Date' not in fight_odds_df.columns:
            fight_odds_df['Date'] = ""

        # Filter for UFC events
        fight_odds_df = fight_odds_df[
            fight_odds_df['Event'].str.contains('UFC', case=False, na=False)
            | fight_odds_df['Event'].isna()
            ]

        # Process rows
        modified_rows = []
        i = 0
        while i < len(fight_odds_df):
            row1 = fight_odds_df.iloc[i].copy()

            if pd.notna(row1['Event']):
                modified_rows.append(row1)

                if i + 1 < len(fight_odds_df):
                    row2 = fight_odds_df.iloc[i + 1].copy()
                    row2['Movement'] = row1['Movement']
                    row2['Event'] = row1['Event']
                    row1['Date'] = row2['Date']
                    modified_rows.append(row2)
                    i += 2
                else:
                    i += 1
            else:
                i += 1

        # Create final DataFrame
        modified_df = pd.DataFrame(modified_rows, columns=fight_odds_df.columns)

        # Parse dates
        modified_df['Date'] = modified_df['Date'].apply(lambda x: BestFightOddsScraperSelenium.parse_custom_date(x))
        modified_df = modified_df[modified_df['Date'].notna()]

        # -------------------------
        # CHANGED THIS LINE ONLY:
        # -------------------------
        # Instead of '%b %d %Y', use '%Y-%m-%d'
        modified_df['Date'] = modified_df['Date'].dt.strftime('%Y-%m-%d')

        # Sort and clean
        modified_df = modified_df.sort_values(['Matchup', 'Date'])
        if 'Event' in modified_df.columns:
            modified_df = modified_df.drop('Event', axis=1)

        # Remove duplicates based on 'Matchup' and 'Date', keeping the first occurrence only
        modified_df = modified_df.drop_duplicates(subset=['Matchup', 'Date'], keep='first')

        modified_df.to_csv(output_csv_path, index=False)
        return modified_df

    @staticmethod
    def parse_custom_date(date_string):
        if pd.isna(date_string):
            return pd.NaT
        date_string = str(date_string)
        date_string = date_string.replace('th', '').replace('st', '').replace('nd', '').replace('rd', '')
        try:
            return datetime.strptime(date_string, '%b %d %Y')
        except ValueError:
            return pd.NaT


if __name__ == "__main__":
    start_time = time.time()

    # --- Configuration ---
    # run_scraping: Controls if scraping and processing of new data occurs
    run_scraping = True
    # run_append_new: Controls if comparison and appending to master file occurs
    # Typically, you'd want this True if run_scraping is True
    run_append_new = True

    # File Paths
    master_fighter_list_file = "../../../data/processed/combined_rounds.csv"
    # This is your main, persistent cleaned data file
    existing_cleaned_data_file = "../../../data/processed/cleaned_fight_odds.csv"
    # Temporary files for the current run's scrape results
    temp_raw_scrape_file = "../../../data/raw/fight_odds_temp_raw.csv"
    temp_cleaned_scrape_file = "../../../data/processed/fight_odds_temp_cleaned.csv"

    # --- Step 1: Load Existing Cleaned Data ---
    df_existing = pd.DataFrame()
    existing_identifiers = set() # To store unique (Matchup, Date) tuples

    print(f"--- Loading existing data from: {existing_cleaned_data_file} ---")
    if os.path.exists(existing_cleaned_data_file):
        try:
            df_existing = pd.read_csv(existing_cleaned_data_file)
            if not df_existing.empty and 'Matchup' in df_existing.columns and 'Date' in df_existing.columns:
                # Ensure columns used for identification are strings for consistent hashing
                df_existing['Matchup'] = df_existing['Matchup'].astype(str)
                df_existing['Date'] = df_existing['Date'].astype(str)
                # Create a set of existing (Matchup, Date) tuples for fast lookup
                existing_identifiers = set(df_existing.apply(lambda row: (row['Matchup'], row['Date']), axis=1))
                print(f"Loaded {len(df_existing)} records ({len(existing_identifiers)} unique identifiers).")
            else:
                 print("Existing file is empty or missing 'Matchup'/'Date' columns.")
        except Exception as e:
            print(f"Warning: Could not load or process existing file {existing_cleaned_data_file}. Assuming empty. Error: {e}")
            df_existing = pd.DataFrame() # Ensure it's empty on error
            existing_identifiers = set()
    else:
        print("Existing cleaned data file not found. Will create a new one if new data is found.")

    # --- Step 2: Scrape Data (Full Rescrape, as per original logic) ---
    df_new_cleaned = pd.DataFrame() # Initialize empty DataFrame for newly cleaned data

    if run_scraping:
        print("\n--- Starting Full Scrape ---")
        try:
            # Read fighter list from master source
            combined_rounds_df = pd.read_csv(master_fighter_list_file)
            fighters = list(set(combined_rounds_df["fighter"].unique().tolist()))
            print(f"Preparing to scrape {len(fighters)} fighters from {master_fighter_list_file}")

            # Initialize scraper
            scraper = BestFightOddsScraperSelenium(num_workers=3) # Adjust workers if needed

            # Scrape data
            new_odds_df_raw = scraper.scrape_all(fighters)

            # Save raw data from this scrape to a temporary file
            new_odds_df_raw.to_csv(temp_raw_scrape_file, index=False)
            print(f"\nRaw scrape data saved to temporary file: {temp_raw_scrape_file}")

            # Print scraping summary (same as before)
            print("\nScraping Summary:")
            total_processed = len(fighters) - len(scraper.unprocessed_fighters)
            print(f"Total fighters processed: {total_processed}")
            success_rate = (total_processed / len(fighters)) * 100 if fighters else 0
            print(f"Success rate: {success_rate:.2f}%")
            if scraper.unprocessed_fighters:
                print(f"\nUnprocessed fighters ({len(scraper.unprocessed_fighters)}):") # Condensed output
            else:
                print("\nAll fighters processed successfully!")

            # --- Step 3: Clean the Newly Scraped Data ---
            print(f"\n--- Cleaning scraped data from {temp_raw_scrape_file} ---")
            if not new_odds_df_raw.empty:
                try:
                    # Clean the *temporary raw file*, output to *temporary cleaned file*
                    df_new_cleaned = BestFightOddsScraperSelenium.clean_fight_odds_from_csv(
                        temp_raw_scrape_file, temp_cleaned_scrape_file
                    )
                    print(f"Cleaned data saved temporarily to: {temp_cleaned_scrape_file}")
                    if df_new_cleaned.empty:
                        print("Warning: Cleaning process resulted in an empty DataFrame.")
                except Exception as e:
                    print(f"Error during cleaning of new data: {e}")
                    traceback.print_exc()
                    df_new_cleaned = pd.DataFrame() # Ensure empty on error
            else:
                print("Raw scrape data was empty, skipping cleaning.")

        except Exception as e:
            print(f"An error occurred during the scraping or cleaning phase: {e}")
            traceback.print_exc()
            df_new_cleaned = pd.DataFrame() # Ensure empty if scraping/cleaning failed
    else:
        print("\n--- Scraping and Cleaning of new data skipped via configuration ---")


    # --- Step 4: Compare, Identify New Rows, and Append ---
    if run_append_new:
        print(f"\n--- Comparing and Appending New Records to {existing_cleaned_data_file} ---")

        # Ensure df_new_cleaned is a DataFrame, even if empty from previous steps
        if not isinstance(df_new_cleaned, pd.DataFrame):
             df_new_cleaned = pd.DataFrame()

        # Proceed only if the newly cleaned data is not empty and has the key columns
        if not df_new_cleaned.empty and 'Matchup' in df_new_cleaned.columns and 'Date' in df_new_cleaned.columns:
            # Ensure key columns are strings for comparison
            df_new_cleaned['Matchup'] = df_new_cleaned['Matchup'].astype(str)
            df_new_cleaned['Date'] = df_new_cleaned['Date'].astype(str)

            # Identify rows in the new data whose (Matchup, Date) are NOT in the existing set
            new_rows_mask = df_new_cleaned.apply(
                lambda row: (row['Matchup'], row['Date']) not in existing_identifiers,
                axis=1
            )
            df_genuinely_new = df_new_cleaned[new_rows_mask]

            if not df_genuinely_new.empty:
                print(f"Identified {len(df_genuinely_new)} genuinely new fight records to add.")

                # Append only the new rows to the existing DataFrame
                # Use ignore_index=True to reset the index for the combined DataFrame
                df_final = pd.concat([df_existing, df_genuinely_new], ignore_index=True)

                # Optional: Sort the final DataFrame for consistency
                df_final = df_final.sort_values(by=['Matchup', 'Date'])

                # Save the updated DataFrame back to the original file path
                try:
                    df_final.to_csv(existing_cleaned_data_file, index=False)
                    print(f"Successfully appended new records. Master file updated: {existing_cleaned_data_file} (Total records: {len(df_final)})")
                except Exception as e:
                    print(f"ERROR: Failed to save updated master file: {e}")
                    traceback.print_exc()

            else:
                print("No genuinely new fight records found in the latest scrape to append.")
                # Optionally, re-save the existing DataFrame to ensure consistent format/columns if needed
                # try:
                #    df_existing.to_csv(existing_cleaned_data_file, index=False)
                #    print(f"Re-saved existing data (no changes) to {existing_cleaned_data_file}")
                # except Exception as e:
                #    print(f"Warning: Failed to re-save existing file: {e}")

        elif run_scraping: # If scraping ran but produced no usable cleaned data
             print("Newly scraped data was empty or unusable after cleaning. No comparison performed.")
        else: # If scraping didn't run
             print("Scraping was skipped. No new data to compare or append.")

    else:
        print("\n--- Comparison and Appending skipped via configuration ---")


    # --- Step 5: Cleanup Temporary Files (Optional) ---
    # You might want to keep these for debugging issues
    # print("\n--- Cleaning up temporary files ---")
    # try:
    #     if os.path.exists(temp_raw_scrape_file):
    #         os.remove(temp_raw_scrape_file)
    #         # print(f"Removed temporary raw file: {temp_raw_scrape_file}")
    #     if os.path.exists(temp_cleaned_scrape_file):
    #         os.remove(temp_cleaned_scrape_file)
    #         # print(f"Removed temporary cleaned file: {temp_cleaned_scrape_file}")
    # except Exception as e:
    #     print(f"Warning: Error removing temporary files: {e}")


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal time: {elapsed_time / 60:.2f} minutes")