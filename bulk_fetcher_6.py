# === VTU SCRAPER WITH SUBJECT EXTRACTION & URL ARG SUPPORT (FIXED) ===
# This file is a patched version of your uploaded bulk_fetcher_6.py.
# Fixes: robust alert handling, safe refresh, stable CAPTCHA retry loop.

import os
import sys
import time
import traceback
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from bs4 import BeautifulSoup
from tensorflow.keras import backend as K
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoAlertPresentException, UnexpectedAlertPresentException, WebDriverException
)

# -----------------------------------------
# Config
# -----------------------------------------
VTU_RESULTS_URL = "https://results.vtu.ac.in/JJEcbcs25/index.php"

# Accept URL from app.py if provided
if len(sys.argv) > 1:
    VTU_RESULTS_URL = sys.argv[1]

MODEL_FILE = "vtu_captcha_predictor.h5"
INPUT_CSV = "students.csv"
RAW_DATA = "raw_results.csv"
RAW_SUMMARY = "raw_summary.csv"
OUTPUT_EXCEL = "vtu_results.xlsx"

IMG_WIDTH = 160
IMG_HEIGHT = 75
CHARACTERS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
num_to_char = {i: c for i, c in enumerate(CHARACTERS)}

MAX_ATTEMPTS = 15


# -----------------------------------------
# Helpers
# -----------------------------------------
def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found or unreadable: {path}")
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    return img.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)


def decode(pred):
    inp = np.ones(pred.shape[0]) * pred.shape[1]
    res = K.ctc_decode(pred, input_length=inp, greedy=True)[0][0]
    out = ""
    for x in res[0]:
        if x == -1:
            break
        out += num_to_char.get(x.numpy(), "")
    return out[:6]


def scrape(html):
    soup = BeautifulSoup(html, "html.parser")

    try:
        t = soup.find("table", {"class": "table-condensed"}).find_all("tr")
        usn = t[0].find_all("td")[1].text.strip().replace(":", "")
        name = t[1].find_all("td")[1].text.strip().replace(":", "")
    except Exception:
        usn, name = "UNKNOWN", "UNKNOWN"

    sub_rows = []
    total, max_total = 0, 0

    try:
        rows = soup.find("div", {"class": "divTableBody"}).find_all("div", {"class": "divTableRow"})
        for r in rows:
            c = r.find_all("div", {"class": "divTableCell"})
            if len(c) != 7 or "Subject Code" in c[0].text:
                continue

            code = c[0].text.strip()
            tmarks = c[4].text.strip()

            sub_rows.append({
                "Subject Code": code,
                "Subject Name": c[1].text.strip(),
                "Internal Marks": c[2].text.strip(),
                "External Marks": c[3].text.strip(),
                "Total Marks": tmarks,
                "Result": c[5].text.strip(),
                "Announced / Updated on": c[6].text.strip(),
            })

            try:
                total += int(tmarks)
            except:
                pass

            max_total += 100
    except Exception:
        pass

    pct = (total / max_total * 100) if max_total else 0

    return (usn, name), sub_rows, {
        "total_obtained": total,
        "total_max": max_total,
        "percentage": pct
    }


# -----------------------------------------
# Safe utilities for alert handling & refresh
# -----------------------------------------
def accept_alert_if_present(driver, timeout=0.5):
    """Attempt to accept an alert if present. Returns True if an alert was accepted."""
    try:
        alert = WebDriverWait(driver, timeout).until(EC.alert_is_present())
        try:
            print("[Info] Alert present — text:", alert.text)
        except Exception:
            pass
        alert.accept()
        return True
    except Exception:
        return False


def safe_refresh(driver):
    """Refresh the page while ensuring any modal alert is closed first."""
    # Try accepting any alert, then refresh
    try:
        try:
            # If an alert is present, accept it
            if accept_alert_if_present(driver, timeout=0.5):
                time.sleep(0.2)
        except Exception:
            pass
        driver.refresh()
    except UnexpectedAlertPresentException:
        # If an alert pops during refresh, accept and try again
        try:
            accept_alert_if_present(driver, timeout=0.5)
        except Exception:
            pass
        try:
            driver.refresh()
        except Exception:
            pass
    except WebDriverException as e:
        # Some WebDriver errors can be transient; print and continue
        print("[Warn] WebDriverException during refresh:", e)


# -----------------------------------------
# MAIN
# -----------------------------------------
def main():
    try:
        df = pd.read_csv(INPUT_CSV)
        usns = df["USN"].astype(str).str.strip().tolist()
    except Exception as e:
        print("Error reading students.csv:", e)
        return

    try:
        model = tf.keras.models.load_model(MODEL_FILE)
    except Exception as e:
        print("Error loading CAPTCHA model:", e)
        return

    # Start Chrome WebDriver (default). If you need to pass options, modify here.
    try:
        driver = webdriver.Chrome()
    except Exception as e:
        print("Error starting Chrome WebDriver:", e)
        return

    all_subs = []
    all_summ = []
    unique = set()

    for usn in usns:
        print(f"\n--- Processing USN: {usn} ---")
        try:
            driver.get(VTU_RESULTS_URL)
        except Exception as e:
            print("[Error] driver.get failed:", e)
            safe_refresh(driver)

        wait = WebDriverWait(driver, 10)
        success = False

        for attempt in range(1, MAX_ATTEMPTS + 1):
            print(f"Attempt {attempt}/{MAX_ATTEMPTS} for {usn}")
            try:
                # Wait for the form fields
                box = wait.until(EC.presence_of_element_located((By.NAME, "lns")))
                cbox = driver.find_element(By.NAME, "captchacode")
                img = driver.find_element(By.XPATH, "//img[contains(@src,'vtu_captcha.php')]")
                btn = driver.find_element(By.ID, "submit")

                # Fill USN
                try:
                    box.clear()
                except Exception:
                    pass
                try:
                    cbox.clear()
                except Exception:
                    pass
                box.send_keys(usn)

                # Capture captcha image
                img.screenshot("cap.png")

                # Preprocess & predict
                try:
                    prep = preprocess("cap.png")
                    pred = decode(model.predict(prep, verbose=0))
                except Exception as e:
                    print("[Warn] CAPTCHA preprocess/predict failed:", e)
                    pred = ""

                print("CAPTCHA predicted:", pred)

                # If prediction length not 6 -> safe refresh and retry
                if not isinstance(pred, str) or len(pred) != 6:
                    print("[Info] Pred length invalid, refreshing and retrying.")
                    safe_refresh(driver)
                    time.sleep(0.5)
                    continue

                # Enter captcha and submit
                try:
                    cbox.send_keys(pred)
                except Exception:
                    try:
                        cbox.clear()
                        cbox.send_keys(pred)
                    except Exception:
                        pass

                # Store current url to detect change
                try:
                    old_url = driver.current_url
                except UnexpectedAlertPresentException:
                    # If an alert is present while getting URL, accept it and refresh
                    accept_alert_if_present(driver, timeout=0.5)
                    safe_refresh(driver)
                    continue

                # Click submit safely
                try:
                    btn.click()
                except UnexpectedAlertPresentException:
                    # Alert popped during click; accept and retry refresh
                    accept_alert_if_present(driver, timeout=0.5)
                    safe_refresh(driver)
                    time.sleep(0.5)
                    continue
                except Exception as e:
                    print("[Warn] Click failed:", e)
                    # try JavaScript click as fallback
                    try:
                        driver.execute_script("arguments[0].click();", btn)
                    except Exception:
                        safe_refresh(driver)
                        continue

                # Immediately check for alert (invalid captcha)
                if accept_alert_if_present(driver, timeout=1):
                    print("[Info] Detected alert after submit (likely invalid captcha). Retrying.")
                    safe_refresh(driver)
                    time.sleep(0.5)
                    continue

                # Wait for page navigation (url change) — if alert occurs here, handle it
                try:
                    WebDriverWait(driver, 4).until(EC.url_changes(old_url))
                except UnexpectedAlertPresentException:
                    # If an unexpected alert interrupts, accept and retry
                    accept_alert_if_present(driver, timeout=0.5)
                    safe_refresh(driver)
                    time.sleep(0.5)
                    continue
                except TimeoutException:
                    # URL did not change — might be invalid captcha or same page loaded
                    # Check for alert one more time
                    if accept_alert_if_present(driver, timeout=1):
                        safe_refresh(driver)
                        time.sleep(0.5)
                        continue
                    # No alert and no url change — retry
                    safe_refresh(driver)
                    time.sleep(0.5)
                    continue

                # If we reached here, page changed successfully — scrape it
                try:
                    (u, name), subs, summ = scrape(driver.page_source)

                    for s in subs:
                        s["USN"] = u
                        s["Name"] = name
                        all_subs.append(s)
                        unique.add(s["Subject Code"])

                    summ["USN"] = u
                    summ["Name"] = name
                    all_summ.append(summ)

                    success = True
                    print(f"[Success] Scraped {u} - {name} with {len(subs)} subjects.")
                    break

                except Exception as e:
                    print("[Error] Scrape failed:", e)
                    traceback.print_exc()
                    safe_refresh(driver)
                    time.sleep(0.5)
                    continue

            except UnexpectedAlertPresentException:
                # Always accept unexpected alerts and retry
                try:
                    accept_alert_if_present(driver, timeout=0.5)
                except Exception:
                    pass
                safe_refresh(driver)
                time.sleep(0.5)
                continue

            except TimeoutException:
                print("[Warn] Timeout waiting for page elements. Refreshing and retrying.")
                safe_refresh(driver)
                time.sleep(0.5)
                continue

            except Exception as e:
                print("[Error] Unexpected exception in attempt loop:", e)
                traceback.print_exc()
                safe_refresh(driver)
                time.sleep(0.5)
                continue

        if not success:
            print(f"FAILED to fetch results for USN: {usn}")
            # Add a summary row marking the failure if desired
            all_summ.append({'USN': usn, 'Name': 'FETCH FAILED', 'percentage': 0, 'total_obtained': 0, 'total_max': 0})

        # Save intermediate CSVs so partial progress is kept
        try:
            pd.DataFrame(all_subs).to_csv(RAW_DATA, index=False)
            pd.DataFrame(all_summ).to_csv(RAW_SUMMARY, index=False)
        except Exception as e:
            print("[Warn] Could not save interim CSVs:", e)

    # End for all USNs
    try:
        driver.quit()
    except Exception:
        pass

    print("\n--- SCRAPING COMPLETE ---")
    print("[SUBJECTS]:" + ",".join(sorted(unique)))

    # Generate Excel without SGPA (SGPA added by app.py)
    try:
        subs_df = pd.read_csv(RAW_DATA)
        summ_df = pd.read_csv(RAW_SUMMARY)

        subs_df.drop_duplicates(inplace=True)
        summ_df.drop_duplicates(subset=["USN"], keep="last", inplace=True)

        pivot = pd.pivot_table(
            subs_df,
            index=["USN", "Name"],
            columns="Subject Code",
            values=["Internal Marks", "External Marks", "Total Marks", "Result"],
            aggfunc="first"
        )
        # flatten multiindex columns
        pivot.columns = [f"{c2} - {c1}" for c1, c2 in pivot.columns]
        pivot.reset_index(inplace=True)

        summ_df["Percentage"] = summ_df["percentage"].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else "")
        summ_df = summ_df.rename(columns={
            "total_obtained": "Overall Total",
            "total_max": "Overall Max Marks"
        })

        final = pd.merge(pivot, summ_df, on=["USN", "Name"], how="outer")
        final.to_excel(OUTPUT_EXCEL, index=False)

        print("[Success] Excel generated:", OUTPUT_EXCEL)
    except Exception as e:
        print("[Error] Excel generation failed:", e)
        traceback.print_exc()


if __name__ == "__main__":
    main()
