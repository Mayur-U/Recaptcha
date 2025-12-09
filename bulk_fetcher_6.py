# === VTU SCRAPER WITH SUBJECT EXTRACTION & URL ARG SUPPORT ===

import os
import sys
import time
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
    TimeoutException, NoAlertPresentException, UnexpectedAlertPresentException
)

# -----------------------------------------
# Config
# -----------------------------------------
VTU_RESULTS_URL = "https://results.vtu.ac.in/JJEcbcs25/index.php"

# 👇 FIX: Accept URL from app.py
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
    img = cv2.GaussianBlur(img, (5,5), 0)
    _, img = cv2.threshold(img, 0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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
    except:
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
    except:
        pass

    pct = (total / max_total * 100) if max_total else 0

    return (usn, name), sub_rows, {
        "total_obtained": total,
        "total_max": max_total,
        "percentage": pct
    }


# -----------------------------------------
# MAIN
# -----------------------------------------
def main():
    try:
        df = pd.read_csv(INPUT_CSV)
        usns = df["USN"].tolist()
    except:
        print("Error reading students.csv")
        return

    try:
        model = tf.keras.models.load_model(MODEL_FILE)
    except:
        print("Error loading CAPTCHA model")
        return

    driver = webdriver.Chrome()

    all_subs = []
    all_summ = []
    unique = set()

    for usn in usns:
        print(f"\n--- {usn} ---")
        driver.get(VTU_RESULTS_URL)
        wait = WebDriverWait(driver, 10)
        success = False

        for attempt in range(MAX_ATTEMPTS):
            try:
                box = wait.until(EC.presence_of_element_located((By.NAME, "lns")))
                cbox = driver.find_element(By.NAME, "captchacode")
                img = driver.find_element(By.XPATH, "//img[contains(@src,'vtu_captcha.php')]")
                btn = driver.find_element(By.ID, "submit")

                box.clear()
                cbox.clear()
                box.send_keys(usn)

                img.screenshot("cap.png")
                pred = decode(model.predict(preprocess("cap.png"), verbose=0))
                print("CAPTCHA:", pred)

                if len(pred) != 6:
                    driver.refresh()
                    continue

                cbox.send_keys(pred)
                old = driver.current_url
                btn.click()

                WebDriverWait(driver, 4).until(EC.url_changes(old))

                # alert check
                try:
                    a = WebDriverWait(driver, 1).until(EC.alert_is_present())
                    a.accept()
                    driver.refresh()
                    continue
                except:
                    pass

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
                break

            except:
                driver.refresh()

        if not success:
            print("FAILED:", usn)

        pd.DataFrame(all_subs).to_csv(RAW_DATA, index=False)
        pd.DataFrame(all_summ).to_csv(RAW_SUMMARY, index=False)

    driver.quit()

    print("\n--- SCRAPING COMPLETE ---")
    print("[SUBJECTS]:" + ",".join(sorted(unique)))

    # Generate Excel without SGPA (added later in app.py)
    try:
        subs = pd.read_csv(RAW_DATA)
        summ = pd.read_csv(RAW_SUMMARY)

        subs.drop_duplicates(inplace=True)
        summ.drop_duplicates(subset=["USN"], keep="last", inplace=True)

        pivot = subs.pivot_table(
            index=["USN", "Name"],
            columns="Subject Code",
            values=["Internal Marks", "External Marks", "Total Marks", "Result"],
            aggfunc="first"
        )
        pivot.columns = [f"{c2} - {c1}" for c1, c2 in pivot.columns]
        pivot.reset_index(inplace=True)

        summ["Percentage"] = summ["percentage"].apply(lambda x: f"{x:.2f}%" if isinstance(x, float) else "")
        summ = summ.rename(columns={
            "total_obtained": "Overall Total",
            "total_max": "Overall Max Marks"
        })

        final = pd.merge(pivot, summ, on=["USN", "Name"], how="outer")
        final.to_excel(OUTPUT_EXCEL, index=False)

        print("Excel generated.")
    except Exception as e:
        print("Excel failed:", e)


if __name__ == "__main__":
    main()
