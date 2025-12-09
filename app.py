import os
import sys
import subprocess
import pandas as pd
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import eventlet

eventlet.monkey_patch()

BACKEND_SCRIPT = "bulk_fetcher_6.py"
DEFAULT_CSV = "students.csv"
RAW_DATA = "raw_results.csv"
OUTPUT_EXCEL = "vtu_results.xlsx"
PY = sys.executable

app = Flask(__name__)
socketio = SocketIO(app, async_mode="eventlet")


# -----------------------------------------
# FRONTEND ROUTES
# -----------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/download")
def download():
    try:
        return send_from_directory(os.getcwd(), OUTPUT_EXCEL, as_attachment=True)
    except FileNotFoundError:
        return "Excel not generated yet", 404


# -----------------------------------------
# IMPORT CSV
# -----------------------------------------
@socketio.on("import-csv")
def import_csv():
    if not os.path.exists(DEFAULT_CSV):
        emit("log-message", {"data": "[Error] students.csv missing\n"})
        return

    with open(DEFAULT_CSV) as f:
        lines = f.readlines()[1:]

    emit("csv-data", {"usns": "".join(lines)})
    emit("log-message", {"data": f"Imported {len(lines)} USNs\n"})


# -----------------------------------------
# START FETCHING
# -----------------------------------------
@socketio.on("start-fetch")
def start_fetch(msg):
    usns = msg.get("usns", [])
    vtu_url = msg.get("url", "")

    if not usns:
        emit("log-message", {"data": "No USNs provided.\n"})
        return

    if not vtu_url:
        emit("log-message", {"data": "No VTU URL provided.\n"})
        return

    # Write USNs to CSV
    with open(DEFAULT_CSV, "w") as f:
        f.write("USN\n")
        for u in usns:
            f.write(u + "\n")

    # Launch scraper with URL
    socketio.start_background_task(target=run_scraper, vtu_url=vtu_url)
    emit("fetch-started")


# -----------------------------------------
# RUN SCRAPER WITH URL ARG
# -----------------------------------------
def run_scraper(vtu_url):
    cmd = [PY, BACKEND_SCRIPT, vtu_url]  # <-- IMPORTANT FIXED ARGUMENT
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # Stream logs back to UI
    for line in iter(process.stdout.readline, ""):
        socketio.emit("log-message", {"data": line})
        eventlet.sleep(0.01)

    process.wait()
    socketio.emit("fetch-complete")

    if os.path.exists(OUTPUT_EXCEL):
        socketio.emit("download-ready")


# -----------------------------------------
# SGPA CALCULATION
# -----------------------------------------
def marks_to_gp(m):
    try:
        m = int(m)
    except:
        return 0

    if m >= 90: return 10
    if m >= 80: return 9
    if m >= 70: return 8
    if m >= 60: return 7
    if m >= 50: return 6
    if m >= 45: return 5
    if m >= 40: return 4
    return 0


@socketio.on("sgpa-credits")
def sgpa_calc(data):
    credits = data.get("credits", {})

    if not os.path.exists(RAW_DATA):
        emit("log-message", {"data": "[SGPA ERROR] raw_results.csv not found\n"})
        return

    df = pd.read_csv(RAW_DATA)
    sgpa_map = {}

    for usn, group in df.groupby("USN"):
        total_pts = 0
        total_cr = 0

        for _, row in group.iterrows():
            sub = row["Subject Code"]
            if sub not in credits:
                continue

            cr = credits[sub]
            gp = marks_to_gp(row["Total Marks"])

            total_pts += gp * cr
            total_cr += cr

        sgpa = round(total_pts / total_cr, 2) if total_cr else 0
        sgpa_map[usn] = sgpa

    # Update Excel file
    excel = pd.read_excel(OUTPUT_EXCEL)
    excel["SGPA"] = excel["USN"].apply(lambda u: sgpa_map.get(u, 0))
    excel.to_excel(OUTPUT_EXCEL, index=False)

    emit("log-message", {"data": "\n[SGPA] SGPA added to Excel.\n"})
    emit("download-ready")


# -----------------------------------------
if __name__ == "__main__":
    socketio.run(app, host="127.0.0.1", port=5000, allow_unsafe_werkzeug=True)
