import os
import sys
import subprocess
import threading
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import eventlet
eventlet.monkey_patch()

# --- Configuration ---
BACKEND_SCRIPT = "bulk_fetcher_4_excelfix.py"
DEFAULT_CSV = "students.csv"
OUTPUT_EXCEL = "vtu_results.xlsx"
PYTHON_EXECUTABLE = sys.executable
# ---------------------

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret-key-for-socketio!'
socketio = SocketIO(app, async_mode='eventlet')

# 1. Serve the HTML Webpage
@app.route('/')
def index():
    """Serves the main index.html page."""
    return render_template('index.html')

# 2. Create the Download Route
@app.route('/download')
def download_file():
    """Handles the file download request."""
    try:
        directory = os.getcwd() 
        print(f"Download requested for: {OUTPUT_EXCEL} from {directory}")
        return send_from_directory(directory, OUTPUT_EXCEL, as_attachment=True)
    except FileNotFoundError:
        print("Error: File not found.")
        return "Error: File not found. Run the fetcher first.", 404

# 3. Handle "Import CSV" button click
@socketio.on('import-csv')
def handle_import_csv():
    """Reads the default CSV and sends its content to the browser."""
    print(f"Client requested to import {DEFAULT_CSV}")
    try:
        if not os.path.exists(DEFAULT_CSV):
            emit('log-message', {'data': f"Error: '{DEFAULT_CSV}' not found.\n"})
            return
        with open(DEFAULT_CSV, 'r') as f:
            usns = f.readlines()[1:] # Read all lines, skip header ("USN")
            emit('csv-data', {'usns': "".join(usns)})
            emit('log-message', {'data': f"Successfully imported {len(usns)} USNs from {DEFAULT_CSV}\n"})
    except Exception as e:
        emit('log-message', {'data': f"Error importing CSV: {e}\n"})

# 4. Handle "Start Fetching" button click
@socketio.on('start-fetch')
def handle_start_fetch(message):
    """Writes the USNs to the CSV and starts the backend script in a thread."""
    usn_list = message.get('usns', [])
    vtu_url = message.get('url') # <-- [NEW] Get the URL from the UI

    if not usn_list:
        emit('log-message', {'data': "Error: No USNs provided.\n"})
        return
    if not vtu_url:
        emit('log-message', {'data': "Error: No VTU URL provided.\n"})
        return

    # 1. Overwrite the students.csv file
    print(f"Writing {len(usn_list)} USNs to {DEFAULT_CSV}...")
    try:
        with open(DEFAULT_CSV, 'w') as f:
            f.write("USN\n") # Write header
            for usn in usn_list:
                f.write(f"{usn}\n")
    except Exception as e:
        emit('log-message', {'data': f"Error writing to {DEFAULT_CSV}: {e}\n"})
        return

    # 2. Start the long-running script in a background thread
    print("Starting backend script thread...")
    # --- [MODIFIED] Pass the vtu_url to the target function ---
    socketio.start_background_task(target=run_fetcher_script, vtu_url=vtu_url)
    emit('fetch-started') # Tell the UI to disable buttons

def run_fetcher_script(vtu_url): # <-- [MODIFIED] Accept the URL
    """
    Runs the bulk_fetcher.py script as a subprocess and streams
    its stdout/stderr to the web UI in real-time.
    """
    print(f"Using Python: {PYTHON_EXECUTABLE}")
    # --- [MODIFIED] Add the URL as a command-line argument ---
    command = [PYTHON_EXECUTABLE, BACKEND_SCRIPT, vtu_url]
    print(f"Running command: {command}")
    
    socketio.emit('log-message', {'data': f"Starting backend script: {BACKEND_SCRIPT}...\n"})
    socketio.emit('log-message', {'data': "-"*30 + "\n"})

    try:
        # Start the subprocess
        process = subprocess.Popen(
            command, # <-- [MODIFIED] Use the command with the URL
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1 # Line-buffered
        )

        # 3. Read output line-by-line in real-time
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                socketio.emit('log-message', {'data': line})
                eventlet.sleep(0.01) # Yield to other tasks

        # 4. Wait for the process to end
        process.wait()
        socketio.emit('log-message', {'data': "\n" + "-"*30 + "\n"})
        socketio.emit('log-message', {'data': f"Backend script finished with exit code {process.returncode}\n"})
    except Exception as e:
        socketio.emit('log-message', {'data': f"\n--- FATAL ERROR ---\n{e}\n"})
    finally:
        # 5. Tell the UI to re-enable the buttons
        socketio.emit('fetch-complete')
        
        if os.path.exists(OUTPUT_EXCEL):
            socketio.emit('download-ready')
        print("Backend script thread finished.")

# 4. Main entry point
if __name__ == '__main__':
    print("Starting Flask-SocketIO server on http://127.0.0.1:5000")
    socketio.run(app, host='127.0.0.1', port=5000, allow_unsafe_werkzeug=True)

