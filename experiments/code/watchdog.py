"""Stop the running Astra session if the shared ledger reaches $18: one runaway call costs at most $6.40, so the total stays under $25."""
import json, os, signal, subprocess, sys, time
ledger = sys.argv[1]
while True:
    pids = subprocess.run(["pgrep", "-f", "astra.py gpt-6-astra"], capture_output=True, text=True).stdout.split()
    if not pids:
        print("astra finished; watchdog exiting"); break
    if json.load(open(ledger))["usd"] >= 18.0:
        for p in pids:
            os.kill(int(p), signal.SIGTERM)
        print("stopped astra at the $18 line"); break
    time.sleep(20)
