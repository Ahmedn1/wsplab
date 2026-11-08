# Experiment notifications

The user requested autonomous follow-through without manual status prompts.
`scripts/live/notify_watch.py` queues messages to this existing conversation via
`codex queue`. It does not launch an editor or change experiment budgets.

Runtime state: `.wsplab/notifications/`. Registered job definitions are in
`jobs/`, delivery receipts and errors in `state/`, and the service heartbeat in
`heartbeat.json`. Never mistake queue acceptance for an acknowledged assistant
turn. The installed CLI has no `queue --resume`; the target conversation must
remain available in Codex. A closed application, unavailable daemon, machine
shutdown or exhausted assistant quota can delay continuation. Failed queue
commands retry after 60 seconds; receipt deduplication survives watcher restarts.
A crash between queue acceptance and receipt persistence can cause a duplicate;
the job/event ID makes it identifiable.

The user service is `wsplab-notifications.service`, with restart on failure.
It is transient: it survives the initiating shell, not a machine reboot.
After a reboot, restart the same service with the same state directory.
Inspect with `systemctl --user status wsplab-notifications.service` and
`journalctl --user -u wsplab-notifications.service` on the host.

For every new background script, use the wrapper (supply a unique attempt ID):

```bash
python scripts/live/notify_watch.py run --id UNIQUE_JOB_ID \
  --thread 01a1af1a-a757-7340-affe-46c38c277e28 -- python path/to/script.py
```

The wrapper preserves the exit code and writes a terminal monitor even for short
scripts. Detach it using the established host launch mechanism when needed.
For an already-running job with its own monitor:

```bash
python scripts/live/notify_watch.py register --id UNIQUE_JOB_ID \
  --thread 01a1af1a-a757-7340-affe-46c38c277e28 --pid PID \
  --monitor /absolute/path/monitor.json --log /absolute/path/job.log
```

Default alerts: terminal states (including awaiting protocol audit), reported
failures, PID exit/reuse, monitor heartbeat stale by 3 minutes, progress unchanged
for 30 minutes, or total runtime over 6 hours. Each alert type is sent once per
job. Change duration limits at registration for known long-running scripts.
Generic wrappers have no domain progress signal; their 30-minute alert is a
request to inspect, not proof of a stall. An existing experiment monitor provides
more useful session progress. Poll interval is 15 seconds.

On a notification: inspect artifacts; audit before external evaluation; report
the outcome; continue the approved follow-up queue within existing spending
limits. Register the next background job. Do not blindly rerun failed paid work.

Validation: six focused tests cover terminal deduplication, failed delivery and
retry, PID reuse, heartbeat-versus-progress, failure alerts, and nonzero exit of
a short wrapped script. Live queue acceptance was tested separately; a received
message in the target thread is the final proof of assistant-side delivery.
The live test message subsequently arrived and triggered an assistant turn;
acknowledgement is recorded in `.wsplab/notifications/delivery-test-ack.json`.
This verifies delivery while the conversation is available, not after app closure.
