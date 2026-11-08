# Workspace instructions

Be skeptical. Challenge assumptions and design choices rather than agreeing by default.

The user requested autonomous experiment monitoring and continuation. Every new
background experiment or long-running script must use the notification wrapper
or be registered with `scripts/live/notify_watch.py`; see
`docs/experiment-notifications.md`. Verify the watcher service is alive before
detaching a job. On completion or failure, inspect the outcome and continue the
approved experiment queue without waiting for a manual status prompt. Preserve
existing provider spending limits, protocol audits and held-out data isolation.
Queue acceptance alone is not proof that an assistant turn actually resumed.
