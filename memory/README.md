`memory/profiles/{user_id}.json` stores long-lived profile data and is read at session start.
`memory/sessions/{session_id}.json` stores turn history and is write-generated when a session ends.
Profiles are read and occasionally write-updated when stable preferences change.
Sessions are write-only runtime artifacts and remain out of git history.