#!/bin/bash
# Run one TP worker and persist its terminal status for the invoking launcher.
# The status file is diagnostic evidence; this wrapper does not alter worker
# arguments or environment.  It forwards termination so cleanup remains safe.
set -u

status_file=${1:?status file required}
shift
child_pid=
termination_forwarded=0
write_status() {
    rc=$1
    if (( rc >= 128 )); then
        signal=$((rc - 128))
    else
        signal=0
    fi
    printf 'exit_code=%d\nsignal=%d\n' "$rc" "$signal" > "$status_file"
}
# Invoked through the TERM/INT trap below.
# shellcheck disable=SC2329
forward_term() {
    termination_forwarded=1
    if [[ -n ${child_pid:-} ]]; then
        kill -TERM "$child_pid" 2>/dev/null || true
    fi
}
trap forward_term TERM INT

"$@" &
child_pid=$!
wait "$child_pid"
rc=$?
if (( termination_forwarded == 1 )); then
    # A trapped signal interrupts Bash's first wait before the child is
    # necessarily gone. Ignore further supervisor signals and reap the worker
    # before publishing terminal status to the launcher.
    trap '' TERM INT
    child_rc=127
    wait "$child_pid"
    child_rc=$?
    if (( child_rc != 127 )); then
        rc=$child_rc
    fi
fi
child_pid=
write_status "$rc"
exit "$rc"
