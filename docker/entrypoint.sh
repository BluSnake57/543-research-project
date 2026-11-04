#!/bin/bash
set -euo pipefail
umask 077

size=${DESKTOP_SIZE:-1600x900}
if [[ ! $size =~ ^[0-9]{3,4}x[0-9]{3,4}$ ]]; then
    echo 'DESKTOP_SIZE must be WIDTHxHEIGHT, e.g. 1600x900.' >&2
    exit 1
fi
export DESKTOP_SIZE=$size

password_file=/run/secrets/vnc_password
if [[ ! -f $password_file || ! -r $password_file ]]; then
    echo 'Missing VNC secret. Run python3 scripts/create_container_secret.py first.' >&2
    exit 1
fi
password=$(cat "$password_file")
if [[ ! $password =~ ^[A-Za-z0-9]{8}$ ]]; then
    echo 'The VNC secret must contain exactly eight ASCII letters or digits.' >&2
    exit 1
fi
x11vnc -storepasswd "$password" /tmp/vnc.pass >/dev/null
unset password

openssl passwd -6 -stdin < "$password_file" | sed 's/^/sentinel:/' > /tmp/downloads.htpasswd

mkdir -p "$XDG_RUNTIME_DIR" "$HOME/Desktop" "$HOME/Exports"
chmod 700 "$XDG_RUNTIME_DIR"
cp /opt/container/sentinel-x.desktop "$HOME/Desktop/Sentinel-X.desktop"
cp /opt/container/sentinel-x-trainer.desktop "$HOME/Desktop/Sentinel-X-Trainer.desktop"
chmod 700 "$HOME/Desktop/"*.desktop
ln -sfn "$HOME/Exports" "$HOME/Desktop/Exports"
python scripts/setup_environment.py --configure-only

exec dbus-run-session -- supervisord -c /opt/container/supervisord.conf
