#!/bin/zsh
# 安裝/移除 launchd 排程: 平日 08:30 更新 D1, 09:01 產生名單.  用法: install_schedule.sh [install|uninstall|status]
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$(which python3)"
AGENTS="$HOME/Library/LaunchAgents"
mkdir -p "$AGENTS" "$ROOT/logs"

write_plist() {  # name hour minute args...
  local name=$1 hour=$2 minute=$3; shift 3
  local args=""; for a in "$@"; do args+="<string>$a</string>"; done
  cat > "$AGENTS/$name.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$name</string>
  <key>ProgramArguments</key><array><string>$PY</string><string>-u</string><string>$ROOT/scripts/daily_candidates.py</string>$args</array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>StartCalendarInterval</key><array>
    <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>$hour</integer><key>Minute</key><integer>$minute</integer></dict>
    <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>$hour</integer><key>Minute</key><integer>$minute</integer></dict>
    <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>$hour</integer><key>Minute</key><integer>$minute</integer></dict>
    <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>$hour</integer><key>Minute</key><integer>$minute</integer></dict>
    <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>$hour</integer><key>Minute</key><integer>$minute</integer></dict>
  </array>
  <key>StandardOutPath</key><string>$ROOT/logs/$name.log</string>
  <key>StandardErrorPath</key><string>$ROOT/logs/$name.log</string>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>$(dirname "$PY"):/usr/bin:/bin:/usr/local/bin</string><key>PYTHONWARNINGS</key><string>ignore</string></dict>
</dict></plist>
PL
}

case "${1:-install}" in
  install)
    write_plist com.quant.refresh-d1 8 30 --refresh
    write_plist com.quant.gen-signals 9 1
    for n in com.quant.refresh-d1 com.quant.gen-signals; do
      launchctl bootout "gui/$(id -u)/$n" 2>/dev/null || true
      launchctl bootstrap "gui/$(id -u)" "$AGENTS/$n.plist"
    done
    echo "已安裝: 平日 08:30 更新 D1 (com.quant.refresh-d1), 09:01 產生名單 (com.quant.gen-signals)"; echo "log: $ROOT/logs/";;
  uninstall)
    for n in com.quant.refresh-d1 com.quant.gen-signals; do launchctl bootout "gui/$(id -u)/$n" 2>/dev/null || true; rm -f "$AGENTS/$n.plist"; done; echo "已移除";;
  status)
    launchctl list | grep com.quant || echo "未安裝";;
esac
