#!/bin/zsh
# hermes-voice 상주 서버(launchd) 설치/재시작. 사용: deploy/install.sh [install|restart|stop|status|serve]
#   install  deploy/hermes-voice.plist.template 의 자리표시자를 채워 ~/Library/LaunchAgents/hermes-voice.plist 로 쓰고 시작
#   serve    tailscale serve 로 https://<tailscale-host>:8464 -> http://127.0.0.1:8480 연결
# 경로는 이 스크립트 위치에서 구한다(프로젝트 폴더를 어디 두든 됨). 개인 설정(API 키·모델 경로 등)은 .env 에.
# 환경변수: TS_HOST(Tailscale DNS 이름, 없으면 tailscale status 에서 자동), TAILSCALE(바이너리 경로)
set -e
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LABEL=hermes-voice
AGENTS="$HOME/Library/LaunchAgents"
PLIST="$AGENTS/$LABEL.plist"
SRC="$PROJECT_DIR/deploy/$LABEL.plist.template"
DOMAIN="gui/$(id -u)"
PORT=8480
HTTPS_PORT=8464

find_tailscale() {
  if [[ -n "$TAILSCALE" ]]; then echo "$TAILSCALE"; return; fi
  if command -v tailscale >/dev/null 2>&1; then command -v tailscale; return; fi
  local app=/Applications/Tailscale.app/Contents/MacOS/Tailscale
  if [[ -x "$app" ]]; then echo "$app"; return; fi
  echo "tailscale 바이너리를 찾지 못함(TAILSCALE=경로 로 지정)" >&2; return 1
}

ts_host() {
  if [[ -n "$TS_HOST" ]]; then echo "$TS_HOST"; return; fi
  local ts; ts="$(find_tailscale)" || return 1
  "$ts" status --json 2>/dev/null | "$PROJECT_DIR/.venv/bin/python" -c \
    'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null \
    || { echo "Tailscale 호스트명을 알 수 없음(TS_HOST=이름 으로 지정)" >&2; return 1; }
}

case "${1:-install}" in
  install)
    [[ -x "$PROJECT_DIR/.venv/bin/python" ]] || { echo "먼저 venv를 만드세요: $PROJECT_DIR/.venv" >&2; exit 1; }
      mkdir -p "$AGENTS" "$HOME/Library/Logs"
    launchctl bootout "$DOMAIN" "$PLIST" 2>/dev/null || true
    # 경로에 | 나 & 가 없다고 가정(sed 구분자 |)
    sed -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" -e "s|__HOME__|$HOME|g" "$SRC" > "$PLIST"
    plutil -lint "$PLIST" >/dev/null
    launchctl bootstrap "$DOMAIN" "$PLIST"
    echo "설치·시작: $LABEL ($PLIST, 로그 ~/Library/Logs/hermes-voice.log)";;
  restart) launchctl kickstart -k "$DOMAIN/$LABEL"; echo "재시작";;
  stop) launchctl bootout "$DOMAIN" "$PLIST"; echo "중지";;
  status) launchctl print "$DOMAIN/$LABEL" 2>/dev/null | grep -E "state|pid" ; curl -s -m 3 "http://127.0.0.1:$PORT/health"; echo;;
  serve)
    # 재부팅 뒤 serve 설정이 사라지는 경우가 있음. 그때 이걸 다시 실행.
    TS="$(find_tailscale)"
    "$TS" serve --bg --https=$HTTPS_PORT "http://127.0.0.1:$PORT"
    "$TS" serve status | grep -A1 $HTTPS_PORT || true
    HOST="$(ts_host)" && echo "접속: https://$HOST:$HTTPS_PORT";;
  *) echo "사용: $0 [install|restart|stop|status|serve]"; exit 2;;
esac
