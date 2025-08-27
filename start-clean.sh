#!/bin/bash

CHECKLIST_ONLY=false
if [ "$1" = "--checklist-only" ] || [ "$1" = "-c" ]; then
    CHECKLIST_ONLY=true
fi

cd "$(dirname "$0")"

export "Logging__Console__FormatterName=simple"
export "Logging__Console__FormatterOptions__SingleLine=true"
export "Logging__Console__FormatterOptions__TimestampFormat=HH:mm:ss "

export "Logging__LogLevel__Default=Warning"
export "Logging__LogLevel__Microsoft=Warning"
export "Logging__LogLevel__Microsoft__AspNetCore=Warning"
export "Logging__LogLevel__Microsoft__AspNetCore__SignalR=Warning"
export "Logging__LogLevel__Microsoft__AspNetCore__Http__Connections__Client=Warning"

export "Logging__LogLevel__Orchestrator=Information"
export "Logging__LogLevel__SupervisorAgent__StatusService=Information"   # BOT STATUS => {...}
export "Logging__LogLevel__BotCore__UserHubAgent=Information"            # UserHub/JWT/connect
export "Logging__LogLevel__BotCore__ApiClient=Information"               # contract picks
export "Logging__LogLevel__BotCore__MarketHubClient=Information"         # stream connects (we filter later)

if [ "$CHECKLIST_ONLY" = true ]; then
    export BOT_QUICK_EXIT='1'
else
    unset BOT_QUICK_EXIT
fi

BOOT_KEEP='(Env config|Using URL|UserHub connected|Available pick|Preflight|MODE =>|BOT STATUS)'

PROMOTE='(MODE\s*=>\s*(LIVE|AUTOPILOT|TRADING)|Promoted|Quotes ready|Healthy|Preflight.*pass)'

RUN_KEEP='(order|fill|trade|position|risk|strategy|looking\s+for\s+a\s+trade|entry|exit|BOT STATUS|lastQuote|lastTrade)'

echo "--------------------------------------------------------------------------------"
echo "$(date '+%H:%M:%S') LAUNCH (clean view)…"
echo "--------------------------------------------------------------------------------"

PHASE='boot'
bash ./launch-bot.sh 2>&1 | while IFS= read -r line; do
    if [ "$PHASE" = 'boot' ]; then
        if echo "$line" | grep -qE "$BOOT_KEEP"; then
            echo "$line"
        fi
        if echo "$line" | grep -qE "$PROMOTE"; then
            echo "--------------------------------------------------------------------------------"
            echo "$(date '+%H:%M:%S') READY: switching to RUN view (quotes & trades only)…"
            echo "--------------------------------------------------------------------------------"
            PHASE='run'
        fi
    else
        if echo "$line" | grep -qE "$RUN_KEEP"; then
            echo "$line"
        fi
    fi
done
