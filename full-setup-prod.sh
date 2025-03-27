#!/bin/bash

./setup-python.sh
./setup-whisper-cpp.sh


# Register native launch agent (macOS)
PLIST=~/Library/LaunchAgents/com.jarvis.whisper.plist
cat <<EOF > $PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.jarvis.whisper</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PWD}/run.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>${PWD}</string>
    <key>StandardOutPath</key>
    <string>${PWD}/whisper.log</string>
    <key>StandardErrorPath</key>
    <string>${PWD}/whisper.err</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
EOF

# Remove previous instance if it exists
launchctl bootout gui/$(id -u) $PLIST 2>/dev/null || true

# Register
launchctl bootstrap gui/$(id -u) $PLIST

