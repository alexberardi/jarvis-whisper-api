#!/bin/bash

./setup-python.sh
./setup-whisper-cpp.sh


# Register native launch agent (macOS)
PLIST=~/Library/LaunchAgents/com.jarvis.whisper.plist
cat <<EOF > \$PLIST
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
</dict>
</plist>
EOF

launchctl load \$PLIST
