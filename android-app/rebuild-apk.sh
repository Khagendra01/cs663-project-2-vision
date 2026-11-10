#!/bin/bash
set -e
export JAVA_HOME=~/jdk/jdk-21.0.3+9
export ANDROID_HOME=~/android-sdk
export GRADLE_USER_HOME=/home/hatch/.gradle
export PATH=$JAVA_HOME/bin:$PATH
export GRADLE_OPTS="-Djava.net.preferIPv4Stack=true"
cd ~/workspace/goals/cs663-project-2-dog-training-assistant/android-app
~/gradle/gradle-8.7/bin/gradle assembleDebug --console=plain --offline "$@"
echo "APK: $(find app/build -name '*.apk' | head -1)"
