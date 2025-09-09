#!/bin/bash

echo "=== Trading Bot Environment & Credential Test ==="
echo "=================================================="
echo

# Change to bot directory
cd "$(dirname "$0")"

echo "1. Testing environment file loading..."
echo "--------------------------------------"

# Create temporary test to verify .env loading
cat > /tmp/quick_env_test.cs << 'EOF'
using System;
using System.IO;

class QuickTest {
    static void Main() {
        LoadDotEnv();
        var topstepUsername = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME");
        var topstepJwt = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
        var alertEmail = Environment.GetEnvironmentVariable("ALERT_EMAIL_FROM");
        var alertPassword = Environment.GetEnvironmentVariable("ALERT_EMAIL_PASSWORD");
        
        Console.WriteLine($"✓ TopstepX Username: {(string.IsNullOrEmpty(topstepUsername) ? "❌ NOT SET" : "✅ " + topstepUsername)}");
        Console.WriteLine($"✓ TopstepX JWT: {(string.IsNullOrEmpty(topstepJwt) ? "❌ NOT SET" : "✅ Present")}");
        Console.WriteLine($"✓ Alert Email: {(string.IsNullOrEmpty(alertEmail) ? "❌ NOT SET" : "✅ " + alertEmail)}");
        Console.WriteLine($"✓ Gmail Password: {(string.IsNullOrEmpty(alertPassword) || alertPassword == "YOUR_GMAIL_APP_PASSWORD_HERE" ? "❌ NOT SET" : "✅ Configured")}");
    }
    
    static void LoadDotEnv() {
        try {
            var candidates = new[] { ".env.local", ".env" };
            string dir = Environment.CurrentDirectory;
            for (int up = 0; up < 5 && dir != null; up++) {
                foreach (var file in candidates) {
                    var path = Path.Combine(dir, file);
                    if (File.Exists(path)) {
                        foreach (var raw in File.ReadAllLines(path)) {
                            var line = raw.Trim();
                            if (line.Length == 0 || line.StartsWith("#")) continue;
                            var idx = line.IndexOf('=');
                            if (idx <= 0) continue;
                            var key = line.Substring(0, idx).Trim();
                            var val = line.Substring(idx + 1).Trim();
                            if (!string.IsNullOrWhiteSpace(key)) Environment.SetEnvironmentVariable(key, val);
                        }
                        return;
                    }
                }
                dir = Directory.GetParent(dir)?.FullName;
            }
        } catch { }
    }
}
EOF

# Create project and run test
mkdir -p /tmp/quick_test
cd /tmp/quick_test
echo '<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net8.0</TargetFramework></PropertyGroup></Project>' > test.csproj
cp /tmp/quick_env_test.cs Program.cs
cd - > /dev/null

echo "Running credential verification..."
dotnet run --project /tmp/quick_test/test.csproj --verbosity quiet 2>/dev/null
echo

echo "2. Testing bot build and startup (10 seconds)..."
echo "-----------------------------------------------"

# Test OrchestratorAgent startup
timeout 10 dotnet run --project src/OrchestratorAgent/OrchestratorAgent.csproj 2>&1 | grep -E "(Env config|AlertService|TOPSTEP|error|Error)" | head -5
echo

echo "3. Environment File Status"
echo "-------------------------"
if [ -f ".env" ]; then
    echo "✅ .env file exists"
    echo "   Lines: $(wc -l < .env)"
    echo "   Size: $(du -h .env | cut -f1)"
else
    echo "❌ .env file missing"
fi

if [ -f ".env.example" ]; then
    echo "✅ .env.example template exists"
else
    echo "❌ .env.example template missing"
fi

echo

echo "4. Cleanup Check"
echo "---------------"
for deprecated_file in ".env.github" ".env.test" ".env.sample.local" ".github/copilot_mechanic/.env"; do
    if [ -f "$deprecated_file" ]; then
        echo "❌ Deprecated file still exists: $deprecated_file"
    else
        echo "✅ Removed: $deprecated_file"
    fi
done

echo
echo "=== Test Complete ==="
echo "Next Steps:"
echo "1. Update ALERT_EMAIL_PASSWORD in .env with your Gmail App Password"
echo "2. Run: make test-alert-email"
echo "3. Run: dotnet run --project src/OrchestratorAgent/OrchestratorAgent.csproj"