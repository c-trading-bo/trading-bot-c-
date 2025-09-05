# GitHub Token Setup Guide

## Step 1: Create Personal Access Token

1. **Go to GitHub Settings:**
   - Visit: https://github.com/settings/tokens
   - Or: GitHub → Profile → Settings → Developer settings → Personal access tokens → Tokens (classic)

2. **Generate New Token:**
   - Click "Generate new token (classic)"
   - Name: `Workflow Monitor Token`
   - Expiration: 90 days (or your preference)

3. **Required Permissions (Scopes):**
   ```
   ✅ repo (Full control of private repositories)
   ✅ workflow (Update GitHub Action workflows)
   ✅ read:org (Read org and team membership)
   ```

4. **Generate and Copy:**
   - Click "Generate token"
   - **COPY THE TOKEN IMMEDIATELY** (you won't see it again!)
   - Format: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

## Step 2: Set Environment Variable

### Option A: Windows Environment Variable (Permanent)
```powershell
# Run as Administrator
setx GITHUB_TOKEN "your_token_here" /M
```

### Option B: PowerShell Session (Temporary)
```powershell
$env:GITHUB_TOKEN = "your_token_here"
```

### Option C: Create .env File (Recommended)
```bash
# Create .env file in your project root
GITHUB_TOKEN=your_token_here
```

## Step 3: Test Token
```powershell
# Test if token works
curl -H "Authorization: token $env:GITHUB_TOKEN" https://api.github.com/user
```

## Step 4: Run Monitoring
```powershell
cd "C:\Users\kevin\Downloads\C# ai bot"
python realtime_workflow_monitor.py
```

## Security Notes
- ⚠️ Never commit tokens to Git
- 🔒 Keep tokens secure and private
- 🔄 Rotate tokens regularly
- 📝 Add .env to .gitignore

## Token Permissions Explained
- **repo**: Access to repository data and actions
- **workflow**: Read workflow runs and status
- **read:org**: Required for organization repositories

## Troubleshooting
- Rate limit: 5,000 requests/hour with token
- Invalid token: Check expiration and permissions
- 403 errors: Verify repository access permissions
