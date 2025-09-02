#!/usr/bin/env pwsh
# Cloud Learning Setup Script

Write-Host "Setting up Cloud Learning for your Trading Bot..." -ForegroundColor Cyan

# Check if AWS CLI is available
$awsInstalled = Get-Command aws -ErrorAction SilentlyContinue
if (-not $awsInstalled) {
    Write-Host "AWS CLI not found. Please install it first:" -ForegroundColor Red
    Write-Host "   Download from: https://aws.amazon.com/cli/" -ForegroundColor Yellow
    Write-Host "   Then run: aws configure" -ForegroundColor Yellow
    exit 1
}

# Get cloud bucket name from user
$bucketName = Read-Host "Enter your S3 bucket name (or press Enter for 'trading-bot-ml-$env:USERNAME')"
if ([string]::IsNullOrEmpty($bucketName)) {
    $bucketName = "trading-bot-ml-$env:USERNAME".ToLower()
}

Write-Host "Creating S3 bucket: $bucketName" -ForegroundColor Green

# Create bucket
try {
    aws s3 mb "s3://$bucketName" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Bucket created successfully!" -ForegroundColor Green
    } else {
        Write-Host "Bucket already exists or you don't have permission to create it" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "Could not create bucket. Make sure AWS is configured: aws configure" -ForegroundColor Yellow
}

# Test bucket access
Write-Host "Testing bucket access..." -ForegroundColor Blue
try {
    aws s3 ls "s3://$bucketName" >$null 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Bucket access confirmed!" -ForegroundColor Green
    } else {
        Write-Host "Cannot access bucket. Check AWS permissions." -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "Error accessing bucket" -ForegroundColor Red
    exit 1
}

# Set environment variables
Write-Host "Configuring environment variables..." -ForegroundColor Blue

$envFile = ".env"
if (Test-Path $envFile) {
    $envContent = Get-Content $envFile -Raw
} else {
    $envContent = ""
}

# Add cloud settings to .env
$cloudSettings = @"

# Cloud Learning Configuration
CLOUD_BUCKET=$bucketName
CLOUD_PROVIDER=aws
AWS_DEFAULT_REGION=us-east-1
RL_ENABLED=1

"@

if ($envContent -notmatch "CLOUD_BUCKET") {
    Add-Content -Path $envFile -Value $cloudSettings
    Write-Host "Environment variables added to .env" -ForegroundColor Green
} else {
    Write-Host "Cloud settings already exist in .env" -ForegroundColor Yellow
}

# Create initial training data structure
Write-Host "Setting up training data directories..." -ForegroundColor Blue
New-Item -ItemType Directory -Path "data/rl_training" -Force | Out-Null
New-Item -ItemType Directory -Path "models/rl" -Force | Out-Null

# Sync initial structure to cloud
Write-Host "Syncing initial structure to cloud..." -ForegroundColor Blue
aws s3 sync data/rl_training/ "s3://$bucketName/training-data/" --exclude "*" --include "*.jsonl"
aws s3 sync models/rl/ "s3://$bucketName/models/"

Write-Host ""
Write-Host "Cloud Learning Setup Complete!" -ForegroundColor Green
Write-Host "Bucket: s3://$bucketName" -ForegroundColor White
Write-Host "Environment: Configured for cloud learning" -ForegroundColor White
Write-Host "GitHub Actions: Ready for 24/7 training" -ForegroundColor White
Write-Host ""
Write-Host "What happens next:" -ForegroundColor Yellow
Write-Host "   1. Your bot will upload training data to the cloud" -ForegroundColor White
Write-Host "   2. GitHub Actions will train models every 6 hours" -ForegroundColor White
Write-Host "   3. Your bot will download improved models automatically" -ForegroundColor White
Write-Host "   4. Learning continues even when your computer is off!" -ForegroundColor White
Write-Host ""
Write-Host "To start your bot with cloud learning:" -ForegroundColor Yellow
Write-Host "   .\launch-bot.ps1" -ForegroundColor Cyan
