#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Uploads local training logs to S3 for 24/7 cloud learning

.DESCRIPTION
    Monitors local bot training data and uploads to AWS S3 when available.
    Designed to run every 10-15 minutes via Windows Task Scheduler.
    Keeps cloud learning supplied with fresh reality logs when PC is online.

.PARAMETER SourceDir
    Local directory containing training logs (default: data\logs)

.PARAMETER S3Bucket
    S3 bucket name for cloud storage

.PARAMETER DryRun
    Test mode - show what would be uploaded without actually uploading

.EXAMPLE
    .\push-logs.ps1
    
.EXAMPLE
    .\push-logs.ps1 -DryRun

.EXAMPLE
    .\push-logs.ps1 -SourceDir "D:\bot\data\logs" -S3Bucket "my-bot-bucket"
#>

param(
    [string]$SourceDir = "data\logs",
    [string]$S3Bucket = $env:S3_BUCKET,
    [switch]$DryRun = $false
)

# Configuration
$LogFile = "logs\push-logs.log"
$LockFile = "logs\push-logs.lock"
$MaxLogSize = 10MB
$RetentionDays = 7

# Initialize logging
if (!(Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" -Force | Out-Null }

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"
    Write-Host $logEntry -ForegroundColor $(if($Level -eq "ERROR") {"Red"} elseif($Level -eq "WARN") {"Yellow"} else {"Green"})
    $logEntry | Add-Content -Path $LogFile -Encoding UTF8
}

function Test-Requirements {
    # Check if AWS CLI is available
    try {
        $awsVersion = aws --version 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Log "AWS CLI not found. Please install AWS CLI." "ERROR"
            return $false
        }
        Write-Log "AWS CLI found: $($awsVersion.Split()[0])"
    }
    catch {
        Write-Log "AWS CLI not available: $($_.Exception.Message)" "ERROR"
        return $false
    }

    # Check AWS credentials
    try {
        $identity = aws sts get-caller-identity --output json 2>$null | ConvertFrom-Json
        if ($LASTEXITCODE -ne 0) {
            Write-Log "AWS credentials not configured. Run 'aws configure'" "ERROR"
            return $false
        }
        Write-Log "AWS identity: $($identity.Arn)"
    }
    catch {
        Write-Log "Failed to verify AWS credentials: $($_.Exception.Message)" "ERROR"
        return $false
    }

    # Check S3 bucket
    if ([string]::IsNullOrEmpty($S3Bucket)) {
        Write-Log "S3_BUCKET environment variable not set" "ERROR"
        return $false
    }

    return $true
}

function Get-FilesToUpload {
    param([string]$SourcePath)
    
    if (!(Test-Path $SourcePath)) {
        Write-Log "Source directory not found: $SourcePath" "WARN"
        return @()
    }

    $files = @()
    
    # Main training data files
    $candidates = Join-Path $SourcePath "candidates.parquet"
    if (Test-Path $candidates) {
        $lastWrite = (Get-Item $candidates).LastWriteTime
        $age = (Get-Date) - $lastWrite
        if ($age.TotalMinutes -le 30) {  # Only upload if modified in last 30 minutes
            $files += @{
                Local = $candidates
                Remote = "s3://$S3Bucket/logs/candidates.parquet"
                Size = (Get-Item $candidates).Length
                LastWrite = $lastWrite
            }
        }
    }

    # JSONL daily logs
    $jsonlFiles = Get-ChildItem -Path $SourcePath -Filter "*.jsonl" | Where-Object {
        $_.LastWriteTime -gt (Get-Date).AddHours(-2)  # Modified in last 2 hours
    }

    foreach ($file in $jsonlFiles) {
        $files += @{
            Local = $file.FullName
            Remote = "s3://$S3Bucket/logs/$($file.Name)"
            Size = $file.Length
            LastWrite = $file.LastWriteTime
        }
    }

    # Training metrics and state files
    $stateFiles = @("learning_state.json", "self_healing_history.json", "integrity.lock.json")
    foreach ($stateFile in $stateFiles) {
        $statePath = Join-Path $SourcePath "..\state\$stateFile"
        if (Test-Path $statePath) {
            $item = Get-Item $statePath
            if ($item.LastWriteTime -gt (Get-Date).AddHours(-1)) {
                $files += @{
                    Local = $statePath
                    Remote = "s3://$S3Bucket/state/$stateFile"
                    Size = $item.Length
                    LastWrite = $item.LastWriteTime
                }
            }
        }
    }

    return $files
}

function Upload-File {
    param($FileInfo, [bool]$DryRunMode)
    
    try {
        $sizeKB = [math]::Round($FileInfo.Size / 1KB, 2)
        $age = [math]::Round(((Get-Date) - $FileInfo.LastWrite).TotalMinutes, 1)
        
        if ($DryRunMode) {
            Write-Log "[DRY-RUN] Would upload: $($FileInfo.Local) -> $($FileInfo.Remote) (${sizeKB}KB, ${age}m old)"
            return $true
        }

        Write-Log "Uploading: $($FileInfo.Local) -> $($FileInfo.Remote) (${sizeKB}KB, ${age}m old)"
        
        # Upload with retry logic
        $maxRetries = 3
        $retryCount = 0
        $success = $false
        
        while ($retryCount -lt $maxRetries -and !$success) {
            $retryCount++
            
            # Use AWS CLI to upload
            $result = aws s3 cp "$($FileInfo.Local)" "$($FileInfo.Remote)" --no-progress 2>&1
            
            if ($LASTEXITCODE -eq 0) {
                $success = $true
                Write-Log "✅ Upload successful (attempt $retryCount)"
            }
            else {
                Write-Log "❌ Upload failed (attempt $retryCount): $result" "WARN"
                if ($retryCount -lt $maxRetries) {
                    $delay = [math]::Pow(2, $retryCount)  # Exponential backoff
                    Write-Log "Retrying in $delay seconds..."
                    Start-Sleep -Seconds $delay
                }
            }
        }
        
        if (!$success) {
            Write-Log "Upload failed after $maxRetries attempts: $($FileInfo.Local)" "ERROR"
            return $false
        }
        
        return $true
    }
    catch {
        Write-Log "Upload exception: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

function Cleanup-Logs {
    try {
        # Rotate log file if too large
        if ((Test-Path $LogFile) -and (Get-Item $LogFile).Length -gt $MaxLogSize) {
            $backupLog = $LogFile -replace "\.log$", "_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
            Move-Item $LogFile $backupLog
            Write-Log "Rotated log file to: $backupLog"
        }

        # Clean old backup logs
        $oldLogs = Get-ChildItem -Path "logs" -Filter "*_*.log" | Where-Object {
            $_.LastWriteTime -lt (Get-Date).AddDays(-$RetentionDays)
        }
        
        foreach ($oldLog in $oldLogs) {
            Remove-Item $oldLog.FullName -Force
            Write-Log "Deleted old log: $($oldLog.Name)"
        }
    }
    catch {
        Write-Log "Cleanup failed: $($_.Exception.Message)" "WARN"
    }
}

function Main {
    # Single instance lock
    if (Test-Path $LockFile) {
        $lockAge = (Get-Date) - (Get-Item $LockFile).LastWriteTime
        if ($lockAge.TotalMinutes -lt 10) {
            Write-Log "Another instance is running (lock age: $($lockAge.TotalMinutes.ToString('F1'))m)" "WARN"
            exit 1
        }
        else {
            Write-Log "Removing stale lock file (age: $($lockAge.TotalMinutes.ToString('F1'))m)" "WARN"
            Remove-Item $LockFile -Force
        }
    }
    
    try {
        # Create lock file
        "PID: $PID, Started: $(Get-Date)" | Set-Content $LockFile
        
        Write-Log "🚀 Starting log upload process (PID: $PID)"
        if ($DryRun) { Write-Log "🧪 DRY RUN MODE - No actual uploads" "WARN" }
        
        # Pre-flight checks
        if (!(Test-Requirements)) {
            Write-Log "Requirements check failed" "ERROR"
            exit 1
        }
        
        # Find files to upload
        $filesToUpload = Get-FilesToUpload -SourcePath $SourceDir
        
        if ($filesToUpload.Count -eq 0) {
            Write-Log "📂 No files need uploading"
            exit 0
        }
        
        Write-Log "📁 Found $($filesToUpload.Count) files to upload"
        
        # Upload files
        $successCount = 0
        $totalSize = 0
        
        foreach ($file in $filesToUpload) {
            $totalSize += $file.Size
            if (Upload-File -FileInfo $file -DryRunMode $DryRun) {
                $successCount++
            }
        }
        
        $totalSizeKB = [math]::Round($totalSize / 1KB, 2)
        
        if ($DryRun) {
            Write-Log "🧪 DRY RUN: Would upload $successCount/$($filesToUpload.Count) files (${totalSizeKB}KB total)"
        }
        else {
            Write-Log "✅ Uploaded $successCount/$($filesToUpload.Count) files (${totalSizeKB}KB total)"
            
            if ($successCount -eq $filesToUpload.Count) {
                Write-Log "🎯 All uploads successful - cloud learning data updated!"
            }
            else {
                Write-Log "⚠️ Some uploads failed - check logs for details" "WARN"
            }
        }
    }
    catch {
        Write-Log "Unexpected error: $($_.Exception.Message)" "ERROR"
        exit 1
    }
    finally {
        # Clean up
        if (Test-Path $LockFile) { Remove-Item $LockFile -Force }
        Cleanup-Logs
    }
}

# Entry point
if ($MyInvocation.InvocationName -ne '.') {
    Main
}
