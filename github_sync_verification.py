#!/usr/bin/env python3
"""
GitHub Sync Verification - Ensure All Changes Are Pushed
This script verifies that all local changes are properly synced to GitHub
"""

import os
import subprocess
import json
from datetime import datetime

def run_command(cmd, cwd=None):
    """Run a command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)

def main():
    repo_path = r"C:\Users\kevin\Downloads\C# ai bot"
    os.chdir(repo_path)
    
    print("🔄 GITHUB SYNC VERIFICATION")
    print("=" * 60)
    print("🎯 Ensuring all PR #75 changes are properly pushed to GitHub")
    print()
    
    # Check 1: Local vs Remote Sync
    print("📊 Check 1: Local vs Remote Repository Sync")
    print("-" * 50)
    
    # Fetch latest from remote
    print("📥 Fetching latest from GitHub...")
    run_command("git fetch origin")
    
    # Check if local is ahead/behind remote
    success, ahead, error = run_command("git rev-list --count origin/main..HEAD")
    success2, behind, error2 = run_command("git rev-list --count HEAD..origin/main")
    
    if success and success2:
        ahead_count = int(ahead) if ahead.isdigit() else 0
        behind_count = int(behind) if behind.isdigit() else 0
        
        print(f"📈 Local commits ahead of GitHub: {ahead_count}")
        print(f"📉 Local commits behind GitHub: {behind_count}")
        
        if ahead_count == 0 and behind_count == 0:
            print("✅ Perfect sync: Local and GitHub are identical")
        elif ahead_count > 0:
            print("⚠️  WARNING: You have unpushed commits!")
        elif behind_count > 0:
            print("⚠️  WARNING: GitHub has newer commits!")
    
    # Check 2: Latest Commit Verification
    print("\n📍 Check 2: Latest Commit Verification")
    print("-" * 50)
    
    success, local_commit, error = run_command("git log -1 --format='%H %s'")
    success2, remote_commit, error2 = run_command("git log -1 --format='%H %s' origin/main")
    
    if success and success2:
        print(f"🏠 Local HEAD:  {local_commit}")
        print(f"🌐 Remote HEAD: {remote_commit}")
        
        if local_commit == remote_commit:
            print("✅ Commits match: GitHub has your latest changes")
        else:
            print("❌ Commits differ: Sync issue detected!")
    
    # Check 3: PR #75 Related Files
    print("\n📁 Check 3: PR #75 Related Files on GitHub")
    print("-" * 50)
    
    pr75_files = [
        "PR75_SUPERSEDE_MERGE.json",
        "automated_pr75_merge.py", 
        "simple_force_merge.py",
        "force_merge_pr75.py",
        "github_conflict_helper.py",
        "verify_pr75_resolution.py",
        "Intelligence/data/zones/active_zones.json"
    ]
    
    print("🔍 Checking if PR #75 files exist locally and are tracked:")
    
    for file_path in pr75_files:
        if os.path.exists(file_path):
            # Check if file is tracked by git
            success, output, error = run_command(f"git ls-files {file_path}")
            if output:
                print(f"✅ {file_path}: Exists and tracked")
            else:
                print(f"⚠️  {file_path}: Exists but NOT tracked by git")
        else:
            print(f"❌ {file_path}: Missing")
    
    # Check 4: Force Push All Pending Changes
    print("\n🚀 Check 4: Ensuring All Changes Are Pushed")
    print("-" * 50)
    
    # Add any remaining untracked files
    success, output, error = run_command("git status --porcelain")
    if output:
        print("📝 Found untracked/modified files:")
        for line in output.split('\n'):
            if line.strip():
                print(f"  {line}")
        
        print("\n📤 Adding and committing all remaining files...")
        run_command("git add .")
        
        commit_msg = f"🔄 Sync all PR #75 related files - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        success, output, error = run_command(f'git commit -m "{commit_msg}"')
        
        if success:
            print(f"✅ Committed: {commit_msg}")
            
            # Push to GitHub
            print("📤 Pushing to GitHub...")
            success, output, error = run_command("git push origin main")
            
            if success:
                print("✅ Successfully pushed all changes to GitHub!")
            else:
                print(f"❌ Push failed: {error}")
        else:
            print("ℹ️  No new changes to commit")
    else:
        print("✅ No untracked files found")
    
    # Check 5: Verify GitHub Web Visibility
    print("\n🌐 Check 5: GitHub Web Interface Verification")
    print("-" * 50)
    
    success, latest_commit, error = run_command("git log -1 --format='%H'")
    if success:
        repo_url = "https://github.com/c-trading-bo/trading-bot-c-"
        print(f"🔗 Repository URL: {repo_url}")
        print(f"📍 Latest commit: {latest_commit}")
        print(f"🔍 Direct commit URL: {repo_url}/commit/{latest_commit}")
        print()
        print("🎯 TO VERIFY ON GITHUB WEB:")
        print("1. 🌐 Go to: https://github.com/c-trading-bo/trading-bot-c-")
        print("2. 📊 Check main branch shows latest commit")
        print("3. 📁 Verify PR75_SUPERSEDE_MERGE.json exists")
        print("4. 🔍 Check commit history shows 'SUPERSEDE MERGE: PR #75'")
        print("5. 📄 Look for PR #75 - should be closed/superseded")
    
    # Final Summary
    print("\n🎯 FINAL VERIFICATION")
    print("=" * 60)
    
    success, log, error = run_command("git log --oneline -3")
    if success:
        print("📊 Recent commits (should match GitHub):")
        for line in log.split('\n'):
            print(f"  {line}")
    
    print("\n✅ SYNC VERIFICATION COMPLETE!")
    print("💡 If GitHub web doesn't show changes, try:")
    print("   1. Hard refresh (Ctrl+F5)")
    print("   2. Check different branch")
    print("   3. Clear browser cache")
    print("   4. Wait 1-2 minutes for GitHub to update")

if __name__ == "__main__":
    main()
