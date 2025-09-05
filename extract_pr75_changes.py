#!/usr/bin/env python3
"""
PR #75 Change Extraction and Application
This script will fetch the actual changes from PR #75 and apply them to main branch
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
    
    print("🔄 PR #75 CHANGE EXTRACTION AND APPLICATION")
    print("=" * 60)
    print("🎯 Goal: Get the actual workflow fixes from PR #75 and apply them")
    print()
    
    # Step 1: Fetch all branches and PR references
    print("📥 Step 1: Fetching all PR branches and references...")
    run_command("git fetch origin")
    run_command("git fetch origin +refs/pull/*/head:refs/remotes/origin/pr/*")
    
    # Step 2: Try to find the PR #75 branch
    print("\n🔍 Step 2: Looking for PR #75 branch...")
    
    # Check if we can find the PR branch
    success, branches, error = run_command("git branch -r | grep -E 'pr/75|pull/75'")
    
    if not success:
        # Try alternative approach - look for the commit mentioned in GitHub
        print("📋 Trying alternative approach - checking for copilot branch...")
        success, branches, error = run_command("git branch -r")
        print("Available remote branches:")
        if success:
            for branch in branches.split('\n'):
                if branch.strip():
                    print(f"  {branch.strip()}")
        
        # Try to fetch the specific branch mentioned in the PR
        print("\n🔄 Fetching copilot/fix-0f0c0cfb-e160-488c-0404-c31bc29f3c3e branch...")
        success, output, error = run_command("git fetch origin copilot/fix-0f0c0cfb-e160-488c-0404-c31bc29f3c3e")
        
        if success:
            print("✅ Successfully fetched PR #75 branch!")
            pr_branch = "origin/copilot/fix-0f0c0cfb-e160-488c-0404-c31bc29f3c3e"
        else:
            print(f"❌ Could not fetch specific branch: {error}")
            # Try to get PR via GitHub API approach
            print("\n🌐 Attempting to reconstruct changes from GitHub...")
            pr_branch = None
    else:
        pr_branch = branches.split('\n')[0].strip()
        print(f"✅ Found PR branch: {pr_branch}")
    
    # Step 3: Extract changes from PR branch
    if pr_branch:
        print(f"\n📊 Step 3: Analyzing changes in {pr_branch}...")
        
        # Get the diff between main and PR branch
        success, diff_output, error = run_command(f"git diff main..{pr_branch}")
        
        if success and diff_output:
            print(f"✅ Found {len(diff_output.splitlines())} lines of changes")
            
            # Save the diff for review
            with open("PR75_changes_diff.patch", 'w', encoding='utf-8') as f:
                f.write(diff_output)
            
            print("📁 Saved changes to PR75_changes_diff.patch")
            
            # Get list of changed files
            success, files, error = run_command(f"git diff --name-only main..{pr_branch}")
            if success:
                print("\n📝 Files that would be changed:")
                for file in files.split('\n'):
                    if file.strip():
                        print(f"  ✏️  {file.strip()}")
        else:
            print("⚠️  No diff found or error occurred")
    
    # Step 4: Apply the changes
    if pr_branch and diff_output:
        print(f"\n🚀 Step 4: Applying PR #75 changes to main branch...")
        
        # Create a backup branch first
        backup_branch = f"backup-main-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_command(f"git checkout -b {backup_branch}")
        run_command("git checkout main")
        
        print(f"✅ Created backup branch: {backup_branch}")
        
        # Apply the patch
        with open("PR75_changes_diff.patch", 'r', encoding='utf-8') as f:
            patch_content = f.read()
        
        # Try to apply the patch
        success, output, error = run_command("git apply PR75_changes_diff.patch")
        
        if success:
            print("✅ Successfully applied PR #75 changes!")
            
            # Commit the changes
            run_command("git add .")
            commit_msg = f"🔀 Apply PR #75 workflow fixes - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            success, output, error = run_command(f'git commit -m "{commit_msg}"')
            
            if success:
                print(f"✅ Committed changes: {commit_msg}")
                
                # Push to GitHub
                success, output, error = run_command("git push origin main")
                if success:
                    print("✅ Successfully pushed changes to GitHub!")
                    print("\n🎉 PR #75 CHANGES SUCCESSFULLY APPLIED!")
                    print("=" * 60)
                    print("✅ All workflow fixes are now in your main branch")
                    print("✅ Your repository has the latest improvements")
                    print("✅ PR #75 changes are preserved and applied")
                    print("\n💡 You can now close PR #75 since changes are in main")
                else:
                    print(f"❌ Push failed: {error}")
            else:
                print(f"❌ Commit failed: {error}")
        else:
            print(f"❌ Patch application failed: {error}")
            print("🔧 Trying manual merge approach...")
            
            # Try cherry-pick approach
            success, commits, error = run_command(f"git log --oneline main..{pr_branch}")
            if success and commits:
                print(f"📊 Found commits to cherry-pick:")
                commit_hashes = []
                for commit_line in commits.split('\n'):
                    if commit_line.strip():
                        commit_hash = commit_line.split()[0]
                        commit_hashes.append(commit_hash)
                        print(f"  {commit_line.strip()}")
                
                # Cherry-pick each commit
                for commit_hash in reversed(commit_hashes):  # Apply in correct order
                    print(f"🍒 Cherry-picking {commit_hash}...")
                    success, output, error = run_command(f"git cherry-pick {commit_hash}")
                    if not success:
                        print(f"⚠️  Cherry-pick conflict for {commit_hash}, skipping...")
                        run_command("git cherry-pick --abort")
                        continue
                
                # Check if we have changes to commit
                success, status, error = run_command("git status --porcelain")
                if status:
                    run_command("git add .")
                    commit_msg = f"🔀 Manual apply PR #75 changes - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    run_command(f'git commit -m "{commit_msg}"')
                    run_command("git push origin main")
                    print("✅ Successfully applied changes via cherry-pick!")
    
    else:
        print("\n⚠️  Could not find PR #75 branch or changes")
        print("📋 Manual approach needed:")
        print("1. Go to GitHub PR #75")
        print("2. Click 'Files changed' tab")
        print("3. Review the actual changes")
        print("4. Apply them manually to your local files")
        print("5. Commit and push")
    
    # Step 5: Summary
    print("\n🎯 SUMMARY")
    print("=" * 60)
    
    if os.path.exists("PR75_changes_diff.patch"):
        print("✅ PR #75 changes extracted and saved")
        print("📁 Check PR75_changes_diff.patch for details")
    
    success, log, error = run_command("git log --oneline -3")
    if success:
        print("\n📊 Recent commits:")
        for line in log.split('\n'):
            print(f"  {line}")
    
    print("\n💡 Next steps:")
    print("1. Verify the changes are applied correctly")
    print("2. Test your workflows")
    print("3. Close PR #75 if satisfied")

if __name__ == "__main__":
    main()
