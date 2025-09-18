#!/usr/bin/env python3
"""
GitHub Actions Session Deduplicator
Prevents multiple agent sessions from launching per commit or PR.
Implements session existence checks and event deduplication.
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta, UTC
from pathlib import Path

class SessionDeduplicator:
    def __init__(self):
        self.session_dir = Path("Intelligence/data/mechanic/active_sessions")
        self.audit_dir = Path("Intelligence/data/mechanic/audit")
        self.lock_dir = Path("Intelligence/data/mechanic")
        
        # Ensure directories exist
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.lock_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_session_key(self, event_type, workflow_name, commit_sha, run_id=None):
        """Generate unique session key for deduplication"""
        # Truncate commit SHA to 8 characters for brevity
        short_sha = commit_sha[:8] if commit_sha else "unknown"
        
        # Clean workflow name for use in file name
        clean_workflow = workflow_name.replace(" ", "-").replace("/", "-").lower()
        
        return f"{clean_workflow}-{event_type}-{short_sha}"
    
    def check_active_session(self, session_key, max_age_minutes=5):
        """Check if session is already active within time window"""
        session_file = self.session_dir / f"{session_key}.json"
        
        if not session_file.exists():
            return False, "No active session found"
        
        # Check if session is still active
        try:
            file_time = session_file.stat().st_mtime
            current_time = time.time()
            age_minutes = (current_time - file_time) / 60
            
            if age_minutes < max_age_minutes:
                return True, f"Active session found ({age_minutes:.1f}m ago)"
            else:
                # Session expired, clean it up
                session_file.unlink(missing_ok=True)
                return False, f"Previous session expired ({age_minutes:.1f}m ago)"
        except Exception as e:
            return False, f"Error checking session: {e}"
    
    def check_push_pr_deduplication(self, event_type, commit_sha):
        """Check for push+PR double trigger on same commit"""
        short_sha = commit_sha[:8] if commit_sha else "unknown"
        pr_lock_file = self.lock_dir / f"recent_pr_{short_sha}.lock"
        
        if event_type == "push" and pr_lock_file.exists():
            # Check if PR lock is recent (within 1 hour)
            try:
                file_time = pr_lock_file.stat().st_mtime
                current_time = time.time()
                age_minutes = (current_time - file_time) / 60
                
                if age_minutes < 60:  # 1 hour
                    return True, f"PR already handled this commit ({age_minutes:.1f}m ago)"
                else:
                    # Old lock, clean it up
                    pr_lock_file.unlink(missing_ok=True)
            except Exception:
                pass
        
        return False, "No PR duplication detected"
    
    def register_session(self, session_key, event_type, workflow_name, run_id, commit_sha):
        """Register new session"""
        session_file = self.session_dir / f"{session_key}.json"
        
        session_data = {
            "session_key": session_key,
            "start_time": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "workflow": workflow_name,
            "run_id": run_id,
            "commit_sha": commit_sha
        }
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        # Create PR lock if this is a PR event
        if event_type == "pull_request":
            short_sha = commit_sha[:8] if commit_sha else "unknown"
            pr_lock_file = self.lock_dir / f"recent_pr_{short_sha}.lock"
            pr_lock_file.touch()
        
        return session_data
    
    def cleanup_old_sessions(self, max_age_minutes=10):
        """Clean up old session files"""
        current_time = time.time()
        cleaned = 0
        
        for session_file in self.session_dir.glob("*.json"):
            try:
                file_time = session_file.stat().st_mtime
                age_minutes = (current_time - file_time) / 60
                
                if age_minutes > max_age_minutes:
                    session_file.unlink()
                    cleaned += 1
            except Exception:
                pass
        
        # Clean up old PR locks (older than 1 hour)
        for lock_file in self.lock_dir.glob("recent_pr_*.lock"):
            try:
                file_time = lock_file.stat().st_mtime
                age_minutes = (current_time - file_time) / 60
                
                if age_minutes > 60:  # 1 hour
                    lock_file.unlink()
                    cleaned += 1
            except Exception:
                pass
        
        return cleaned
    
    def create_audit_entry(self, session_key, event_type, workflow_name, run_id, commit_sha, executed, job_status, duplicate_prevented):
        """Create audit log entry with retry suppression tracking"""
        audit_file = self.audit_dir / f"session_audit_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Enhanced audit entry with retry suppression status
        entry = {
            "session_key": session_key,
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "workflow": workflow_name,
            "run_id": run_id,
            "commit_sha": commit_sha,
            "executed": executed,
            "job_status": job_status,
            "duplicate_prevented": duplicate_prevented,
            "retry_suppression": {
                "enabled": True,
                "status": "suppressed" if duplicate_prevented else "single_execution",
                "early_gating": True,
                "cost_optimization": duplicate_prevented
            },
            "session_management": {
                "early_exit": duplicate_prevented,
                "single_launch_enforced": True,
                "premium_session_saved": duplicate_prevented
            }
        }
        
        # Append to audit log
        try:
            if audit_file.exists():
                with open(audit_file, 'r') as f:
                    audit_data = json.load(f)
            else:
                audit_data = []
            
            audit_data.append(entry)
            
            with open(audit_file, 'w') as f:
                json.dump(audit_data, f, indent=2)
        
        except Exception as e:
            print(f"Warning: Could not write audit log: {e}")
        
        return entry
    
    def cleanup_session(self, session_key):
        """Clean up specific session file"""
        session_file = self.session_dir / f"{session_key}.json"
        if session_file.exists():
            try:
                session_file.unlink()
                return True
            except Exception:
                return False
        return False

def main():
    """Main CLI interface for session deduplication"""
    if len(sys.argv) < 2:
        print("Usage: session_deduplicator.py <command> [args...]")
        print("Commands:")
        print("  check <event_type> <workflow_name> <commit_sha> [run_id]")
        print("  register <session_key> <event_type> <workflow_name> <run_id> <commit_sha>")
        print("  cleanup <session_key>")
        print("  audit <session_key> <event_type> <workflow_name> <run_id> <commit_sha> <executed> <job_status>")
        sys.exit(1)
    
    deduplicator = SessionDeduplicator()
    command = sys.argv[1]
    
    if command == "check":
        if len(sys.argv) < 5:
            print("Usage: check <event_type> <workflow_name> <commit_sha> [run_id]")
            sys.exit(1)
        
        event_type = sys.argv[2]
        workflow_name = sys.argv[3]
        commit_sha = sys.argv[4]
        run_id = sys.argv[5] if len(sys.argv) > 5 else None
        
        session_key = deduplicator.generate_session_key(event_type, workflow_name, commit_sha, run_id)
        
        # Check for active session
        is_active, reason = deduplicator.check_active_session(session_key)
        
        # Check for push+PR deduplication
        is_duplicate, dup_reason = deduplicator.check_push_pr_deduplication(event_type, commit_sha)
        
        should_skip = is_active or is_duplicate
        
        # Enhanced output with retry suppression details
        print(f"SESSION_KEY={session_key}")
        print(f"SKIP_EXECUTION={'true' if should_skip else 'false'}")
        print(f"REASON={reason if is_active else dup_reason if is_duplicate else 'No conflicts detected'}")
        print(f"RETRY_SUPPRESSION={'ENFORCED' if should_skip else 'ACTIVE'}")
        print(f"EARLY_GATING={'SUCCESS' if should_skip else 'PROCEEDING'}")
        
        # Set GitHub Actions output if available
        if os.getenv('GITHUB_OUTPUT'):
            with open(os.getenv('GITHUB_OUTPUT'), 'a') as f:
                f.write(f"session_key={session_key}\n")
                f.write(f"skip_execution={'true' if should_skip else 'false'}\n")
                f.write(f"reason={reason if is_active else dup_reason if is_duplicate else 'No conflicts detected'}\n")
                f.write(f"retry_suppression={'ENFORCED' if should_skip else 'ACTIVE'}\n")
                f.write(f"early_gating={'SUCCESS' if should_skip else 'PROCEEDING'}\n")
        
        sys.exit(0 if not should_skip else 1)
    
    elif command == "register":
        if len(sys.argv) < 7:
            print("Usage: register <session_key> <event_type> <workflow_name> <run_id> <commit_sha>")
            sys.exit(1)
        
        session_key = sys.argv[2]
        event_type = sys.argv[3]
        workflow_name = sys.argv[4]
        run_id = sys.argv[5]
        commit_sha = sys.argv[6]
        
        session_data = deduplicator.register_session(session_key, event_type, workflow_name, run_id, commit_sha)
        print(f"Session registered: {session_key}")
        print(json.dumps(session_data, indent=2))
    
    elif command == "cleanup":
        if len(sys.argv) < 3:
            print("Usage: cleanup <session_key>")
            sys.exit(1)
        
        session_key = sys.argv[2]
        success = deduplicator.cleanup_session(session_key)
        cleaned_count = deduplicator.cleanup_old_sessions()
        
        print(f"Session cleanup: {'success' if success else 'not_found'}")
        print(f"Old sessions cleaned: {cleaned_count}")
    
    elif command == "audit":
        if len(sys.argv) < 9:
            print("Usage: audit <session_key> <event_type> <workflow_name> <run_id> <commit_sha> <executed> <job_status>")
            sys.exit(1)
        
        session_key = sys.argv[2]
        event_type = sys.argv[3]
        workflow_name = sys.argv[4]
        run_id = sys.argv[5]
        commit_sha = sys.argv[6]
        executed = sys.argv[7].lower() == 'true'
        job_status = sys.argv[8]
        duplicate_prevented = not executed
        
        entry = deduplicator.create_audit_entry(
            session_key, event_type, workflow_name, run_id, commit_sha,
            executed, job_status, duplicate_prevented
        )
        
        print("Audit entry created:")
        print(json.dumps(entry, indent=2))
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()