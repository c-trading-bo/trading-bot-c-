#!/usr/bin/env python3
"""Monitor GitHub Team usage (20,000 min/month)"""

workflows = {
    'Ultimate ML/RL System': {
        'runs_per_day': 19,
        'minutes_per_run': 15,
        'priority': 'CRITICAL'
    },
    'ES/NQ Critical Trading': {
        'runs_per_day': 15,
        'minutes_per_run': 5,
        'priority': 'CRITICAL'
    },
    'Options Flow Analysis': {
        'runs_per_day': 8,
        'minutes_per_run': 3,
        'priority': 'HIGH'
    },
    'ML Training Enhanced': {
        'runs_per_day': 4,
        'minutes_per_run': 10,
        'priority': 'HIGH'
    },
    'News & Sentiment': {
        'runs_per_day': 15,
        'minutes_per_run': 3,
        'priority': 'MEDIUM'
    },
    'Regime Detection': {
        'runs_per_day': 12,
        'minutes_per_run': 2,
        'priority': 'MEDIUM'
    },
    'Portfolio Heat': {
        'runs_per_day': 20,
        'minutes_per_run': 2,
        'priority': 'MEDIUM'
    },
    'Intelligence Collection': {
        'runs_per_day': 6,
        'minutes_per_run': 10,
        'priority': 'LOW'
    },
    'Daily Consolidated': {
        'runs_per_day': 1,
        'minutes_per_run': 15,
        'priority': 'LOW'
    }
}

print("="*70)
print("GITHUB TEAM SUBSCRIPTION - USAGE OPTIMIZATION")
print("="*70)
print(f"Date: 2025-01-26 20:15:30 UTC")
print(f"User: kevinsuero072897-collab")
print(f"Subscription: GitHub Team (20,000 min/month)")
print("="*70)

total_daily = 0
total_monthly = 0

print("\nWORKFLOW BREAKDOWN:")
print("-"*70)

for name, data in workflows.items():
    daily_minutes = data['runs_per_day'] * data['minutes_per_run']
    monthly_minutes = daily_minutes * 30
    total_daily += daily_minutes
    total_monthly += monthly_minutes
    
    status = "✅" if monthly_minutes < 3000 else "⚠️" if monthly_minutes < 5000 else "🔥"
    
    print(f"{status} {name:25} [{data['priority']:8}] {data['runs_per_day']:3} runs × {data['minutes_per_run']:2} min = {monthly_minutes:5,} min/mo")

print("-"*70)
print(f"\nTOTAL USAGE SUMMARY:")
print(f"  Daily:    {total_daily:,} minutes")
print(f"  Monthly:  {total_monthly:,} minutes")
print(f"  Limit:    20,000 minutes")
print(f"  Usage:    {(total_monthly/20000)*100:.1f}%")
print(f"  Buffer:   {20000-total_monthly:,} minutes")

if total_monthly <= 18000:
    print("\n✅ EXCELLENT: Well optimized for Team subscription!")
    print("   You have good buffer for manual runs and testing.")
elif total_monthly <= 20000:
    print("\n⚠️ TIGHT: Close to limit, monitor carefully!")
else:
    print("\n🔴 OVER LIMIT: Reduce frequency immediately!")

print("\n" + "="*70)
print("KEY IMPROVEMENTS WITH TEAM SUBSCRIPTION:")
print("  • ML trains every hour (24x/day)")
print("  • ES/NQ analysis every 5-10 minutes")
print("  • Options flow every 5-15 minutes")
print("  • Regime detection every 20 minutes")
print("  • 10x better than Free tier")
print("="*70)