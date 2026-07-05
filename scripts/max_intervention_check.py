import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mas_engagement.blackboard import SharedState
from mas_engagement.config import MAX_INTERVENTIONS

print(f'MAX_INTERVENTIONS = {MAX_INTERVENTIONS}')
print()

bb = SharedState(warmup_duration=0)
all_ok = True

for count in range(MAX_INTERVENTIONS + 1):
    snap = bb.snapshot()
    should_silent = snap['intervention_count'] >= MAX_INTERVENTIONS
    print(f'  intervention_count={snap["intervention_count"]}  should_silent={should_silent}')
    if count < MAX_INTERVENTIONS:
        bb.inc_intervention_count()

# Final check: at count=MAX_INTERVENTIONS, should_silent must be True
bb2 = SharedState(warmup_duration=0)
for _ in range(MAX_INTERVENTIONS):
    bb2.inc_intervention_count()
snap_full = bb2.snapshot()
cap_active = snap_full['intervention_count'] >= MAX_INTERVENTIONS

print()
print(f'Budget cap aktif saat count={MAX_INTERVENTIONS}: {cap_active}  (harus True)')
print(f'Hasil: {"LULUS" if cap_active else "GAGAL"}')
