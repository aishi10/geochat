import json
from pathlib import Path

root = Path(r"C:\Geochat\geochat\output\bigearthnet_1000_3epochs_lora")

state = json.loads((root / "trainer_state.json").read_text(encoding="utf-8"))
print("TRAINER STATE")
print("Global step:", state.get("global_step"))
print("Epoch:", state.get("epoch"))
print("Best metric:", state.get("best_metric"))

history = state.get("log_history", [])
print("\nLAST 10 LOG ENTRIES")
for item in history[-10:]:
    print(item)

print("\nADAPTER FILES")
for p in sorted(root.iterdir()):
    if p.is_file():
        print(p.name, p.stat().st_size, "bytes")
