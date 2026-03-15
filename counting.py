from load import *
 
counts = {label: 0 for label in LABELS}
for _, label_index in couples:
    counts[LABELS[label_index]] += 1
 
print("Class counts:")
for label, count in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {label}: {count}")
print(f"Total: {sum(counts.values())}")
