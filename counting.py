from load import *

metadata = pd.read_csv("./IMA205-challenge/train_metadata.csv")

couples = []
for _, row in metadata.iterrows():
    img_name = row["ID"] # for instance it can be : "train_00000.png"
    img_number = int(img_name.split("_")[1].split(".")[0]) # for instance here it would be 0 (the integer)
    label_index = label_to_index[row["label"]]
    couples.append((img_number, label_index))
 
counts = {label: 0 for label in LABELS}
for _, label_index in couples:
    counts[LABELS[label_index]] += 1
 
print("Class counts:")
for label, count in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {label}: {count}")
print(f"Total: {sum(counts.values())}")
