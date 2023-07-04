import csv
import random

# List of classes
classes = ['person', 'car', 'dog', 'cat', 'tree', 'bench', 'truck']

# Generate random class numbers
class_mapping = {}
for class_name in classes:
    if class_name == 'bench' or class_name == 'truck':
        class_number = 999
    else:
        class_number = random.randint(0, len(classes) - 1)
    class_mapping[class_name] = class_number

# Write class mapping to CSV
csv_file = 'class_mapping.csv'
csv_fields = ['class_name', 'class_number']
with open(csv_file, 'w') as file:
    writer = csv.DictWriter(file, fieldnames=csv_fields)
    writer.writeheader()
    for class_name, class_number in class_mapping.items():
        writer.writerow({'class_name': class_name, 'class_number': class_number})

print('class_mapping.csv generated successfully!')

