import csv
def dic2csv(dicti, name):
    with open(name+'.csv', 'w') as f:
        writer = csv.DictWriter(f, dicti[0].keys())
        writer.writeheader()
        writer.writerows(dicti)
