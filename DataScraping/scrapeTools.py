import csv
def dic2csv(dicti, name):
        with open(name+'.csv', 'w') as f:
                w = csv.DictWriter(f,dicti[0].keys())
                w.writeheader()
                w.writerows(dicti)
