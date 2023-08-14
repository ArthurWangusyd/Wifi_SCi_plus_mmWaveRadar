import csv

def polar_add_time(file_name,save_name):

    csvFile = open(save_name, 'w', newline='', encoding='utf-8')
    writer = csv.writer(csvFile)
    csvRow = []

    f = open(file_name, 'r', encoding='GB2312')
    for line in f:
        csvRow = line.split(';')
        writer.writerow(csvRow)

    f.close()
    csvFile.close()