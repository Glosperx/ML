import pandas as pd
def convert_mde_to_csv(mde_path, csv_path):
    with open(mde_path, 'r') as infile, open(csv_path, 'w') as outfile:
        header_written = False
        for line in infile:
            if line.startswith('#'):
                if not header_written:
                    # Scriem header-ul fără #
                    outfile.write(line.replace('#', '').strip() + '\n')
                    header_written = True
                continue
            if line.strip() == '':
                continue
            outfile.write(line)

convert_mde_to_csv('pentilfuran.MDE', 'pentilfuran.csv')