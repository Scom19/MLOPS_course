import pandas as pd

def download_data():
    url = '[invalid url, do not cite]
    df = pd.read_excel(url)
    df.to_csv('data/raw/titanic.csv', index=False)
    return df

if __name__ == "__main__":
    download_data()