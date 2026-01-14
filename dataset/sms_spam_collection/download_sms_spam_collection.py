import requests
import zipfile

from pathlib import Path


urls = [
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip",  # Backup
]

zip_path = Path("sms_spam_collection.zip")
extract_dir = Path(".")
data_file = extract_dir / "SMSSpamCollection.tsv"

if not data_file.exists():
    for url in urls:
        try:
            print(f"Downloading from {url}")
            with requests.get(url, timeout=60) as r:
                r.raise_for_status()
                zip_path.write_bytes(r.content)


            with zipfile.ZipFile(zip_path) as z:
                z.extractall(extract_dir)

            # If the extracted file is named 'SMSSpamCollection', rename to 'SMSSpamCollection.tsv'
            extracted_file = extract_dir / "SMSSpamCollection"
            if extracted_file.exists():
                extracted_file.rename(data_file)
            zip_path.unlink(missing_ok=True)

            print(f"Saved dataset to {data_file}")
            break

        except Exception as e:
            print(f"Failed with {url}: {e}")
            zip_path.unlink(missing_ok=True)
    else:
        raise RuntimeError("All download attempts failed")
else:
    print(f"{data_file} already exists. Skipping.")
