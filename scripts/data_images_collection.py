"""Module to download images from Arquivo."""
import logging

from pathlib import Path
import sys

parent = Path().absolute().as_posix()
sys.path.insert(0, parent)

from joblib import Parallel, delayed

from utils import DataExtractor

if __name__ == "__main__":

    logger = logging.getLogger("Arquivo-Scraper")
    logging.basicConfig(level=logging.INFO)

    newspapers = {
        "expresso": "expresso.pt",
        "dn": "dn.pt",
        "publico": "publico.pt",
        "jn": "jn.pt",
        "sapo": "sapo.pt",
        "cmjornal": "cmjornal.pt",
        "observador": "observador.pt",
        "sol": "sol.sapo.pt",
        "sabado": "sabado.pt"
    }

    for journal, url in newspapers.items():

        logging.info('Initializing scraping process for: %s.', journal.upper())

        de = DataExtractor(
            newspaper=journal,
            url=f"https://arquivo.pt/textsearch?prettyPrint=true&versionHistory={url}&maxItems=2000&from=2010010100000&to=2022123100000",
            num_records_by_year=20
        )

        data_to_extract = de()
        logging.info('URLs downloaded')
        logging.info('Found %s urls.',  len(data_to_extract))

        logging.info('Initializing picture downloads.')
        Parallel(n_jobs=-1)(delayed(de.download_picture_from_arquivo)(file) for file in data_to_extract)
