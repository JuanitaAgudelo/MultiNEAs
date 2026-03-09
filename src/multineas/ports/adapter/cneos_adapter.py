from multineas.ports.model.data_table import DataTable
from multineas.ports.database_conector_port import DatabaseConnectorPort
import requests
import pandas as pd

class CNEOSAdapter(DatabaseConnectorPort): 

    BASE_URL = "https://ssd-api.jpl.nasa.gov/fireball.api"

    def get_data(self, params: dict) -> DataTable:

        r = requests.get(self.BASE_URL, params=params)
        response = r.json()

        return DataTable(
            data=pd.DataFrame(response["data"], columns=response["fields"]), 
            source="CNEOS",
            metadata=params
        )