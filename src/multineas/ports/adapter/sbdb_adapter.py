import requests
from multineas.ports.database_conector_port import DatabaseConnectorPort
from multineas.ports.model.data_table import DataTable
import pandas as pd

class SBDBAdapter(DatabaseConnectorPort):

    BASE_URL = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"

    def get_data(self, params: dict) -> DataTable:

        r = requests.get(self.BASE_URL, params=params)
        response = r.json()
        data = pd.DataFrame(response["data"], columns=response["fields"])

        return DataTable(
            data=data,
            source="SBDB",
            metadata=params
        )