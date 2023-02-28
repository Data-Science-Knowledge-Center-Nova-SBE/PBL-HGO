def date_format_alertP1(alertP1):
    import pandas as pd

    alertP1["DATA_ENVIO"] = pd.to_datetime(alertP1["DATA_ENVIO"],dayfirst=True, yearfirst=False )
    alertP1["DATA_RECEPCAO"] = pd.to_datetime(alertP1["DATA_RECEPCAO"],dayfirst=True, yearfirst=False)
    alertP1["DATA_RETORNO"] = pd.to_datetime(alertP1[alertP1["DATA_RETORNO"]!='26/06/0214']["DATA_RETORNO"],dayfirst=True, yearfirst=False)
    alertP1["DATA_REALIZACAO"] = pd.to_datetime(alertP1["DATA_REALIZACAO"],dayfirst=True, yearfirst=False)
    alertP1["DATA_MARCACAO"] = pd.to_datetime(alertP1["DATA_MARCACAO"],dayfirst=True, yearfirst=False)
    return alertP1

def replace_blank(alertP1):
    import pandas as pd
    
    alertP1['PROVENIENCIA'][alertP1['PROVENIENCIA']=='']='unknown'
    alertP1['CTH_PRIOR'][alertP1['CTH_PRIOR']=='']='unknown'
    return alertP1

def load_data(filename, df):
        """
        Ensure:
        ---------------
        This method will download the data file into
        a downloads/ directory in the root directory of the project.
        If the data file already exists, the method will not download it again.

        Returns:
        ---------------
        self.data:
            An atribute that allows us to call the dataframe and present in
            pd.Dataframe() format.

        """

        import os
        
        # set up download directory
        download_dir = "downloads"
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # check if file already exists in download directory
        file_path = os.path.join(download_dir, filename)
        if os.path.exists(file_path):
            print(
                f"Data file '{filename}' already exists in '{download_dir}' directory."
            )
            self.data = pd.read_csv(file_path)  # pylint: disable=W0201
        else:
            # save file to download directory
            with open(file_path, "wb") as save:
                df.to_csv(save, index=False)
            print(
                f"Data file '{filename}' downloaded and saved to '{download_dir}' directory."
            )

        return df