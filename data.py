# urls for the csv schedule P data files
urls = dict(
    pers_auto = "https://www.casact.org/sites/default/files/2021-04/ppauto_pos.csv"
    , wc = "https://www.casact.org/sites/default/files/2021-04/wkcomp_pos.csv"
    , comm_auto = "https://www.casact.org/sites/default/files/2021-04/comauto_pos.csv"
    , med_mal = "https://www.casact.org/sites/default/files/2021-04/medmal_pos.csv"
    , products = "https://www.casact.org/sites/default/files/2021-04/prodliab_pos.csv"
    , other_liab = "https://www.casact.org/sites/default/files/2021-04/othliab_pos.csv"
)

column_letter = dict(
    pers_auto = "B"
    , wc = "D"
    , comm_auto = "C"
    , med_mal = "F2"
    , products = "R1"
    , other_liab = "h1"
)

import pandas as pd
import numpy as np

# general function to read in/clean a csv file
def read_naic_csv(url : str = None, lob : str = None) -> pd.DataFrame:
    """
    Read in a csv file from the NAIC website and clean it up.

    Parameters
    ----------
    url : str, optional
        The url of the csv file to be read in.
        Default is None, in which case it will just read in the
        personal auto file 
        ("https://www.casact.org/sites/default/files/2021-04/ppauto_pos.csv")
    lob : str, optional
        The line of business to be read in. Looks up the url from the
        urls dictionary.
        Default is None, in which case it will just read in the
        personal auto file. This is ignored if url is not None.

    Returns
    -------
    df : pd.DataFrame
        The cleaned dataframe.
    """
    # read in the csv file (or use the default url) based on whatever is passed in
    # the hierarchy is url > lob > default
    if url=="other_liab":
        rawdf = pd.read_csv(urls['other_liab'])
        col_letter = "h1"
    else:
        if url is not None:
            rawdf = pd.read_csv(url)
            # get the column letter from the CumPaidLoss_{} column name
            col_name = rawdf.columns[rawdf.columns.str.contains('CumPaidLoss_')][0]
            col_letter = col_name.split('_')[1]

            # get the lob from the column letter
            lob = [k for k, v in column_letter.items() if v == col_letter][0]
        elif lob is not None:
            rawdf = pd.read_csv(urls[lob])

            # get the column letter from the lob
            col_letter = column_letter[lob]
        else:
            rawdf = pd.read_csv(urls['pers_auto'])
            col_letter = column_letter['pers_auto']
    
    # get a lookup table for the group codes/names
    grp_df = rawdf['GRCODE GRNAME'.split()].drop_duplicates().set_index('GRCODE')

    # clean up the dataframe
    df = (rawdf
          .copy()
          
          # drop these columns that won't be used
          .drop(columns='GRNAME DevelopmentYear'.split())

          # these columns vary by lob
          .drop(columns=[f'EarnedPremDIR_{col_letter}'
                         , f'EarnedPremCeded_{col_letter}'
                         , f'PostedReserve97_{col_letter}'
                         ])
          
          # rename the columns
          .rename(columns={ 
              'GRCODE': 'group_code',
              'AccidentYear': 'ay',
              'DevelopmentLag': 'dev_lag',
              f'IncurLoss_{col_letter}': 'incurred_loss',
              f'CumPaidLoss_{col_letter}': 'paid_loss',
              f'BulkLoss_{col_letter}': 'bulk_reserve',
              f'EarnedPremNet_{col_letter}': 'ep'
          })
  
          # add a case reserve column
          .assign(case_reserve=lambda x: x.incurred_loss - x.paid_loss - x.bulk_reserve)
  
          # add a reported loss column
          .assign(reported_loss=lambda x: x.paid_loss + x.case_reserve)
  
          # add a calendar year column
          .assign(cy=lambda x: x.ay + x.dev_lag - 1)
  
          # drop rows with cy > max(ay)
          .query('cy <= ay.max()')
          )

    # melt the dataframe keeping index = ['group_code', 'ay', 'cy', 'dev_lag', 'ep']
    df = (df
            .melt(id_vars=['group_code', 'ay', 'cy', 'dev_lag', 'ep']
                    , value_vars=['incurred_loss', 'reported_loss', 'paid_loss', 'case_reserve', 'bulk_reserve']
                    , var_name='type_of_loss'
                    , value_name='loss')
            .sort_values(['group_code', 'ay', 'cy', 'dev_lag', 'ep', 'type_of_loss'])
            .reset_index(drop=True)
            )
    
    # pivot the dev lag column (also gets rid of the cy column)
    df = (df.pivot_table(index=['group_code', 'type_of_loss', 'ay', 'ep']
                        , columns='dev_lag'
                        , values='loss'
                        ).reset_index()
            .rename_axis(None, axis=1)
    )

    # # reorder the columns
    # column_ord = ['group_code', 'ay', 'type_of_loss', 'ep']
    
    # return the dataframe and the group code lookup table
    return df, grp_df
    
