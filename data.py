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

from dataclasses import dataclass

@dataclass
class NAICtriangles:
    """
    A class to hold the NAIC triangles.
    """
    data = {}
    cum = None
    incr = None
    cum_lr = None
    incr_lr = None
    naic_df = None
    cnn_df = None


    # general function to read in/clean a csv file
    def read_naic_csv(self, url : str = None, lob : str = None) -> pd.DataFrame:
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

    def build_cumulative(self):
        """
        Build cumulative columns for the dev lag columns.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to build the cumulative columns for.

        Returns
        -------
        df : pd.DataFrame
            The dataframe with the cumulative columns added.
        """
        lobs = ['pers_auto', 'wc', 'comm_auto', 'med_mal', 'products', 'other_liab']
        dflist = []
        for lob in lobs:
            df, _ = self.read_naic_csv(lob=lob)
            df['lob'] = lob
            dflist.append(df)
        cum_df = pd.concat(dflist)
        cum_df = cum_df.loc[cum_df.type_of_loss.isin('reported_loss paid_loss'.split())].reset_index(drop=True)
        cum_df['is_cum'] = 1
        cum_df = cum_df.set_index('is_cum group_code lob type_of_loss ay ep'.split())
        
        self.cum = cum_df

    def build_incremental(self):
        """
        Build incremental columns for the dev lag columns.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to build the incremental columns for.

        Returns
        -------
        df : pd.DataFrame
            The dataframe with the incremental columns added.
        """
        if self.cum is None:
            print('Building cumulative dataframes...')
            self.build_cumulative()

        print('Building incremental dataframes...')
        incr_df = self.cum.copy()
        for i, c in enumerate(incr_df.columns):
            if i == 0:
                continue
            else:
                incr_df.iloc[:, i] = self.cum.iloc[:, i] - self.cum.iloc[:, i-1]

        incr_df.reset_index(inplace=True)

        incr_df['is_cum'] = 0

        incr_df = incr_df.set_index('is_cum group_code lob type_of_loss ay ep'.split())

        self.incr = incr_df


    def build_naic(self):
        """
        Build the NAIC dataframes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # build cumulative and incremental dataframes
        self.build_incremental()

        # concatenate the cumulative and incremental dataframes
        self.naic_df = pd.concat([self.cum, self.incr])

        # add a column to indicate if the data is in dollars or loss ratio
        self.naic_df['is_dollar'] = 1

        print('Building loss ratio dataframes...')
        # store the earned premium column
        self.ep = pd.DataFrame({'ep':self.naic_df.reset_index().ep.values}, index=self.naic_df.index)
        
        # build the loss ratio dataframes
        lr_df = self.naic_df.copy()

        # loop through the columns and divide by the earned premium
        for c in self.naic_df.columns.tolist():
            lr_df[c] = self.naic_df[c] / self.ep.ep

        # set the is_dollar column to 0
        lr_df['is_dollar'] = 0

        # concatenate the loss ratio dataframes
        self.naic_df = pd.concat([self.naic_df, lr_df])

        # reset the index
        self.naic_df.reset_index(inplace=True)

        # drop ay, ep, and the dev lag columns, drop duplicates, and set a triangle_id
        self.tri_df = (self.naic_df
                        .drop(columns='ay ep'.split() + [c for c in self.cum.columns.tolist() if c != 'ep'])
                        .drop_duplicates()
                        .reset_index(drop=True)
                        .assign(triangle_id=lambda x: x.index)
                        )
        
        # create dataset for the convolutional neural network:
        # dictionary of matrices from naic_df with key = triangle_id
        # join the triangle_id to the naic_df to get the triangle_id column
        self.naic_df = self.naic_df.merge(self.tri_df[['triangle_id', 'group_code', 'lob', 'type_of_loss', 'is_cum', 'is_dollar']], on=['group_code', 'lob', 'type_of_loss', 'is_cum', 'is_dollar'])

        self.naic_df = self.naic_df.set_index(['triangle_id', 'group_code', 'lob', 'type_of_loss', 'is_cum', 'is_dollar']).drop(columns='ep')
        # self.cnn_df = self.naic_df.set_index('triangle_id').drop(columns='group_code lob type_of_loss is_cum is_dollar'.split())
        # self.cnn_df = self.cnn_df.to_dict(orient='index')

        # ensure that when ay + dev_lag <= max(ay) + dev_lag[0], the value is not NA -- set to 0
        # this does not apply on the bottom half of the triangle -- eg where ay + dev_lag > max(ay) + dev_lag[0]
        # in this case the value should be NA
        self.naic_df = self.naic_df.reset_index()
        self.naic_df = self.naic_df.melt(id_vars='triangle_id group_code lob type_of_loss is_cum is_dollar ay'.split(), var_name='dev_lag', value_name='value')
        self.naic_df = self.naic_df.assign(ay=lambda x: x.ay.astype(int))
        self.naic_df = self.naic_df.assign(dev_lag=lambda x: x.dev_lag.astype(int))
        self.naic_df = self.naic_df.assign(max_ay=lambda x: x.groupby(['triangle_id', 'group_code', 'lob', 'type_of_loss', 'is_cum', 'is_dollar']).ay.transform(max))
        self.naic_df = self.naic_df.assign(first_dev_lag=lambda x: x.groupby(['triangle_id', 'group_code', 'lob', 'type_of_loss', 'is_cum', 'is_dollar']).dev_lag.transform(min))
        self.naic_df = self.naic_df.assign(is_na=lambda x: (x.ay + x.dev_lag) > (x.max_ay + x.first_dev_lag))
        self.naic_df = self.naic_df.assign(value=lambda x: np.where(x.is_na, np.nan, x.value))
        self.naic_df = self.naic_df.drop(columns='max_ay first_dev_lag is_na'.split())
        self.naic_df = self.naic_df.pivot_table(index='triangle_id group_code lob type_of_loss is_cum is_dollar ay'.split(), columns='dev_lag', values='value')



        
