# Beiwe Pain Prediction from Accelerometer Data
This notebook describes how to transform smartphone accelerometer data from the Beiwe app into features that can be used for supervised machine learning when paired with patients' daily pain surveys.
The steps:
- Load acceleromet and pain data from .csv files into a PostgreSQL database
- Transform data into features
- Perform supervised learning

This notebook makes use of the pscopg2 library for interfacing with the PosgreSQL server.

## Import data from .csv to SQL
Build two tables: a table with all accerometer data, and a table with all pain data (for now with one patient)


```python
import psycopg2, sys, glob, re
import pandas as pd

"""
This class takes a table name and column names and types and a directory to look
for .csv files and builds a SQL table from those files. 
"""
class BeiweTable:

    def __init__(self, patient_ID, path, table_name, column_names, 
                 column_types, db, user, password):

        self.db = db
        self.user = user
        self.password = password
        self.patient_ID = patient_ID
        self.path = path
        self.table_name = table_name
        self.column_names = column_names
        self.column_types = column_types

        con = None

        with psycopg2.connect(database = self.db, user = self.user, 
                              password = self.password) as con:
            with con.cursor() as cur:
                drop_command = f"DROP TABLE IF EXISTS {self.table_name}"
                cur.execute(drop_command)
                create_command = self.build_create_query()
                cur.execute(create_command)
                con.commit()
        con.close()
    
    def build_create_query(self):
        create_command = f"CREATE TABLE {self.table_name}("
        for i,_ in enumerate(self.column_names):
            create_command += f"{self.column_names[i]} {self.column_types[i]}, "
        create_command = create_command[:-2] + ")"
        return create_command
        # Example statement build by this function:
        # CREATE TABLE pain(question_id varchar, question_type varchar, 
        # question_text varchar, question_answer_options varchar, answer int)"

    def __load__(self):

        con = None
        f = None

        # add .csv files in directory to table
        for file in glob.glob(self.path)[0:2]:

            with psycopg2.connect(database = self.db, user = self.user, 
                                  password = self.password) as con:
                with con.cursor() as cur:
                    f = open(file, 'r')
                    next(f)
                    cur.copy_from(f, self.table_name, sep=",")
                    con.commit()
            con.close()

# extend class BeiweTable to have a specific method for the accelerometer table which
# dds a column for the patient id
class AccelTable(BeiweTable):

    def __cleanup__(self):

        con = None

        # add column to table with patient ID
        with psycopg2.connect(database = self.db, user = self.user, 
                              password = self.password) as con:
            with con.cursor() as cur:
                command = f"""ALTER TABLE {self.table_name} 
                            ADD COLUMN patientID varchar(10) 
                            DEFAULT '{self.patient_ID}';"""
                cur.execute(command)
                con.commit()
        con.close() 
    
path = """/Users/alex/Documents/CNOC/DP Pituitary Data (Asad)/
            3epwzqki/3epwzqki/accelerometer/*.csv"""
column_names = ['time_stamp',
                'utc_time', 
                'accuracy', 
                'x', 
                'y', 
                'z']
column_types = ['bigint', 
                'timestamp', 
                'varchar', 
                'double precision', 
                'double precision', 
                'double precision']
accel_table = AccelTable(patient_ID = '3epwzqki', 
                         path = path, 
                         table_name = 'accel2', 
                         column_names = column_names, 
                         column_types = column_types, 
                         db = 'beiwe', 
                         user = 'postgres', 
                         password = 'dwa')
accel_table.__load__()
accel_table.__cleanup__()
```


```python
# set up the iPython SQL Magic extension and connect to postgres server
# execute the commented out code below:

# !pip install ipython-sql
# %load_ext sql
# %sql postgresql://postgres:dwa@localhost/beiwe
```

### Accerometer Table


```python
%%sql
SELECT * FROM accelerometer limit 5;
```

     * postgresql://postgres:***@localhost/beiwe
    5 rows affected.





<table>
    <tr>
        <th>time_stamp</th>
        <th>utc_time</th>
        <th>accuracy</th>
        <th>x</th>
        <th>y</th>
        <th>z</th>
        <th>patientid</th>
    </tr>
    <tr>
        <td>1559592226057</td>
        <td>2019-06-03 20:03:46.057000</td>
        <td>unknown</td>
        <td>0.235000610351562</td>
        <td>-0.123504638671875</td>
        <td>-0.933425903320312</td>
        <td>3epwzqki</td>
    </tr>
    <tr>
        <td>1559592226157</td>
        <td>2019-06-03 20:03:46.157000</td>
        <td>unknown</td>
        <td>0.141128540039062</td>
        <td>-0.1883544921875</td>
        <td>-0.76336669921875</td>
        <td>3epwzqki</td>
    </tr>
    <tr>
        <td>1559592226257</td>
        <td>2019-06-03 20:03:46.257000</td>
        <td>unknown</td>
        <td>0.0904693603515625</td>
        <td>-0.276092529296875</td>
        <td>-0.710693359375</td>
        <td>3epwzqki</td>
    </tr>
    <tr>
        <td>1559592226357</td>
        <td>2019-06-03 20:03:46.357000</td>
        <td>unknown</td>
        <td>-0.037506103515625</td>
        <td>-0.3280029296875</td>
        <td>-0.848831176757812</td>
        <td>3epwzqki</td>
    </tr>
    <tr>
        <td>1559592226457</td>
        <td>2019-06-03 20:03:46.457000</td>
        <td>unknown</td>
        <td>0.157318115234375</td>
        <td>-0.395263671875</td>
        <td>-0.743881225585938</td>
        <td>3epwzqki</td>
    </tr>
</table>




```python
%%sql
SELECT reltuples AS approximate_row_count FROM pg_class WHERE relname = 'accelerometer';
```

     * postgresql://postgres:***@localhost/beiwe
    1 rows affected.





<table>
    <tr>
        <th>approximate_row_count</th>
    </tr>
    <tr>
        <td>46268404.0</td>
    </tr>
</table>



### Build Pain Table


```python
"""
This class extends the BeiweTable class and replaces its __load__ method.
The timestamp for the pain survey answers has to be extracted from the name
of the .csv file.
"""
class PainTable(BeiweTable):

    def __load__(self):

        con = None

        # add .csv files in directory to table
        for file in glob.glob(self.path)[0:2]:
            
            df_pain_row = pd.read_csv(file, header = None, skiprows = [0])
            pain_score = df_pain_row.iloc[0][4]
            
            if pain_score != 'NO_ANSWER_SELECTED':
                
                match = re.search(r'9ee\/(.*).csv', file)
                time_stamp = match.group(1)
                time_stamp = time_stamp.replace('_',':')

                with psycopg2.connect(database = self.db, user = self.user, password = self.password) as con:
                    with con.cursor() as cur:
                        cur.execute(f"INSERT INTO {self.table_name}({self.column_names[0]}, {self.column_names[1]}, {self.column_names[2]}) VALUES('{time_stamp}', {pain_score}, '{self.patient_ID}')")
                        con.commit()
                con.close()       

path = "/Users/alex/Documents/CNOC/DP Pituitary Data (Asad)/3epwzqki/3epwzqki/survey_answers/56d60b801206f7036f8919ee/*.csv"
column_names = ['utc_time', 
                'pain_score',
                'patientID']
column_types = ['timestamp', 
                'int', 
                'varchar']
ptable = PainTable(patient_ID = '3epwzqki', 
                   path = path, 
                   table_name = 'pain2', 
                   column_names = column_names, 
                   column_types = column_types)
ptable.__load__()
```

### Pain Table


```python
%%sql
SELECT * FROM pain limit 5;
```

     * postgresql://postgres:***@localhost/beiwe
    5 rows affected.





<table>
    <tr>
        <th>utc_time</th>
        <th>pain_score</th>
        <th>patientid</th>
    </tr>
    <tr>
        <td>2019-06-05 22:58:15</td>
        <td>3</td>
        <td>3epwzqki</td>
    </tr>
    <tr>
        <td>2019-06-06 21:11:49</td>
        <td>1</td>
        <td>3epwzqki</td>
    </tr>
    <tr>
        <td>2019-06-07 21:12:46</td>
        <td>1</td>
        <td>3epwzqki</td>
    </tr>
    <tr>
        <td>2019-06-08 21:13:43</td>
        <td>5</td>
        <td>3epwzqki</td>
    </tr>
    <tr>
        <td>2019-06-09 23:51:32</td>
        <td>6</td>
        <td>3epwzqki</td>
    </tr>
</table>




```python
%%sql
SELECT Count(*) FROM pain;
```

     * postgresql://postgres:***@localhost/beiwe
    1 rows affected.





<table>
    <tr>
        <th>count</th>
    </tr>
    <tr>
        <td>83</td>
    </tr>
</table>



## Build Features from Accelerometer Data
So far, we have an accelerometer table with 50 million rows and a pain table with 83 rows for one patient. The accelerometer data is collected at 10 Hz with a 50% duty cycle. The pain survery is administered once per day. We need to distill the accelerometer data into features that can be associated with individual pain scores.

The first step will be to convert from x-y-z accelerometer readings into a magnitude.


```python
class ModAccel:

    """
    The purpose of this class is to take the accelerometer table in Beiwe from having columns
    time_stamp   |        utc_time         | accuracy |         x          |         y          |         z          | patientid 
    and make a new derived table with just utc_time | mag | patientid
    """

    def __init__(self, db, user, password):

        self.db = db
        self.user = user
        self.password = password

    def deriveTable(self):

        con = None

        with psycopg2.connect(database = self.db, user = self.user, password = self.password) as con:
            with con.cursor() as cur:
                
                drop_command = "DROP VIEW IF EXISTS accel_view"
                cur.execute(drop_command)

                command = """CREATE VIEW accel_view AS SELECT patientid, 
                            utc_time, date_trunc('day', utc_time) AS utc_time_day, 
                            date_trunc('hour', utc_time) AS utc_time_hour, 
                            (sqrt( x ^ 2.0 + y ^ 2.0 + z ^ 2.0) - 1) * 1000 
                            AS MAG FROM accelerometer;"""
                cur.execute(command)

                drop_command = "DROP TABLE IF EXISTS accel_mod2"
                cur.execute(drop_command)

                command = "CREATE TABLE accel_mod2 AS SELECT * FROM accel_view"
                cur.execute(command)
                
                command = "ALTER TABLE accel_mod2 ALTER COLUMN mag SET DATA TYPE real;"
                cur.execute(command)

                con.commit()
                
        con.close()  

am = ModAccel(db = 'beiwe', user = 'postgres', password = 'dwa')
am.deriveTable()
```

### View table with accelerometer magnitude
This table drops some unnecessary rows from the accelerometer table, and includes new derived rows ushc as the day and the hour, but most importantly the magnitude of the acceleromter reading (in units of 10^-3 g).


```python
%%sql
SELECT * FROM accel_mod2 limit 5;
```

     * postgresql://postgres:***@localhost/beiwe
    5 rows affected.





<table>
    <tr>
        <th>patientid</th>
        <th>utc_time</th>
        <th>utc_time_day</th>
        <th>utc_time_hour</th>
        <th>mag</th>
    </tr>
    <tr>
        <td>3epwzqki</td>
        <td>2019-06-03 20:03:46.057000</td>
        <td>2019-06-03 00:00:00</td>
        <td>2019-06-03 20:00:00</td>
        <td>-29.555464</td>
    </tr>
    <tr>
        <td>3epwzqki</td>
        <td>2019-06-03 20:03:46.157000</td>
        <td>2019-06-03 00:00:00</td>
        <td>2019-06-03 20:00:00</td>
        <td>-201.17374</td>
    </tr>
    <tr>
        <td>3epwzqki</td>
        <td>2019-06-03 20:03:46.257000</td>
        <td>2019-06-03 00:00:00</td>
        <td>2019-06-03 20:00:00</td>
        <td>-232.21303</td>
    </tr>
    <tr>
        <td>3epwzqki</td>
        <td>2019-06-03 20:03:46.357000</td>
        <td>2019-06-03 00:00:00</td>
        <td>2019-06-03 20:00:00</td>
        <td>-89.22726</td>
    </tr>
    <tr>
        <td>3epwzqki</td>
        <td>2019-06-03 20:03:46.457000</td>
        <td>2019-06-03 00:00:00</td>
        <td>2019-06-03 20:00:00</td>
        <td>-143.06264</td>
    </tr>
</table>




```python
import numpy as np
from scipy import signal
#import time

class Hourly_FFT(BeiweTable): 

    def __load__(self):
        #tic = time.time()
        # get list of days using distinct(utc_time_day) from table accel_mod
        with psycopg2.connect(database = self.db, user = self.user, 
                              password = self.password) as con:
            with con.cursor() as cur:
                command = f"SELECT distinct(utc_time_day) FROM accel_mod ORDER BY utc_time_day;"

                df_day_list = pd.read_sql_query(command, con)
                day_list = list(df_day_list['utc_time_day'])

                con.commit()
        con.close()  

        # for each day
        for day in day_list:

            # get a pandas dataframe with the accel data from that day
            with psycopg2.connect(database = self.db, user = self.user, 
                              password = self.password) as con:
                with con.cursor() as cur:
                    command = f"""SELECT utc_time, utc_time_hr, 
                    mag FROM accel_mod WHERE patientID = '{self.patient_ID}' 
                    and utc_time_day = '{day}' ORDER BY utc_time_day;"""

                    df_one_day = pd.read_sql_query(command, con)

                    con.commit()
            con.close()  

            # get a list of distinct hours from the dataframe
            hours_list = list(df_one_day.utc_time_hr.unique())

            # initialize list that will contain rows to be writtent to the hourly_fft table
            fft_by_hour = []

            # for each hour
            for hour in hours_list:

                # get all ordered mag column from all accel_mod rows that match that hour
                # first get all matching rows from df_one_day
                df_hour = df_one_day.loc[df_one_day['utc_time_hr'] == hour]
                # then sort on utc_time
                df_hour = df_hour.sort_values(by=['utc_time'])
                # then get just the mag column as a list
                mag_vect = df_hour['mag'].tolist()

                hr_len = len(mag_vect)

                if hr_len >= 100:
                    # get energy of signal
                    hr_len = len(mag_vect)
                    energy = np.sum(np.square(mag_vect)) / hr_len;

                    # get power spectral density of signal
                    nfft = hr_len // 11
                    freqs, Pxx = signal.periodogram(mag_vect, fs = 10, nfft = nfft)

                    # get average power in several frequency windows
                    # due to the sampling frequency of 10 Hz, the FFT range is from 0 to 5 Hz
                    # bins: 0-1, 1-2, 2-3, 3-4, 4-5
                    df_power = pd.DataFrame()
                    df_power['freqs'] = freqs
                    df_power['Pxx'] = Pxx

                    power1 = np.array(df_power[df_power['freqs'].between(0, 1)]['Pxx'])
                    pow_len = len(power1)
                    power1 = np.sum(power1) / pow_len

                    power2 = np.array(df_power[df_power['freqs'].between(1, 2)]['Pxx'])
                    pow_len = len(power2)
                    power2 = np.sum(power2) / pow_len

                    power3 = np.array(df_power[df_power['freqs'].between(2, 3)]['Pxx'])
                    pow_len = len(power3)
                    power3 = np.sum(power3) / pow_len

                    power4 = np.array(df_power[df_power['freqs'].between(3, 4)]['Pxx'])
                    pow_len = len(power4)
                    power4 = np.sum(power4) / pow_len

                    power5 = np.array(df_power[df_power['freqs'].between(4, 5)]['Pxx'])
                    pow_len = len(power5)
                    power5 = np.sum(power5) / pow_len

                    this_hour = [self.patient_ID, hour, energy, power1, power2, power3, power4, power5]
                    fft_by_hour.append(this_hour)

            # write to SQL table
            with psycopg2.connect(database = self.db, user = self.user, 
                              password = self.password) as con:
                with con.cursor() as cur:
                    for hour in fft_by_hour:
                        command = f"""INSERT INTO {self.table_name}(
                                {column_names[0]}, {column_names[1]},
                                {column_names[2]}, {column_names[3]}, 
                                {column_names[4]}, {column_names[5]}, 
                                {column_names[6]}, {column_names[7]}) 
                                VALUES('{self.patient_ID}', '{hour[1]}', 
                                {hour[2]}, {hour[3]}, {hour[4]}, {hour[5]}, 
                                {hour[6]}, {hour[6]})"""
                        cur.execute(command)

                    con.commit()
            con.close()

        #toc = time.time()
        #print(toc-tic)

column_names = ['patientID', 
                'utc_time_hr', 
                'energy',
                'power1', 
                'power2', 
                'power3',
                'power4',
                'power5']
column_types = ['varchar', 
                'timestamp', 
                'real', 
                'real', 
                'real', 
                'real', 
                'real', 
                'real']
fft_table = Hourly_FFT(patient_ID = '3epwzqki',
                       path = "",
                       table_name = 'hourly_fft2', 
                       column_names = column_names, 
                       column_types = column_types,
                       db = 'beiwe', 
                       user = 'postgres', 
                       password = 'dwa')
fft_table.__load__()
```

    105.5047287940979


### Hourly FFT Table
This table has 6 features derived from the accelerometer data. All acccelerometer data from a given hour is binned together to generate these features. Energy is equal the the magnitude of the signal, squared. Power1 refers to the average of the power spectrum in the frequency band of 0-1 Hz, Power to is 1-2 Hz, etc. These values are normalized by the number of points used (they are average values).


```python
%%sql
SELECT * FROM hourly_fft limit 5;
```

     * postgresql://postgres:***@localhost/beiwe
    5 rows affected.





<table>
    <tr>
        <th>patientid</th>
        <th>utc_time_hr</th>
        <th>energy</th>
        <th>power1</th>
        <th>power2</th>
        <th>power3</th>
        <th>power4</th>
        <th>power5</th>
    </tr>
    <tr>
        <td>3epwzqki</td>
        <td>2019-06-03 20:00:00</td>
        <td>2924.1116</td>
        <td>1767.599</td>
        <td>1246.7458</td>
        <td>1415.6505</td>
        <td>1664.6732</td>
        <td>1664.6732</td>
    </tr>
    <tr>
        <td>3epwzqki</td>
        <td>2019-06-03 21:00:00</td>
        <td>1432.4454</td>
        <td>191.40022</td>
        <td>102.934525</td>
        <td>97.83585</td>
        <td>123.842384</td>
        <td>123.842384</td>
    </tr>
    <tr>
        <td>3epwzqki</td>
        <td>2019-06-03 22:00:00</td>
        <td>98.982</td>
        <td>130.32928</td>
        <td>181.7762</td>
        <td>152.33188</td>
        <td>90.53865</td>
        <td>90.53865</td>
    </tr>
    <tr>
        <td>3epwzqki</td>
        <td>2019-06-03 23:00:00</td>
        <td>20.038174</td>
        <td>16.689966</td>
        <td>22.00016</td>
        <td>24.686247</td>
        <td>11.551547</td>
        <td>11.551547</td>
    </tr>
    <tr>
        <td>3epwzqki</td>
        <td>2019-06-04 13:00:00</td>
        <td>10413.467</td>
        <td>0.14755344</td>
        <td>0.16216113</td>
        <td>0.1401401</td>
        <td>0.19015</td>
        <td>0.19015</td>
    </tr>
</table>



### Build table associating daily average of derived features with daily pain label
We can use a SQL query to build a new table where for each pain score, we take that time that it wsa recorded, look back 24 hours, and average the derived features (energy, power1, etc.) over that period (and take the standard deviation).


```python
%%sql

CREATE TABLE labeled_features2 AS
Select
h.patientid, utc_time
,avg(energy) as energy_avg
,stddev_samp(energy) as energy_dev
,avg(power1) as power1_avg
,stddev_samp(power1) as power1_dev
,avg(power2) as power2_avg
,stddev_samp(power2) as power2_dev
,avg(power3) as power3_avg
,stddev_samp(power3) as power3_dev
,avg(power4) as power4_avg
,stddev_samp(power4) as power4_dev
,avg(power5) as power5_avg
,stddev_samp(power5) as power5_dev
,pain_score
From hourly_fft h
Inner Join pain p ON h.patientid = p.patientid
    and p.utc_time - interval '24 hours' <= h.utc_time_hr
    and p.utc_time >= h.utc_time_hr
Group by utc_time, h.patientid, pain_score
Order by utc_time;
```

     * postgresql://postgres:***@localhost/beiwe
    83 rows affected.





    []




```python
%%sql

SELECT * FROM labeled_features2 LIMIT 5;
```

     * postgresql://postgres:***@localhost/beiwe
    5 rows affected.





<table>
    <tr>
        <th>patientid</th>
        <th>utc_time</th>
        <th>energy_avg</th>
        <th>energy_dev</th>
        <th>power1_avg</th>
        <th>power1_dev</th>
        <th>power2_avg</th>
        <th>power2_dev</th>
        <th>power3_avg</th>
        <th>power3_dev</th>
        <th>power4_avg</th>
        <th>power4_dev</th>
        <th>power5_avg</th>
        <th>power5_dev</th>
        <th>pain_score</th>
    </tr>
    <tr>
        <td>3epwzqki</td>
        <td>2019-06-05 22:58:15</td>
        <td>1964.9636526107788</td>
        <td>3030.263212218565</td>
        <td>451.9461304138725</td>
        <td>1216.4262385054885</td>
        <td>1069.459881034835</td>
        <td>2626.0913585140797</td>
        <td>574.981796545287</td>
        <td>1385.818456771253</td>
        <td>578.2732340143993</td>
        <td>1493.2195729690727</td>
        <td>578.2732340143993</td>
        <td>1493.2195729690727</td>
        <td>3</td>
    </tr>
    <tr>
        <td>3epwzqki</td>
        <td>2019-06-06 21:11:49</td>
        <td>1892.8634694317977</td>
        <td>3211.900180406324</td>
        <td>246.93419240880758</td>
        <td>790.4697256414902</td>
        <td>459.94185037755716</td>
        <td>1481.5787950218823</td>
        <td>299.33030486200005</td>
        <td>879.3400616006656</td>
        <td>167.23186470288783</td>
        <td>394.76330727706943</td>
        <td>167.23186470288783</td>
        <td>394.76330727706943</td>
        <td>1</td>
    </tr>
    <tr>
        <td>3epwzqki</td>
        <td>2019-06-07 21:12:46</td>
        <td>5137.93660830458</td>
        <td>8691.344462069257</td>
        <td>236.962443299902</td>
        <td>454.2264539631129</td>
        <td>404.1182154485335</td>
        <td>742.5606022381246</td>
        <td>276.74742526654154</td>
        <td>506.22133238755754</td>
        <td>315.38311126315966</td>
        <td>655.6198569604643</td>
        <td>315.38311126315966</td>
        <td>655.6198569604643</td>
        <td>1</td>
    </tr>
    <tr>
        <td>3epwzqki</td>
        <td>2019-06-08 21:13:43</td>
        <td>9076.543785959482</td>
        <td>12066.397019641707</td>
        <td>1425.719606517038</td>
        <td>2541.8506476890593</td>
        <td>5557.923047574547</td>
        <td>13619.041261379447</td>
        <td>2225.0699506939077</td>
        <td>3869.8344778713217</td>
        <td>1454.7240335786094</td>
        <td>2282.8069754808625</td>
        <td>1454.7240335786094</td>
        <td>2282.8069754808625</td>
        <td>5</td>
    </tr>
    <tr>
        <td>3epwzqki</td>
        <td>2019-06-09 23:51:32</td>
        <td>5621.321787749727</td>
        <td>9565.948190990692</td>
        <td>4646.340947562363</td>
        <td>20869.568744144646</td>
        <td>4466.984863543107</td>
        <td>11182.524607459078</td>
        <td>1414.1268346863799</td>
        <td>3741.2137494491144</td>
        <td>800.5456196715435</td>
        <td>1512.907236880206</td>
        <td>800.5456196715435</td>
        <td>1512.907236880206</td>
        <td>6</td>
    </tr>
</table>




```python
# excecute this query to save this table to a .csv file:
#\copy (SELECT * FROM labeled_features2) to '/Users/alex/Documents/Data_Science/Postgres/labeled_features2.csv' csv header
```

# Machine Learning
Make a model to predict pain state from the derived accelerometer features.

### Linear Regression


```python
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# load data from .csv
df_features = pd.read_csv('/Users/alex/Documents/Data_Science/Postgres/labeled_features.csv')
df_features = df_features.dropna()

# partition into training and test set
data = df_features[['energy_avg','energy_dev','power1_avg','power1_dev','power2_avg',
                    'power2_dev','power3_avg','power3_dev','power4_avg','power4_dev',
                    'power5_avg','power5_dev']]
labels = df_features[['pain_score']]

# Create a classifier object with the classifier and parameters
clf = LinearRegression(normalize = True)

# Train the classifier on data1's feature and target data
clf.fit(data, labels)

# Get predictions for the test data
preds = clf.predict(data)

# get coefficient of determination
R2 = clf.score(data, labels)

# Plot predictions vs labels
plt.plot(labels, preds, 'o');
plt.xlabel('Pain Score');
plt.ylabel('Predicted Pain Score');
plt.title(f"R^2: {R2:02f}");
```


![png](output_24_0.png)


The coefficient of determination is quite low; the linear regression is not very predictive.

### Random Forest Classifier


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# load data from .csv
df_features = pd.read_csv('/Users/alex/Documents/Data_Science/Postgres/labeled_features.csv')
df_features = df_features.dropna()

# partition into training and test set
data = df_features[['energy_avg','energy_dev','power1_avg','power1_dev','power2_avg',
                    'power2_dev','power3_avg','power3_dev','power4_avg','power4_dev',
                    'power5_avg','power5_dev']]
labels = df_features[['pain_score']]

X_train, X_test, y_train, y_test = train_test_split(data, labels, 
                                                   test_size=0.2, random_state=0)

parameter_candidates = [
  {'n_estimators': [10, 100, 500], 
   'max_depth': [None, 5, 10, 50], 
   'criterion': ['gini','entropy']},
]

# Create a classifier object with the classifier and parameter candidates
clf = GridSearchCV(estimator=RandomForestClassifier(), 
                   param_grid=parameter_candidates, n_jobs=-1, cv = 3)

# Train the classifier on data1's feature and target data
clf.fit(X_train, y_train['pain_score'].ravel())

# View the accuracy score
print('Best score for data1:', clf.best_score_)

# View the best parameters for the model found using grid search
print('Best n_estimators:',clf.best_estimator_.n_estimators) 
print('Best max_depth:',clf.best_estimator_.max_depth)
print('Best Criterion:',clf.best_estimator_.criterion)

# Get predictions for the test data
preds = clf.predict(X_test)

# Plot predictions vs labels
plt.plot(y_test, preds, 'o');
plt.xlabel('Pain Score');
plt.ylabel('Predicted Pain Score');
```

    /opt/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:667: UserWarning: The least populated class in y has only 2 members, which is less than n_splits=3.
      % (min_groups, self.n_splits)), UserWarning)


    Best score for data1: 0.24675324675324672
    Best n_estimators: 10
    Best max_depth: 50
    Best Criterion: entropy



![png](output_27_2.png)


The accuracy score is quite low. What if instead of having to correctly guess 1 of 11 labels, the classificaiton was made binary?

### Binary Classification
Sort pain states into two categories: low (0-3) and high (4-10)


```python
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns;
from sklearn.metrics import f1_score

# turn labels to binary (low or high pain)
pain_cat = []
for label in labels['pain_score']:
    if label <= 3:
        pain_cat += [0]
    else:
        pain_cat += [1]
        
X_train, X_test, y_train, y_test = train_test_split(data, pain_cat, 
                                                   test_size=0.2, random_state=0)

parameter_candidates = [
  {'n_estimators': [10, 100, 500], 
   'max_depth': [None, 5, 10, 50], 
   'criterion': ['gini','entropy']},
]

# Create a classifier object with the classifier and parameter candidates
clf = GridSearchCV(estimator=RandomForestClassifier(), 
                   param_grid=parameter_candidates, n_jobs=-1, cv = 3, scoring = 'f1')

# Train the classifier on data1's feature and target data
clf.fit(X_train, y_train)

# View the accuracy score
print('Best score for data1:', clf.best_score_)

# View the best parameters for the model found using grid search
print('Best n_estimators:',clf.best_estimator_.n_estimators) 
print('Best max_depth:',clf.best_estimator_.max_depth)
print('Best Criterion:',clf.best_estimator_.criterion)

# get predicitons for test data, compute F1 score
predictions = clf.predict(X_test)
f1 = f1_score(y_test, predictions, average='macro')

# generate and plot confusion matrix
con_mat = metrics.confusion_matrix(y_test, predictions)
sns.set(font_scale = 1)
plt.figure(figsize=(3,3))
ax = sns.heatmap(con_mat, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'F1 Score: {0:.5f}'.format(f1)
plt.title(all_sample_title, size = 12);
#ax = sns.heatmap(df_corr, annot=True) #notation: "annot" not "annote"
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5);
```

    Best score for data1: 0.1851851851851852
    Best n_estimators: 10
    Best max_depth: None
    Best Criterion: entropy



![png](output_30_1.png)


The model cannot generalize the rare high pain state conditions; in the test data it never predicts the high pain state.

## Ideas for Improvements
- Exploratory data analysis on derived features (energy, power1, etc.). Are there differences in these features between the low pain and high pain categories?
- Derivation of new features using singular spectrum analysis (note: use absolute value of acceleration)

