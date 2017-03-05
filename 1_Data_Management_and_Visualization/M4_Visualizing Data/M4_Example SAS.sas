/*program for NESARC data set*/

LIBNAME mydata "/courses/d1406ae5ba27fe300 " access=readonly;

DATA new; set mydata.nesarc_pds;

LABEL TAB12MDX="Tobacco Dependence Past 12 Months"
	  CHECK321="Smoked Cigarettes in Past 12 Months"
	  S3AQ3B1="Usual Smoking Frequency"
	  S3AQ3C1="Usual Smoking Quantity";

IF AGE LE 20 THEN AGEGROUP=1; /*18, 19, 20 year olds*/
ELSE IF AGE LE 22 THEN AGEGROUP=2; /*21, 22 year olds*/
ELSE AGEGROUP=3; /*23, 24, 25 year olds*/

IF S3AQ3B1=9 THEN S3AQ3B1=.;
IF S3AQ3C1=99 THEN S3AQ3C1=.;

IF S3AQ3B1=1 THEN USFREQ=6;
ELSE IF S3AQ3B1=2 THEN USFREQ=5;
ELSE IF S3AQ3B1=3 THEN USFREQ=4;
ELSE IF S3AQ3B1=4 THEN USFREQ=3;
ELSE IF S3AQ3B1=5 THEN USFREQ=2;
ELSE IF S3AQ3B1=6 THEN USFREQ=1;

/*values for new variable USFREQ
1=once a month or less
2=2-3 days a month
3=1-2 days a week
4=3-4 days a week
5=5-6 days a week
6=every day*/

IF S3AQ3B1=1 THEN USFREQMO=30;
ELSE IF S3AQ3B1=2 THEN USFREQMO=22;
ELSE IF S3AQ3B1=3 THEN USFREQMO=14;
ELSE IF S3AQ3B1=4 THEN USFREQMO=5;
ELSE IF S3AQ3B1=5 THEN USFREQMO=2.5;
ELSE IF S3AQ3B1=6 THEN USFREQMO=1;

/*USFREQMO usual smoking days per month
1=once a month or less
2.5=2-3 days per month
5=1-2 days per week 
14=3-4 days per week
22=5-6 days per week
30=everyday*/

NUMCIGMO_EST=USFREQMO*S3AQ3C1;

NUMCIGMO_EST=USFREQMO*S3AQ3C1;
PACKSPERMONTH=NUMCIGMO_EST/20;
IF PACKSPERMONTH LE 5 THEN PACKCATEGORY=3;
ELSE IF PACKSPERMONTH LE 10 THEN PACKCATEGORY=7;
ELSE IF PACKSPERMONTH LE 20 THEN PACKCATEGORY=15;
ELSE IF PACKSPERMONTH LE 30 THEN PACKCATEGORY=25;
ELSE IF PACKSPERMONTH GT 30 THEN PACKCATEGORY=58;

IF TAB12MDX=1 THEN SMOKEGRP=1; /*nicotine dependent*/
ELSE IF S3AQ3B1=1 THEN SMOKEGRP=2; /*daily smoker*/
ELSE SMOKEGRP=3; /*non daily smoker*/

IF S3AQ3B1=1 THEN DAILY=1;
ELSE IF S3AQ3B1 NE . THEN DAILY=0;

/*subsetting the data to include only past 12 month smokers, age 18-25*/
IF CHECK321=1;
IF AGE LE 25;

PROC SORT; by IDNUM;

PROC PRINT; VAR USFREQMO S3AQ3C1 NUMCIGMO_EST;

PROC FREQ; TABLES TAB12MDX CHECK321 S3AQ3B1 S3AQ3C1 AGE USFREQMO NUMCIGMO_EST;

PROC GCHART; VBAR TAB12MDX/Discrete type=PCT width=30; /*Categorical variable example*/

PROC GCHART; VBAR NUMCIGMO_EST/ type=PCT; /*Quantitatitve variable example*/

PROC UNIVARIATE; VAR NUMCIGMO_EST; /*Univariate PROC only appropriate to use for quantitative variables*/

PROC FREQ; TABLES PACKSPERMONTH;

PROC GCHART; VBAR TAB12MDX/Discrete type=PCT width=30; /*Categorical variable example*/

PROC GCHART; VBAR NUMCIGMO_EST/ type=PCT; /*Quantitatitve variable example*/

PROC UNIVARIATE; VAR NUMCIGMO_EST; /*Only appropriate to use for quantitative variables*/

PROC GCHART; VBAR PACKCATEGORY/discrete TYPE=mean SUMVAR=TAB12MDX;

PROC GCHART; VBAR ETHRACE2A/discrete type=mean SUMVAR=DAILY;

RUN;

/*Program for Gapminder data set*/

LIBNAME mydata "/courses/d1406ae5ba27fe300 " access=readonly;

DATA new2; set mydata.gapminder;

IF incomeperperson eq . THEN incomegroup=.;
ELSE IF incomeperperson LE 744.239 THEN incomegroup=1;
ELSE IF incomeperperson LE 2553.496 THEN incomegroup=2;
ELSE IF incomeperperson LE 9425.236 THEN incomegroup=3;
ELSE IF incomeperperson GT 9425.236 THEN incomegroup=3;

PROC SORT; by COUNTRY;

PROC FREQ; TABLES incomegroup;

PROC UNIVARIATE; VAR urbanrate internetuserate;

PROC GPLOT; PLOT internetuserate*urbanrate;

PROC UNIVARIATE; VAR incomeperperson HIVrate;

PROC GPLOT; PLOT HIVrate*incomeperperson;

PROC GCHART; VBAR incomegroup/discrete type=mean SUMVAR=HIVrate;
RUN;
