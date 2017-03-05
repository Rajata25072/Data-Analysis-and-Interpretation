/*Program for NESARC data set*/
LIBNAME mydata "/courses/d1406ae5ba27fe300 " access=readonly;

DATA new1; set mydata.nesarc_pds;

LABEL TAB12MDX="Tobacco Dependence Past 12 Months"
	  CHECK321="Smoked Cigarettes in Past 12 Months"
	  S3AQ3B1="Usual Smoking Frequency"
	  S3AQ3C1="Usual Smoking Quantity";

IF AGE LE 20 THEN AGEGROUP=1; /*18, 19, 20 year olds*/
ELSE IF AGE LE 22 THEN AGEGROUP=2; /*21, 22 year olds*/
ELSE AGEGROUP=3; /*23, 24, 25 year olds*/

/*set missing data*/
IF S3AQ3B1=9 THEN S3AQ3B1=.;

/*Reverse code values*/
IF S3AQ3B1=1 THEN USFREQ=6;
ELSE IF S3AQ3B1=2 THEN USFREQ=5;
ELSE IF S3AQ3B1=3 THEN USFREQ=4;
ELSE IF S3AQ3B1=4 THEN USFREQ=3;
ELSE IF S3AQ3B1=5 THEN USFREQ=2;
ELSE IF S3AQ3B1=6 THEN USFREQ=1;

/*recode with more meaningful quantitative values*/
IF S3AQ3B1=1 THEN USFREQMO=30;
ELSE IF S3AQ3B1=2 THEN USFREQMO=22;
ELSE IF S3AQ3B1=3 THEN USFREQMO=14;
ELSE IF S3AQ3B1=4 THEN USFREQMO=5;
ELSE IF S3AQ3B1=5 THEN USFREQMO=2.5;
ELSE IF S3AQ3B1=6 THEN USFREQMO=1;

IF S3AQ3C1=99 THEN S3AQ3C1=.;

/*coding in valid data*/
IF S2AQ3 NE 9 AND S2AQ8A=. THEN S2AQ8A=11;

/*secondary variables estimating the number of cigaretted smoked in the past 30 days*/
NUMCIGMO_EST=USFREQMO*S3AQ3C1;

/*subsetting the data to include only past 12 month smokers, age 18-25*/
IF CHECK321=1;
IF AGE LE 25;
PROC SORT; by IDNUM;

PROC PRINT; VAR USFREQMO S3AQ3C1 NUMCIGMO_EST;

PROC FREQ; TABLES TAB12MDX CHECK321 S3AQ3B1 S3AQ3C1 AGE USFREQMO NUMCIGMO_EST;
RUN;
/*Program for AddHealth data set*/

LIBNAME mydata "/courses/d1406ae5ba27fe300 " access=readonly;

DATA new2; set mydata.addhealth_pds;

IF H1GI4 GE 6 then H1GI4=.;
IF H1GI6A GE 6 then H1GI6A=.;
IF H1GI6B GE 6 then H1GI6B=.;
IF H1GI6C GE 6 then H1GI6C=.;
IF H1GI6D GE 6 then H1GI6D=.;

NUMETHNIC=SUM(of H1GI4 H1GI6A H1GI6B H1GI6C H1GI6D);

IF NUMETHNIC GE 2 THEN ETHNICITY=1; /*multiple race/ethnic endorsed*/
ELSE IF H1GI4=1 THEN ETHNICITY=2; /*Hispanic or Latino*/
ELSE IF H1GI6A=1 THEN ETHNICITY=3; /*Black or African American*/
ELSE IF H1GI6B=1 THEN ETHNICITY=4; /*American Indian or Native American*/ 
ELSE IF H1GI6C=1 THEN ETHNICITY=5; /*Asian or Pacific Islander*/
ELSE IF H1GI6D=1 THEN ETHNICITY=6; /*White*/

PROC SORT; by AID;

PROC FREQ; TABLES H1GI4 H1GI6A H1GI6B H1GI6C H1GI6D NUMETHNIC ETHNICITY;

RUN;
