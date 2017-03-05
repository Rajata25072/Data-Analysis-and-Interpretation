LIBNAME mydata "/courses/d1406ae5ba27fe300 " access=readonly;

DATA new; set mydata.nesarc_pds;

LABEL TAB12MDX="Tobacco Dependence Past 12 Months" 
	  CHECK321="Smoked Cigarettes in Past 12 Months"
	  S3AQ3B1="Usual Smoking Frequency"
	  S3AQ3C1="Usual Smoking Quantity";

/*subsetting the data to include only past 12 month smokers, age 18-25*/
IF CHECK321=1;
IF AGE LE 25;

PROC SORT; by IDNUM;

PROC FREQ; TABLES TAB12MDX CHECK321 S3AQ3B1 S3AQ3C1 AGE;

RUN;

