libname mydata "/courses/d1406ae5ba27fe300/c_3054" access=readonly;

data new; set mydata.gapminder;
run;

****************************************************************************************************
BASIC LINEAR REGRESSION
****************************************************************************************************;

* scatterplot with linear regression line;
proc sgplot;
  reg x=urbanrate y=internetuserate / lineattrs=(color=blue thickness=2);
  title "Scatterplot for the Association Between Urban Rate and Internet Use Rate";
  yaxis label="Female Employment Rate";
  xaxis label="Urbanization Rate";
run;
title;

* basic linear regression;
PROC glm; 
model internetuserate=urbanrate/solution;
run;


















