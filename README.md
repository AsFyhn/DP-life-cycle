# DP-life-cycle
Used for group-work on our term paper 


Inspiration: 
Thomas Code:
https://github.com/ThomasHJorgensen/Sensitivity/tree/master/GP2002

DREAM-gruppen:
https://dreamgroup.dk/Media/638202824872355318/MAKRO_modeling_choices.pdf

Campbell and Mankiw (1989):
https://www.nber.org/system/files/chapters/c10965/c10965.pdf
Notes: 
Two types of households: 
One that consumes as under the permanent income hypothesis and one that consumes as under the life-cycle hypothesis. 


Status 23/05-2024: 

    Still missing:
    - initW = 0 is a problem since m[t=0,:]=0 in pi times of the drawings
    - Does the income process work?
        Notice that the income is very high what is observed in the data.
    - Do we use a log too much? Notice how we take the exponential function of C_avg before returning it?
    - Extention of the model to include two types of consumers.