#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:16:28 2024

@author: bhumanyu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 00:12:42 2024

@author: bhumanyu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 11:03:32 2024

@author: bhumanyu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 12:43:26 2024

@author: bhumanyu
"""

# to do - valuation of KIKO , valuation of IN options; payoff in spot space and not log space 


# USEFUL INFO for {INSTRUMENT TYPE} {KNOCK STYLE} {BARRIER STYLE}

# INSTRUMETN LIST  => BARRIER ; TOUCH ; DIGITAL ;(same payoff formula for single or double barrier options)

# KNOCK STYLE LIST: 
# Single barriers  - OUT (UO, DO) ; IN (UI, DI)        
# Double barriers - OUT (KO) ; IN (KI)
# Double KIKO - upin downout; upout downin
# For touch options 
# single barrier - OUT (NTU, NTD) ; IN (OTU, OTD)  #identical to Single barrier conditions
# double barrier - OUT (DNT) ; IN (DOT)   #identical to double barrier conditions 

# BARRIER STYLE LIST: 
# American - barrier check throughout its life 
# European - only at maturity 
# Partial - only in a partial window (barrier_windw_start )
# Bermudan - check on discrete dates (for a given set of dates )


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| LIBRARIES |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| 

import numpy as np 
import pandas as pd

import scipy.sparse as sp  
from scipy.sparse.linalg import splu        #will be faster for flat vol and flat IR 
from scipy.sparse.linalg import spsolve     # will be faster for non flat vol adn IR - basically when M1 and M2 are found in each step 
from numpy.linalg import cond
from scipy import stats

from datetime import datetime

import QuantLib as ql

import matplotlib.pyplot as plt
from math import ceil as ceil
import os



#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::INPUT DATA:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 

 
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||MARKET AND TRADE DATA|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

spot = 1.358

flat_rdom = 0.03893
flat_rfor = 0.04827
constant_vol = 0.04990

ir_path = None 
vol_path = None 

#TRADE DATA  ============================================================================================================================================================

strike = 1.3546
notional = 1000                            #for digital and touch options this is the payoff itself

domestic_currency = 'CAD'
foreign_currency = 'USD'
payout_currency = 'CAD'

if payout_currency == domestic_currency :
    currency_adjustment = 1
else: 
    currency_adjustment = 0 

#INSTRUMENT RELATED DATA ===================================================================================================================================

# instrument list  => BARRIER ; TOUCH ; DIGITAL 
instrument_type =  'barrier'
option_type = 'call'

#BARRIER RELATED DATA ===================================================================================================================================================

# knock style list: Single barriers  - OUT (UO, DO) ; IN (UI, DI); Double barriers - OUT (KO) ; IN (KI); Double KIKO - 'UIDO'upin downout; 'UODI'upout downin
# for touch options -> single barrier - OUT (NTU, NTD) ; IN (OTU, OTD); double barrier - OUT (NTD) ; IN (OTD)   
# barrier_style list -> 'american', 'partial', 'european', bermudan

#barrier_types  --------------------------------------------------------------------------------------------------------------------------------------
knock_style =  'UO'  #pass as per the list provided below
barrier_style =  'American'  #pass as per the list provided below, for vanilla option pricing pass 'no' dont pass None 
rebate_flag = 0
rebate_immediate = 0

#barrier levels 
up_barrier = 1.4937   #if empty then pass as 'None'
down_barrier = None   #if empty then pass 'None'
rebate = None

#some error handling here woudl be good - instrument style and knock style relation (digital and its corresponding) 
#and also the knock style and the values of how many barriers 

#|||||||||||||||||||||||||||||||||||||||||||||||||||FINITE DIFFERENCE SHCEME GRID INPUTS |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||   


#SPATIAL BOUNDARIES IN SPOT AND BARRIER LEVEL ADJUSTMENTS ------------------------------------------------------------------------------------------------------------------------

multiple = 5 #tells how far above the s_max is from max of (spot, strike, up_barrier)

s_min = 1e-3   

if knock_style.lower() == 'vanilla':
    s_max = float(max(strike, spot)* multiple) 
    up_barrier = s_max
    down_barrier = s_max     #this way there wont be problems with none characters and also the np.sort will take care of these 
    #since the rebate window will be empty and so will be the barrier indicators, this will not impact the output in any other way 

else: 
    if up_barrier:    #however, this would also not run for the case where up_barrier = 0 which is actually the right thing to do in that case. 
        s_max = float(max(strike,spot,up_barrier)* multiple)
    else: 
        s_max = float(max(strike, spot)* multiple)   # lets make it 2 times rather than 10 times
        up_barrier = s_max
        print('up_barrier not passed, updated using s_max')

    if down_barrier == None :    
        down_barrier = s_min 
        print('down_barrier not passed, updated using s_min')


#STEP SIZE =========================================================================================================================================================

#spot space ------------------------------------------------------------------------------------------------------------------------------------------------
relative_dense_space_step_continuous = 0.005   #/100
relative_dense_space_step_discrete = 0.0005   #/100

    
#time space ------------------------------------------------------------------------------------------------------------------------------------------------
time_step_continuous = 0.002
time_step_discrete = 0.01 


#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| TESTING TOGGLES ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  


flat_ts = 1     #if flat = 1 then use flat term structure of rates 
flat_vol = 1    #if flat = 1 then use flat term structure of Vol surface


#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| DATE INPUTS AND QUANTLIB ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  


#DATE INPUTS =========================================================================================================================================================

def safe_date_parse(date_str, date_format='%d-%m-%Y'):
    if date_str and date_str != '--':  # Check for non-empty and non-placeholder strings
        try:
            return datetime.strptime(date_str, date_format)
        except ValueError:
            print(f"Warning: '{date_str}' does not match format '{date_format}'")
            return None
    else:
        return None


current_date = safe_date_parse('18-9-2024')
maturity_date = safe_date_parse('19-12-2024')

# For partial barriers style
barrier_window_start = safe_date_parse('--')
barrier_window_end = safe_date_parse('--')

# For bermudan barrier style
bermudan_window_dates_string_input = ['', '']
bermudan_window_dates = [safe_date_parse(i) for i in bermudan_window_dates_string_input if i]

# For rebate schedule
rebate_payment_dates_string_input = ['', '']
rebate_payment_dates = [safe_date_parse(i) for i in rebate_payment_dates_string_input if i]


#QUANTLIB DATES =========================================================================================================================================================

# Function to safely convert to QuantLib Date
def to_ql_date(date_obj):
    if date_obj:
        return ql.Date(date_obj.day, date_obj.month, date_obj.year)
    else:
        return None

# Valuation date 
evaluation_date = to_ql_date(current_date)

# Maturity date
maturity_date_ql = to_ql_date(maturity_date)

# Partial style barrier window
barrier_window_start_ql = to_ql_date(barrier_window_start)
barrier_window_end_ql = to_ql_date(barrier_window_end)

# Bermudan style barrier observation date set
bermudan_window_dates_ql = [to_ql_date(i) for i in bermudan_window_dates if i]

# Rebate dates set
rebate_payment_dates_ql = [to_ql_date(i) for i in rebate_payment_dates if i]

#QUANTLIB CALENDAR AND YEAR DAYS =========================================================================================================================================================

# Calendars and day count conventions (no changes here)
day_count_dom = ql.Actual365Fixed()
day_count_for = ql.Actual365Fixed()
days_in_year = 365

dom_calendar = ql.Canada()
for_calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
joint_calendar = ql.JointCalendar(for_calendar, dom_calendar)

#ADJUSTED DATES =========================================================================================================================================================

adjusted_evaluation_date = joint_calendar.adjust(evaluation_date, ql.Following) if evaluation_date else None
adjusted_maturity_date_ql = joint_calendar.adjust(maturity_date_ql, ql.Following) if maturity_date_ql else None
adjusted_barrier_window_start_ql = joint_calendar.adjust(barrier_window_start_ql, ql.Following) if barrier_window_start_ql else None
adjusted_barrier_window_end_ql = joint_calendar.adjust(barrier_window_end_ql, ql.Following) if barrier_window_end_ql else None

adjusted_bermudan_window_dates_ql = [joint_calendar.adjust(i, ql.Following) for i in bermudan_window_dates_ql]
adjusted_rebate_payment_dates_ql = [joint_calendar.adjust(i, ql.Following) for i in rebate_payment_dates_ql]

#ADJUSTED DURATIONS =========================================================================================================================================================

maturity = day_count_dom.yearFraction(adjusted_evaluation_date, adjusted_maturity_date_ql) if adjusted_evaluation_date and adjusted_maturity_date_ql else None
adjusted_partial_barrier_window_start_ql_term = day_count_dom.yearFraction(adjusted_evaluation_date, adjusted_barrier_window_start_ql) if adjusted_barrier_window_start_ql else None
adjusted_partial_barrier_window_end_ql_term = day_count_dom.yearFraction(adjusted_evaluation_date, adjusted_barrier_window_end_ql) if adjusted_barrier_window_end_ql else None

adjusted_bermudan_window_dates_ql_term = [day_count_dom.yearFraction(adjusted_evaluation_date, i) for i in adjusted_bermudan_window_dates_ql]
adjusted_rebate_payment_dates_ql_term = [day_count_dom.yearFraction(adjusted_evaluation_date, i) for i in adjusted_rebate_payment_dates_ql]

#QUANTLIB EVALUATION DATE =========================================================================================================================================================

ql.Settings.instance().evaluationDate = adjusted_evaluation_date  


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||| SANITY CHECKS AND KI OPTIONS ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  


#INSTRUMENT TYPE AND KNOCK STYLES =========================================================================================================================================================

original_knock_style = knock_style
knock_in_flag = 0 

#barrier instruments
if instrument_type.lower() == 'barrier':
    
    if (knock_style in ['UO','DO', 'UI', 'DI', 'KO', 'KI', 'UIDO', 'UODI']):
        print(f'Correct knock style chosen for the instrument - barrier options: {knock_style}')
        
        #single barriers
        if (knock_style in ['UO','DO']):
            print(f'Single OUT barrier selected: {knock_style}')
        elif (knock_style in ['UI', 'DI']):
            print('Single IN barrier selected')
            knock_in_flag = 1
            if (knock_style == 'UI'):
                knock_style = 'UO'
            else: 
                knock_style = 'DO'
            print('Complementary OUT knock style selected for calcualtion. IN value to be calculated through IN OUT parity in the end')
            print(f'{original_knock_style} replaced with {knock_style}')
        
        #double barriers
        elif (knock_style in ['KO']):
            print('Doube barriers OUT option selected')
        elif (knock_style in ['KI']):
            print('Doube barriers IN option selected')
            knock_in_flag = 1
            knock_style = 'KO'
            print('Complementary OUT knock style selected for calcualtion. IN value to be calculated through IN OUT parity in the end')
            print(f'{original_knock_style} replaced with {knock_style}')
        
        #KIKO
        elif (knock_style in ['UIDO', 'UODI']):
            print('Double barrier KIKO option selected')
            
    
    elif (knock_style in ['NTU' , 'NTD' , 'OTU' , 'OTD' ,'DOT','DNT']):
        print('Instrument chosen is barrier but knock style chosen is for digital and touch options')
    else: 
        print('Chosen knock style does not exist')
           
#digital and touch instruments        
elif (instrument_type.lower() == 'touch' or instrument_type.lower() == 'digital'):
    
    if (knock_style in ['NTU' , 'NTD' , 'OTU' , 'OTD' ,'DOT','DNT']):
        print('Correct knock style chosen for the instrument - barrier options')
        
        #single barriers
        if (knock_style in ['NTU' , 'NTD' ]):
            print('Single OUT barrier selected')
        elif (knock_style in ['OTU' , 'OTD']):
            print('Single IN barrier selected')
            knock_in_flag = 1
            if (knock_style == 'OTU'):
                knock_style = 'NTU'
            else: 
                knock_style = 'NTD'
            print('Complementary OUT knock style selected for calcualtion. IN value to be calculated through IN OUT parity in the end')
            print(f'{original_knock_style} replaced with {knock_style}')
        
        #double barriers
        elif (knock_style in ['DNT']):
            print('Doube barriers OUT knock style selected')
        elif (knock_style in ['DOT']):
            print('Doube barriers IN knock style selected')
            knock_in_flag = 1
            knock_style = 'KO'
            print('Complementary OUT knock style selected for calcualtion. IN value to be calculated through IN OUT parity in the end')
            print(f'{original_knock_style} replaced with {knock_style}')
            
    elif (knock_style in ['UO','DO', 'UI', 'DI', 'KO', 'KI', 'UIDO', 'UODI']):
        print('Instrument chosen is touch/digital but knock style chosen is for barrier optons')
    else: 
        print('Chosen knock style does not exist')
        

#KNOCK STYLE AND BARRIER LEVELS=========================================================================================================================================================

if (knock_style in ['UO', 'UI','KO', 'KI', 'UIDO', 'UODI', 'NTU', 'OTU', 'DOT','DNT' ]):   #digital and touch to be added 
    if (up_barrier == s_max):
        print('Up barrier chosen but value for up barrier not given')
if (knock_style in ['DO','DI','KI','KO','UIDO','UODI','NTD','OTD', 'DOT','DNT' ]):
    if (down_barrier == s_min):
        print('Down barrier chosen but value for down barrier not given')


#KNOCK STYLE AND REBATES=========================================================================================================================================================

if knock_in_flag == 1: 
    if adjusted_rebate_payment_dates_ql_term: 
        print('For KI options the rebates are only applied on maturity')
       
        
# ROUNDING UP OR DOWN =========================================================================================================================================================

def custom_round(value):
    decimal_part = value - int(value)
    if decimal_part >= 0.7:
        return ceil(value)  # Round up
    else:
        return int(value)  # Round down


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::MAKING THE GRID::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  
    

#for american/partial monitorinG,  time step size is 0.0002 years and spatial step size is 0.005 (in percent of spot price) - percentage taken for S0, also a non unmiform grid and a uniform step size will not make sense. there is some sort of meaning attached with this step size being the "initial step size" 
#for european/bermudan monitoring, time step size is 0.01 years and spatial step size is 0.0005 (in percent of spot price)


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| SPATIAL GRID |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  

def spacegrid(): 
    
    ds = 0.05
    key_spatial_points = np.array([spot,s_min, s_max, strike, up_barrier, down_barrier])   
    key_spatial_points = np.sort(key_spatial_points)
    key_spatial_points = np.unique(key_spatial_points)
    #print(key_spatial_points)
    
    final_grid = []
    key_indexes = {}
    
    for i in range(len(key_spatial_points)-1):
        sub_grid = np.arange(key_spatial_points[i], key_spatial_points[i+1], ds)
        final_grid.extend(sub_grid)
        key_indexes[key_spatial_points[i+1]] = len(final_grid)
        
    final_grid.append(key_spatial_points[-1])
    
    up_barrier_index = key_indexes.get(up_barrier, None)
    down_barrier_index = key_indexes.get(down_barrier, None)
    strike_index = key_indexes.get(strike, None)
    spot_index = key_indexes.get(spot, None)
    #print(up_barrier_index)
    
    final_grid = np.array(final_grid)
    #print(final_grid)
    print(f'Key indexes - {key_indexes}')
    
    #to check if correct indices: 
    indices_check = list(key_indexes.values())
    print([final_grid[i] for i in indices_check])
    sparse_space_step = ds
    
    key_interior_points = np.array(key_spatial_points[1:-1]) 
    dense_step_sizes = [ds]*len(key_interior_points)
    
    return final_grid, up_barrier_index, down_barrier_index, strike_index, spot_index, sparse_space_step, dense_step_sizes


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| TIME GRID |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  

#continuous monitoring - 0.0002 years ; discrete - 0.01 years    
def timegrid():
    
    # Initialize an empty list to store the values
    terms_list = []
    barrier_observation_terms= []
    rebate_observation_terms = []
  
    # Check each variable and append it to the list if it's not empty
    if maturity is not None:
        terms_list.append(maturity)
    
    #partial or bermudan barrier dates details 
    if adjusted_partial_barrier_window_start_ql_term is not None:
        terms_list.append(adjusted_partial_barrier_window_start_ql_term)
        barrier_observation_terms.append(adjusted_partial_barrier_window_start_ql_term)
    
    if adjusted_partial_barrier_window_end_ql_term is not None:
        terms_list.append(adjusted_partial_barrier_window_end_ql_term)
        barrier_observation_terms.append(adjusted_partial_barrier_window_end_ql_term)
    
    if adjusted_bermudan_window_dates_ql_term:
        terms_list.extend(adjusted_bermudan_window_dates_ql_term)  
        barrier_observation_terms.append(adjusted_bermudan_window_dates_ql_term)
    
    if adjusted_rebate_payment_dates_ql_term:
        terms_list.extend(adjusted_rebate_payment_dates_ql_term)
        rebate_observation_terms.extend(adjusted_rebate_payment_dates_ql_term)
    
    key_time_points = np.array(terms_list)   #when passign lists to arrays - dont use [] it creates array inside of array 
    barrier_observation_schedule = np.array(barrier_observation_terms)
    rebate_observation_schedule = np.array(rebate_observation_terms)
    #print(key_time_points)
    #print(barrier_observation_schedule)
    #print(rebate_observation_schedule)
    
    key_time_points = np.sort(key_time_points)
    barrier_observation_schedule = np.sort(barrier_observation_schedule)
    rebate_observation_schedule = np.sort(rebate_observation_schedule)
    
    key_time_points = np.unique(key_time_points)
    #print(barrier_observation_schedule)
    #print(key_time_points)
    
    T_list = []   
    key_grid_indices = {}
    pointer = 0
    
    for i in key_time_points : 
        if (barrier_style.lower() == 'american' or barrier_style.lower() =='partial'):   
            segment = np.arange(pointer,i,time_step_continuous)  
            key_grid_indices[i] = len(segment)
            T_list.extend(segment)
            pointer = i
           
            
        elif (barrier_style.lower() == 'european' or barrier_style.lower() =='bermudan'): 
            segment = np.arange(pointer,i,time_step_discrete)  
            key_grid_indices[i] = len(segment)
            T_list = np.concatenate(T_list,segment)
            pointer = i
           
        else: 
            print('Creating a uniform time grid for vanilla options (using discrete timestep), otherwise - Use the correct barrier_style ')
            T = np.arange(0,maturity,time_step_continuous)
            barrier_observation_indices, rebate_observation_indices, barrier_observation_schedule  = None, None, None
            return T, barrier_observation_indices, rebate_observation_indices, barrier_observation_schedule 
    
        if T_list[-1] < key_time_points[-1]:
            T_list.append(key_time_points[-1])
            key_grid_indices[key_time_points[-1]] = len(T_list)

           
    T = np.array(T_list)      
    #print(key_grid_indices)
    barrier_observation_indices = [key_grid_indices.get(i,None) for i in barrier_observation_schedule]
    rebate_observation_indices = [key_grid_indices.get(i,None) for i in rebate_observation_schedule]
    #print(barrier_observation_indices)
    #print(rebate_observation_indices)
    
    #returning barrier_observation_schedule for sanity checking purposes of functions (window function in particular)
    return T, barrier_observation_indices, rebate_observation_indices, barrier_observation_schedule 


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::FUNCTIONS::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  


#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||PAYOFF |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  


def payoff(S): 
    epsilon = 0 
    
    if(option_type.lower() == 'call'):
        epsilon = 1 
    elif(option_type.lower() == 'put'):
        epsilon = -1
    else: 
        print('Use the correct option type if using barrier options')
        
    payoff = np.zeros_like(S)
    
    if(instrument_type.lower() == 'barrier' or instrument_type.lower() == 'vanilla'): 
        payoff[:] = np.maximum(0,epsilon*(S - strike))
        
    elif(instrument_type.lower() == 'touch' or instrument_type.lower() == 'digital'): 
        if(currency_adjustment == 1): 
            payoff[:] = notional 
        elif(currency_adjustment == 0 ): 
            payoff[:] = notional*S
        else: 
            print('use the correct value for currency adjustment')
            
    else: 
        print('Use the correct instrument type')
        
    return payoff
        

        
    
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| BARRIER CONDITIONS |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  


def barrier_condition(S, up_barrier_index, down_barrier_index, strike_index):
    
    barrier_indicator1 = np.ones_like(S)  #using numerical comparisons
    barrier_indicator2 = np.ones_like(S)  #using indexes
    digital_indicator1 = np.ones_like(S)
    digital_indicator2 = np.ones_like(S)
    
    print(f'Up barrier index is {up_barrier_index}')
    #SINGLE BARRIERS
    if (knock_style == 'UO' or knock_style == 'NTU'):
        barrier_indicator1[S >= up_barrier]= 0                       
        
        barrier_indicator2[up_barrier_index:] = 0 
        print(barrier_indicator2[29:33])
            
    elif (knock_style == 'DO' or knock_style == 'NTD'):
        barrier_indicator1[S <= down_barrier]= 0    
        
        barrier_indicator2[:down_barrier_index+1] = 0                  
    
    #DOUBLE BARRIERS 
    elif (knock_style == 'KO' or knock_style == 'DNT'):         
        barrier_indicator1[ (S <= down_barrier) | (S >= up_barrier) ] = 0  
        
        barrier_indicator2[up_barrier_index:] = 0 
        barrier_indicator2[:down_barrier_index+1] = 0  
    
    else: 
        print('No barriers applied for the instrument')
        
        
    if(instrument_type.lower() == 'digital'):       #we have the option when digital option condition is ITM , for call, above the call price
        if(option_type.lower == 'call'): 
            digital_indicator1[S <= strike]
            digital_indicator2[0:strike_index+1] = 0 
            
        elif(option_type.lower() == 'put'):
            digital_indicator1[S >= strike] = 0
            digital_indicator2[strike_index: ] = 0 
        
        sanity_check2 = np.array_equal(digital_indicator1, digital_indicator2)
        if sanity_check2 == 'False': 
            print('check digital_barrier in barrier_condition function - results not as expected')        
        
        barrier_indicator1 = barrier_indicator1 * digital_indicator1
        barrier_indicator2 = barrier_indicator2 * digital_indicator2
     
    sanity_check = np.array_equal(barrier_indicator1,barrier_indicator2)
    if sanity_check == 'False': 
        print('check barrier_condition function - results not as expected')    
        
        
    return barrier_indicator2



#|||||||||||||||||||||||||||||||||||||||||||||||||||||| BARRIER MONITORING WINDOW FOR PDE SOLVER |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  


def barrier_monitoring_window(T, barrier_observation_indices, rebate_observation_indices, ntime, barrier_observation_schedule): 
    
    barrier_window = np.zeros_like(T)     #for indices computations
    barrier_window2 = np.zeros_like(T)    #for numerical operations
    
    implicit_applicable_time_points = np.zeros_like(T)
    implicit_applicable_time_points2 = np.zeros_like(T)
    
    rebate_window = np.zeros_like(T)
    
    if(barrier_style.lower() == 'american'): 
        print('American barrier monitoring will be applied')
        barrier_window[:] = 1
        print('For American monitoring implicit FD scheme will be applied througout')
        implicit_applicable_time_points[:] = 1
        
    elif(barrier_style.lower() == 'european'): 
        print('European barrier monitoring will be applied')
        barrier_window[-1] = 1
        print('For European monitoring crank nicolson FD scheme will be applied througout')
        implicit_applicable_time_points[:] = 0 
        
        
    elif (barrier_style.lower() == 'partial'):
        if (len(barrier_observation_indices) != 2):
            print("Barrier observation schedule not correctly specified for partial monitoring - only one range allowed")
        else: 
            #numerical comparisons
            window_start = np.where(T == barrier_observation_schedule[0])[0]    #np.where will return a tuple of arrays (so using [0] gives the array of indexes for the first dimension. if lets say 3 is present 10 times and we wnat to know when it is there the first time. then use [0][0]
            window_end = np.where(T == barrier_observation_schedule[1])[0]
            barrier_window2[window_start: window_end+1] = 1
            implicit_applicable_time_points2[window_start+1:window_end+3] = 1 
            
            #indices
            barrier_window[barrier_observation_indices[0]:barrier_observation_indices[1]] = 1
            implicit_applicable_time_points[barrier_observation_indices[0]+1:barrier_observation_indices[1]+3] = 1
            
            sanity_check = np.array_equal(barrier_window, barrier_window2)
            if sanity_check == 'False': 
                print('check barrier_monitoring_window function for partial barrier - results not as expected')   
    
    elif (barrier_style.lower() == 'bermudan'):
        
        #numerical comparisons
        indexes_for_monitoring2 = np.concatenate([np.where(T == i)[0] for i in barrier_observation_schedule])
        barrier_window2[indexes_for_monitoring2]= 1
        indexes_for_implicit2 = np.concatenate([np.array([i+1,i+2])for i in indexes_for_monitoring2])
        indexes_for_implicit2 = np.unique(indexes_for_implicit2)
        indexes_for_implicit2 = indexes_for_implicit2[:ntime]
        implicit_applicable_time_points2[indexes_for_implicit2]= 1
        
        #indices
        barrier_window[barrier_observation_indices] = 1
        indexes_for_implicit = np.concatenate([np.array([i+1,i+2]) for i in barrier_observation_indices])
        indexes_for_implicit = np.unique(indexes_for_implicit)
        implicit_applicable_time_points = implicit_applicable_time_points[:ntime]
        
        sanity_check_monitoring_indexes = np.array_equal(barrier_observation_indices, indexes_for_monitoring2)
        sanity_check_implicit_indexes = np.array_equal(indexes_for_implicit ,indexes_for_implicit2)
        if (sanity_check_monitoring_indexes == False or sanity_check_implicit_indexes == False):
            print('check barrier_monitoring_window function for bermudan barrier - results not as expected')
     
    else :
        print('No barrier monitoring present')
        print('Will return all 0s for binary arrays of barrier, implicit and rebate monitoring windows')
        return barrier_window, implicit_applicable_time_points, rebate_window        
     
        
    if rebate_flag == 1: 
        if rebate_immediate == 1:
            print('Rebate set to be paid immediately on knock out event happening')
            rebate_window = barrier_window
        else: 
            if(barrier_style.lower() == 'american'): 
                print('Rebate set to be paid on maturity')
                rebate_window[-1] = 1
            
            elif(barrier_style.lower() == 'european'): 
                print('Rebate set to be paid on maturity')
                rebate_window[-1] = 1
                    
            elif (barrier_style.lower() == 'partial' or barrier_style.lower() == 'bermudan'):
                rebate_window[rebate_observation_indices] = 1 
         
    else: 
        print('No Rebate applied')
        
    return barrier_window, implicit_applicable_time_points, rebate_window



#|||||||||||||||||||||||||||||||||||||||||||||||||||||| REBATE DISCOUNT FACTORS |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  


def rebate_discount_factors(barrier_window, rebate_window, rebate_observation_indices, dom_ts_discount_factors): 
    
    rebate_discount_factors = np.ones_like(barrier_window)  
    
    if rebate_flag == 0 or knock_in_flag == 1: 
        rebate_discount_factors[:] = 0 
        return rebate_discount_factors
        
    if rebate_immediate == 1:         #no discounted rebate values required, all rebates applied in full 
        return rebate_discount_factors
    
    else: 
        
        if(barrier_style == 'American'): 
            return rebate_discount_factors
        
        elif(barrier_style == 'European'): 
            return rebate_discount_factors
        
        elif (barrier_style == 'Partial' or barrier_style == 'Bermudan'):
        
            for i in range(len(barrier_window)):
                if barrier_window[i] == 0:     # No knock out, no rebate
                    continue
                elif barrier_window[i] == 1: 
                    if rebate_window[i] == 1:  # in these cases rebate on same date as ko, so full rebate value used
                        continue
                    elif rebate_window[i] == 0: 
                        next_rebate_index = min(rebate_observation_indices, key=lambda j: abs(i - j))
                        df = dom_ts_discount_factors[next_rebate_index]/ dom_ts_discount_factors[i]
                        rebate_discount_factors[i] = df
                    
    return rebate_discount_factors



#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| REBATE ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  


def rebate_array(up_barrier_index, down_barrier_index, S):
    
    rebate_vector = np.zeros_like(S)
    
    print(f'rebate flag is {rebate_flag}, if 0 then no rebates')
    print(f'knock in flag is {knock_in_flag}, if 1 then no rebates')
    
    if rebate_flag==0 : 
        print('No rebates set up for the instrument')
        return rebate_vector    # just return all zero rebates  
        
    if knock_in_flag == 1:  
        print('In out parity applied on rebateless KO options, hence no rebate set up')
        return rebate_vector    # just return all zero rebates
    
    rebate_vector[:] = rebate
    
    if (knock_style == 'UO' or knock_style == 'NTU'):
        rebate_vector[:up_barrier_index] = 0 
       
    
    elif (knock_style == 'DO' or knock_style == 'NTD'):
        rebate_vector[down_barrier_index+1:] = 0 
        
    elif (knock_style == 'KO' or knock_style == 'DNT'):   
        rebate_vector[down_barrier_index:up_barrier_index+1] = 0 
    
    return rebate_vector


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
#:::::::::::::::::::::::::::::::::::::::::::::::::::INPUT MODELS - IR TERM STRUCTURE ADN VOL :::::::::::::::::::::::::::::::::::::::::::::::::
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#CLIENT'S MODEL ASSUMES THAT INTEREST RATES AND VOLATILTIY ARE STABLE IN A SHORT TIME HORIZON OF 0.1 TO 1 DAY. SO WILL USING QL.DAILY FREQUENCY FOR INTERPOLATION IS FINE


#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| INTEREST RATES ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  


#MAKING THE TERM STRUCTURES ===================================================================================================================================================

def term_structures():
    
    # FLAT TERM STRUCTURE ----------------------------------------------------------------------------------------------------------------------------
    if(flat_ts == 1): 
       
        flat_dom_ts = ql.YieldTermStructureHandle(ql.FlatForward(adjusted_evaluation_date, flat_rdom, day_count_dom, ql.Compounded, ql.Annual)) #0, dom_calendar, flat_rdom, day_count_dom - with calendar; without calendar - adjusted_evaluation_date, flat_rfor, day_count_for, ql.Compounded, ql.Annual
        flat_for_ts = ql.YieldTermStructureHandle(ql.FlatForward(adjusted_evaluation_date, flat_rfor, day_count_for, ql.Compounded, ql.Annual))   # Although its a flat term structure, it still should consider calendar. discount factors should depend on the holidays etc.
        
        return flat_dom_ts, flat_for_ts   #remember returning domestic term structure to the first variable
    
    # CLIENT'S TERM STRUCTURE --------------------------------------------------------------------------------------------------------------------------------    
    elif(flat_ts ==0):   
    
    #enter the client's term structure - input model's calibrated rates 
    
        #first column with header 'dates', 2nd column - for rates, 3rd column -- dom rates (just like fordom convention)
        df_ir = pd.read_excel(ir_path)      
        df_ir.sort_values(by = 'dates', inplace = True)
        
        #convert dates into quantlib dates 
        dates = [ql.Date(d.day, d.month, d.year) for d in pd.to_datetime(df_ir['dates'])]
        
        #extract rates 
        client_dom_rates = df_ir['dom rates'].tolist()     #These rates should be zero rates - if not then change the piecewiselinear (zero, forward ) accordingly 
        client_for_rates = df_ir['for rates'].tolist()
        
        
        # Construct the yield curve using piecewise linear interpolation (or another method if preferred) -  scedules created globally already
        client_dom_curve = ql.PiecewiseLinearZero(adjusted_evaluation_date, list(zip(dates, client_dom_rates)), day_count_dom, dom_calendar)   
        client_for_curve = ql.PiecewiseLinearZero(adjusted_evaluation_date, list(zip(dates, client_for_rates)), day_count_dom, for_calendar)


        # Create YieldTermStructureHandle objects for both domestic and foreign term structures
        client_dom_ts = ql.YieldTermStructureHandle(client_dom_curve)
        client_for_ts = ql.YieldTermStructureHandle(client_for_curve)
        
        return client_dom_ts, client_for_ts   #remember returning domestic term structure to the first variable
    
    else:
        raise ValueError("Invalid term structure type or missing client data.")


# DISCOUNT FACTORS ===================================================================================================================================================

def discount_factors(term_structure, T): 
    maturity_time = T[-1]  # The last value in T represents the maturity time
    time_to_maturity = [maturity_time - t for t in T]  # Calculate time to maturity for each point
    return np.array([term_structure.discount(adjusted_evaluation_date + ql.Period(int(t * 365), ql.Days)) for t in time_to_maturity])


# INTERPOLATED ZERO RATES ===================================================================================================================================================

def interpolated_rates(term_structure, T): 
    return np.array([term_structure.zeroRate(adjusted_evaluation_date + ql.Period(int(t * 365), ql.Days),day_count_dom,ql.Compounded, ql.Annual).rate() for t in T])


#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| VOL SURFACE ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  


# MAKING THE VOL SURFACE ==========================================================================================================================

def vol_surface():  

# Flat vol surface ----------------------------------------------------------------------------------------------------------------------------------------     
    if (flat_vol == 1):
        flat_vol_surface = ql.BlackConstantVol(evaluation_date, joint_calendar, constant_vol, day_count_dom )
        return flat_vol_surface

# Client's vol surface ----------------------------------------------------------------------------------------------------------------------------------------         


# INTERPOLATED VOLS ============================================================================================================================================

def interpolated_vol_surface(S,T):
    print('to do as of now')



#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: VANILLA OPTION PRICE :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  


def price_european_option_ql(dom_ts, for_ts, sigma_ql):
    
    if option_type.lower() == 'call': 
        option_type_ql = ql.Option.Call 
    else: 
        option_type_ql = ql.Option.Put
        
    print(option_type_ql)
    
    # Create the option payoff (PlainVanillaPayoff)
    payoff = ql.PlainVanillaPayoff(option_type_ql, strike)

    # Set up the European exercise (at the maturity date)
    exercise = ql.EuropeanExercise(adjusted_maturity_date_ql)

    # Create the European option
    european_option = ql.VanillaOption(payoff, exercise)

    # Set up the Black-Scholes-Merton process
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    bsm_process = ql.BlackScholesMertonProcess(spot_handle, for_ts,  dom_ts,  ql.BlackVolTermStructureHandle(sigma_ql) )

    # Price the option using the analytic European pricing engine
    engine = ql.AnalyticEuropeanEngine(bsm_process)
    european_option.setPricingEngine(engine)
    
    # Return the calculated NPV (Net Present Value)
    return european_option.NPV()


def price_european_option_analytic():
        d1 = (np.log(spot/strike) + (flat_rdom-flat_rfor + (constant_vol**2/2))*maturity)/(constant_vol*np.sqrt(maturity))
        d2 = d1 - constant_vol*np.sqrt(maturity) 
        
        if(option_type.lower() == "call"): 
            optPrice = spot*np.exp(-flat_rfor*maturity)*stats.norm.cdf(d1) - strike*np.exp(-flat_rdom*maturity)*stats.norm.cdf(d2)
        else: 
            optPrice = -spot*np.exp(-flat_rfor*maturity)*stats.norm.cdf(-d1) + strike*np.exp(flat_rdom*maturity)*stats.norm.cdf(-d2)
        return optPrice 



#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::SOLVING THE GRID:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  


def PDEsolver():
    S, up_barrier_index, down_barrier_index, strike_index, spot_index,sparse_space_step, dense_step_sizes = spacegrid()
    
    T, barrier_observation_indices, rebate_observation_indices, barrier_observation_schedule  = timegrid()
    
    nspace = len(S) 
    ntime = len(T) 
    
    dS = np.diff(S)  
    dS_right = dS[1:]    
    dS_left = dS[:-1]  
    
    dT_right = np.diff(T) 
    

# PRECOMPUTED VECTORS ================================================================================================================================= 
    
    # IR (along time dimension)
    dom_ts, for_ts = term_structures()     
    dom_ts_discount_factors = discount_factors(dom_ts,T)    
    rdom = interpolated_rates(dom_ts,T)
    rfor = interpolated_rates(for_ts,T) 
    
    # VOL (along time and space dimension)
    sigma = constant_vol   
    sigma_ql = vol_surface()  
    
    # spatial grid vectors 
    payoffs = payoff(S) 
    barrier_indicator = barrier_condition(S, up_barrier_index, down_barrier_index, strike_index)
    rebate_vector = rebate_array(up_barrier_index, down_barrier_index, S)
    
    
    # time grid vectors
    barrier_window, implicit_applicable_time_points, rebate_window = barrier_monitoring_window(T, barrier_observation_indices, rebate_observation_indices, ntime, barrier_observation_schedule)
    rebate_df = rebate_discount_factors(barrier_window, rebate_window, rebate_observation_indices, dom_ts_discount_factors)


#Finding the vanilla price 
    vanilla_price = price_european_option_ql(dom_ts, for_ts, sigma_ql)
    if flat_ts == 1 and flat_vol == 1: 
        vanilla_price_analytic = price_european_option_analytic()
        
        difference = (vanilla_price-vanilla_price_analytic)/vanilla_price
        print(f'{difference*100} percent')
    
# VECTORS DIAGNOSTICS =======================================================================================================================

        
    def spacegrid_info():
        print("")
        print('//////SPACEGRID INFO//////')
        print("")
        print(f'Length of the space grid is {nspace}; s_max is at index {nspace-1}')        
        print(f'index of up barrier {up_barrier_index} which is at {up_barrier}')
        print(f'index of down barrier {down_barrier_index} which is at {down_barrier}')
        print(f'index of strike {strike_index} which is at {strike}')
        print(f'index of spot {spot_index} which is at {spot}')
        print(f'grid starts at {s_min}')
        print(f'grid ends at {s_max}')
        print(f'grid is {multiple} times extended over max of (spot, strike, up barrier)')
        
        print('Checking if indexes given are correct')
        print(f'up barrier - {S[up_barrier_index]}')
        print(f'down barrier - {S[down_barrier_index]}')
        print(f'stike - {S[strike_index]}')
        print(f'spot - {S[spot_index]}')
        
        print(f'sparse space step is {sparse_space_step}')
        print(f'dense step sizes are {dense_step_sizes}')
     
    def timegrid_info():
        print("")
        print('//////TIMEGRID INFO//////')
        print("")
        print(f'Length of the space grid is {ntime}; maturity is at index {ntime-1}')  
        print(f'barrier monitoring on these indices - {barrier_observation_indices}')
        print(f'rebate payment on these indices - {rebate_observation_indices}')
        print(f'barrier observation schedule is - {barrier_observation_schedule}')
        print(f'time step size is {time_step_discrete} for discrete and {time_step_continuous} for continuous')
         
        
    def spacegrid_diagnostics():
        print("")
        print('//////SPACEGRID DIAGONSTICS//////')
        print("")
        
        zero_dS_flag = 0 
        
        zero_dS = np.where(dS == 0)[0]  
        if zero_dS: 
            print('Spacegrid has repeated entries - which will lead to zero values for dS which will lead to non diagonal dominant matrices')
            print(f'dS is zero for these dS array indices {zero_dS}')
            print(f'For reference, if 4,4.2,4.5 is at indices 20,21,22 in S array, then dS is 0.2,0.3 at indices 20,21 , ds_right is 0.2, 0.3 at 19,20 and ds_left is 0.2,0.3 at 20,21 ')
            
            print(f'ERROR - FIRST ReplacE these zero space steps with sparse space step {sparse_space_step}')
            #dS_left[zero_dS] = sparse_space_step
            #dS_right[zero_dS - 1] = sparse_space_step
            
            zero_dS_flag = 1 
        
        else: 
            print('Spacegrid has no repeated entries')
        
        
        small_dS_flag = 0 
        #add a test for dS being elss than e-4 these lead to problems as well. such small numbers. 
        print('Looking if problematically small values of dS are there in the grid')
        smallest_ten_dS_elements = np.sort(dS)[:10]
        smallest_ten_dS_elements_indices = np.argsort(dS)[:10]
        print(f' smallest 10 elements in the dS array - {smallest_ten_dS_elements}')
        print(f' indices of these 10 elements - {smallest_ten_dS_elements_indices}')
        
        replace_these_element = smallest_ten_dS_elements[smallest_ten_dS_elements<1e-6]
        if replace_these_element.size>0:
            replace_these_element_indcies = smallest_ten_dS_elements_indices[:len(replace_these_element)]
            print(f'replace these elmemets of dS with sparse pace step - {replace_these_element}')
            print(f'corresponding dS indices - {replace_these_element_indcies}')
            small_dS_flag = 1
        
        if small_dS_flag: 
            return zero_dS_flag, small_dS_flag, replace_these_element_indcies
        
        else: 
            return zero_dS_flag, small_dS_flag, None
            
        
        
        
        def histogram(): 
            plt.hist(dS, bins=30)  # You can adjust the number of bins based on how spread out the values are.
            plt.title('Histogram of dS')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.show()
       
        def scatterplot(): 
            plt.scatter(range(len(dS)), dS)
            plt.title('Scatter Plot of dS')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.show()
            
        scatterplot()
        histogram()


    
    def timegrid_diagnostics(): 
        print("")
        print('//////TIMEGRID DIAGONSTICS//////')
        print("")
        zero_dT = np.where(dT_right == 0)[0]
        if zero_dT: 
            print('Timegrid has repeated entries - very less chance of this happening in this code')
            print(f'dT is zero for these indices {zero_dT}')
            
        
    def precomputed_vectors_diagnostics():
        print("")
        print('//////PRECOMPUTED VECTORS DIAGONSTICS//////')
        print("")
        
        if len(dom_ts_discount_factors) == len(rdom) == len(rfor) == len(T) == len(barrier_window) == len(rebate_window) == len(rebate_df) == len(implicit_applicable_time_points) :
            print(f'All time dimension vectors and IR and vol vectors are of correct length, which is {ntime}')
        else: 
            print('Some time axis vector has incorrect length')
        
        if len(S) == len(payoffs) == len(rebate_vector) == len(barrier_indicator):
            print(f'All space dimension vectors are of correct length, which is {nspace}')
        else: 
            print('Some space axis vector have incorrect length')
        
        
        print(f'df for maturity - {dom_ts_discount_factors[-1]}')
        print(f'df for third index - {dom_ts_discount_factors[2]}, df matrity shoud be smaller unless inverted yield curve')
        
        
        print('Payoff, barrier indicator and rebate indicator diagnostics - should change on one side of :')
        
        print(f'strike - Printing payoffs for 5 indices, 2 before and 2 after strike index which is at {strike_index}: ')
        print(payoffs[strike_index-2: strike_index+3])
        
        if up_barrier_index:
            print(f'barrier level - Printing barrier indicator for 5 indices, 2 before and 2 after up barrier index which is at {up_barrier_index}: ')
            print(barrier_indicator[up_barrier_index -2: up_barrier_index +3])
        if down_barrier_index:
            print(f'barrier level - Printing barrier indicator for 5 indices, 2 before and 2 after down barrier index which is at {down_barrier_index}: ')
            print(barrier_indicator[down_barrier_index -2: down_barrier_index+3])
        
        if up_barrier_index:
            print(f'barrier level - Printing rebate indicator for 5 indices, 2 before and 2 after up barrier index which is at {up_barrier_index}: ')
            print(rebate_vector[up_barrier_index -2: up_barrier_index +3])
        if down_barrier_index: 
            print(f'barrier level - Printing rebate indicator for 5 indices, 2 before and 2 after down barrier index which is at {down_barrier_index}: ')
            print(rebate_vector[down_barrier_index -2: down_barrier_index+3])
            
        
    spacegrid_info()
    timegrid_info()
    zero_dS_flag, small_dS_flag, replace_these_element_indcies = spacegrid_diagnostics()
    timegrid_diagnostics()
    precomputed_vectors_diagnostics()
    
    print("")
    if small_dS_flag: 
        dS_right[replace_these_element_indcies-1] = max(sparse_space_step,dense_step_sizes[0])
        dS_left[replace_these_element_indcies] = sparse_space_step
        print('dS_left and dS_right changed for problematically small element')
     

# INITIALIZING THE GRID =======================================================================================================================

 
    V = np.zeros((nspace,ntime))
    offset = np.zeros(nspace-2)    

    #Time boundary 
    V[:,-1] = payoffs * np.where(barrier_window[-1],barrier_indicator,1) + np.where(barrier_window[-1], rebate_vector * rebate_df[-1], 0) 
    
    #Spatial boundaries
    V[0, :] = (payoffs[0] * np.where(barrier_window, barrier_indicator[0], 1) + np.where(barrier_window, rebate_vector[0] * rebate_df[0], 0) )* dom_ts_discount_factors   
    V[-1, :] = (payoffs[-1] * np.where(barrier_window, barrier_indicator[-1], 1) + np.where(barrier_window, rebate_vector[-1] * rebate_df[-1], 0) )* dom_ts_discount_factors

        
# BACKWARD STEPPING FOR SOLVING THE INTERIOR GRID -------------------------------------------------------------------------------------------------------

    for i in range(ntime - 2, -1, -1):
        
        #after benchmarking with the online code. these are the correct formulas for the alpha beta and gamma. I am sure of these formula
        alpha = 0.25 * dT_right[i] * ((sigma**2) * S[1:-1]**2 / (dS_left*dS_right) - (rdom[i] - rfor[i]) * S[1:-1]  / ((dS_left + dS_right)/2)) 
        beta = -dT_right[i] * 0.5 * (sigma**2 * S[1:-1]**2 / (dS_left*dS_right) + (rdom[i] - rfor[i]))
        gamma = 0.25 * dT_right[i] * (sigma**2 * S[1:-1]**2 / (dS_left*dS_right) + (rdom[i] - rfor[i])  / (dS_left + dS_right)/2)
        
        
        # Sparse matrices M1 (implicit) and M2 (explicit) for Crank-Nicolson method
        M1 = sp.diags([-alpha[1:], 1 - beta, -gamma], [-1, 0, 1], shape=(nspace-2, nspace-2))  #first index of alpha will be absent in the matrices and the last index of gamma will be absent
        M2 = sp.diags([alpha[1:], 1 + beta, gamma], [-1, 0, 1], shape=(nspace-2, nspace-2))
        
        
        # BACKSTEPPING LOOP DIAGNOSIS:
        
        # Basically diagonal dominance and condition number 
        
        #if i in [20,ntime-2]  # only see for two indices for short, because the time grid is not that non uniform 
            #print(i)
            #diag_M1 = M1.diagonal()
            #offdiag_M1 = np.abs(M1.sum(axis=1)) - np.abs(diag_M1)
            #if np.any(np.abs(diag_M1) < offdiag_M1):
            #    print(f"Warning: M1 is not diagonally dominant in time step {i}")
            
            #print(f"Condition number of M1: {cond(M1.toarray())}")
            #print(f"Condition number of M2: {cond(M2.toarray())}")
            
            #still very high condition number - probably because the numbers are very close to 1 - will try to see how close the numbers are to 1 
            # avg_diff_beta = np.mean(np.abs(np.diag(M1, k=0) - 1))
            # avg_diff_alpha = np.mean(np.abs(np.diag(M1, k=-1) - 1))
            # avg_diff_gamma = np.mean(np.abs(np.diag(M1, k=1) - 1))
            # print(f'beta difference {avg_diff_beta} ')
            # print(f'alpha difference {avg_diff_alpha} ')
            # print(f'gamma difference {avg_diff_gamma} ')
        
        # Update offset with boundary conditions
        offset[0] = alpha[0] * V[0, i + 1]
        offset[-1] = gamma[-1] * V[-1, i + 1]
        
        #for diagonstic purposes 
        if i == ntime-2 :
            alpha_testing = alpha 
            beta_testing = beta 
            gamma_testing = gamma 
            M1_testing = M1 
            M2_testing = M2
        
        if implicit_applicable_time_points[i]:   #implicit method is applied
            rhs = V[1:-1,i+1]

        else:                                    #crank nicolson method is applied
            rhs = M2 @ V[1:-1,i+1] + offset
    
        lu = splu(M1)
        
        V[1:-1,i] = lu.solve(rhs)    
        
        if (T[i] in barrier_window):
            V[:, i] = V[:, i]* barrier_indicator + rebate_vector * rebate_df[i]
        else: 
            continue    
 
    
# VALUE OF OPTION  -------------------------------------------------------------------------------------------------------
        
    value_KO = V[spot_index,0]
    
    if knock_in_flag == 0: 
        value_option = value_KO
        
    else: 
        if rebate_flag == 0: 
            value_opton = vanilla_price - value_KO
            
        elif rebate_flag == 1: 
            value_option = vanilla_price - value_KO - rebate * dom_ts_discount_factors[-1]  #assuming in case of KI rebates only at maturity
    
    final_value = value_option * notional 
    
    
# FD SCHEMES DIAGNOSTICS =======================================================================================================================
        
        
    def boundary_conditions_diagnostics():
        #add conditions here - for the time maturity 
        print("")
        print('BOUNDARY CONDITIONS DIAGONSTICS') 
        print("")
        non_zero_indices = np.nonzero(V[:,-1])
        print(f'non_zero_indices {non_zero_indices}')
        
        
    def backstepping_loop_diagonstics(): 
        print("")
        print('//////BACKSTEPPING LOOP DIAGONSTICS//////') 
        print("")
        
        diagonally_dominant_flag = 0 
        high_condition_number_flag = 0 
        
        print(f'M1 testing array shape is : {M1_testing.shape}')
        
        diag_M1 = M1_testing.diagonal()
        offdiag_M1 = np.abs(M1_testing.sum(axis=1)) - np.abs(diag_M1)
        if np.any(np.abs(diag_M1) < offdiag_M1):
            print(f"Warning: M1 is not diagonally dominant in time step {ntime-1}")
            diagonally_dominant_flag = 1
        
        cond1 = cond(M1_testing.toarray())
        cond2 = cond(M2_testing.toarray())
        print(f"Condition number of M1: {cond1}")
        print(f"Condition number of M2: {cond2}")
        
        
        if cond1> 2.5 or cond2>2.5: 
            high_condition_number_flag = 1 
            print('The matrices have very high condition number')
            
        #most useful is this one 
        alpha_reversed = sorted(np.abs(alpha_testing), reverse=True)
        beta_reversed  = sorted(np.abs(beta_testing), reverse = True)  #absolute to get the biggest value 
        gamma_reversed  = sorted(np.abs(gamma_testing), reverse = True)
        
        print(f'alpha in decreasing order of value - {alpha_reversed[:5]}')
        print(f'beta in decreasing order of value - {beta_reversed[:5]}')
        print(f'gamma in decreasing order of value - {gamma_reversed[:5]}')
        
        highest_alphas = [i for i in alpha_reversed[:15] if i > 3] 
        highest_betas = [i for i in beta_reversed[:15] if i > 3] 
        highest_gammas = [i for i in gamma_reversed[:15] if i > 3] 
        
        
        print('indexes of alpha beta gamma where the value is greater than 3')
        indices_alpha = np.array([np.where(np.abs(alpha_testing) == i)[0][0] for i in highest_alphas])
        indices_beta =  np.array([np.where(np.abs(beta_testing) == i)[0][0] for i in highest_betas])    
        indices_gamma =  np.array([np.where(np.abs(gamma_testing) == i)[0][0] for i in highest_gammas ])
        print(f'alpha - {indices_alpha}')
        print(f'beta - {indices_beta}')
        print(f'gamma - {indices_gamma}')
    
            
        if  diagonally_dominant_flag == 1 or high_condition_number_flag == 1:
            
            print('to see if the matrix is near singular that is all values close to 1') 
            M1_testing_dense = M1_testing.toarray()
            avg_diff_beta = np.mean(np.abs(np.diag(M1_testing_dense, k=0) - 1))
            avg_diff_alpha = np.mean(np.abs(np.diag(M1_testing_dense, k=-1) - 1))
            avg_diff_gamma = np.mean(np.abs(np.diag(M1_testing_dense, k=1) - 1))
            print(f'beta difference {avg_diff_beta} ')
            print(f'alpha difference {avg_diff_alpha} ')
            print(f'gamma difference {avg_diff_gamma} ')
            
            print('to visualize the spread of alpha beta and gamma')
            def scatterplot_diagonals(K): 
                plt.scatter(range(len(K)), K)
                plt.title(f'Scatter Plot of {K}')
                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.show()
            
            def histogram_diagonals(K): 
                plt.hist(K, bins=100)  # You can adjust the number of bins based on how spread out the values are.
                plt.title(f'Histogram of {K}')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.show()
            
            scatterplot_diagonals(alpha_testing[1:]) 
            scatterplot_diagonals(beta_testing[:]) 
            scatterplot_diagonals(gamma_testing[:-1]) 
            histogram_diagonals(alpha_testing[1:])   
            histogram_diagonals(beta_testing[:])
            histogram_diagonals(gamma_testing[:-1])
            
            print('To see if the large values of alpha beta gamma might correspond to small or erratic values of dS around those asset prices')
            if indices_alpha.size == 0 and indices_beta.size == 0 and indices_gamma.size == 0:
                all_troublesome_indices = np.array([])
                print('no troublesome indices')
            else: 
                all_troublesome_indices = np.concatenate((indices_alpha, indices_beta, indices_gamma))
                all_troublesome_indices = np.unique(all_troublesome_indices)
                all_troublesome_indices = all_troublesome_indices.astype(int)
                print(all_troublesome_indices)
                print(f'Correspoinding dS -> {dS[all_troublesome_indices]}')
                #print(f'Corresponding Spot prices -> {S[all_troublesome_indices-1]}')
                print('P.S. the dS_right and dS_left might have been updated by spacegrid diagnostic already even though the dS is the same. ')
                print(f'Corresponding dS_right {dS_right[all_troublesome_indices]}')
                print(f'Corresponding dS_left {dS_left[all_troublesome_indices-1]}')
            
            
    def value_grid_diagnostics():
        print("")
        print('//////VALUE GRID DIAGONSTICS//////') 
        print("")
        any_negative = np.any(V<0)
        print(any_negative)
        
        if any_negative: 
            space_indexes, time_indexes = np.where(V < 0)
            space_indexes = np.unique(space_indexes)
            time_indexes = np.unique(time_indexes)
            
            print(f"Rows (prices) where V < 0: {space_indexes}")
            print(f"Columns (time) where V < 0: {time_indexes}")
            
            print('Checking if any of the vectors of payoffs, asset prices, dom dfs, interpolated rates,  contain negative values ')
            any_negative_payoff = np.any(payoffs<0)
            print(any_negative_payoff)
            any_problematic_df = np.any(dom_ts_discount_factors<=0)
            print(any_problematic_df)
            any_problematic_interpolated_rates_dom = np.any(rdom<=0)
            print(any_problematic_interpolated_rates_dom)
            any_problematic_interpolated_rates_for = np.any(rfor<=0)
            print(any_problematic_interpolated_rates_for)
            any_negative_prices = np.any(S<=0)
            print(any_negative_prices)
                
    backstepping_loop_diagonstics()
    value_grid_diagnostics()
    
#FEATURES SELECTED =======================================================================================================================
    
    
    def excel_output():
        
        # Create an Excel writer object
        with pd.ExcelWriter('accumulator_output_simple_grid.xlsx', engine='xlsxwriter') as writer:
            
            # --- First sheet: Feature Description ---
            features_df = pd.DataFrame({
                "Description": ["INSTRUMENT", "OPTION TYPE", "KNOCK STYLE", "BARRIER STYLE", "REBATE FLAG ", "MATURITY IN YEARS ", " DOWN BARRIER", "SPOT ", "STRIKE ", "UP BARRIER  ", " REBATE ", "UP BARRIER INDEX", " DOWN BARRIER INDEX"],
                "Values": [instrument_type, option_type, knock_style, barrier_style, rebate_flag, maturity,down_barrier, spot,strike, up_barrier, rebate , up_barrier_index, down_barrier_index ]
            })
            features_df.to_excel(writer, sheet_name='Features', index=False)
            
            # --- Second sheet: Space Vectors ---
            space_vectors_data = {
                "S": S,
                "payoffs": payoffs,
                "barrier_indicator": barrier_indicator,
                "rebate_vector": rebate_vector,
                "dS": dS,
                "dS_left": dS_left,
                "dS_right": dS_right
            }
            space_vectors_df = pd.DataFrame.from_dict(space_vectors_data, orient='index')
            space_vectors_df.columns = [f"Index_{i}" for i in range(space_vectors_df.shape[1])]
            space_vectors_df.to_excel(writer, sheet_name='Space Vectors')

            # --- Third sheet: Time Vectors ---
            time_vectors_data = {
                "T": T,
                "dom_ts_discount_factors": dom_ts_discount_factors,
                "rdom": rdom,
                "rfor": rfor,
                "barrier_window": barrier_window,
                "implicit_applicable_time_points": implicit_applicable_time_points,
                "rebate_window": rebate_window,
                "rebate_df": rebate_df
            }
            time_vectors_df = pd.DataFrame.from_dict(time_vectors_data, orient='index')
            time_vectors_df.columns = [f"Index_{i}" for i in range(time_vectors_df.shape[1])]
            time_vectors_df.to_excel(writer, sheet_name='Time Vectors')

            # --- Fourth sheet: Finite Difference Grid Data ---
            fd_grid_data = {
                "S[1:-1]": S[1:-1],
                "dS": dS,
                "dS_left": dS_left,
                "dS_right": dS_right,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "M1": M1.toarray(),
                "M2": M2.toarray(),
                "alpha_testing": alpha_testing,
                "beta_testing": beta_testing,
                "gamma_testing": gamma_testing,
                "M1_testing": M1_testing.toarray(),
                "M2_testing": M2_testing.toarray()
            }
            fd_grid_df = pd.DataFrame.from_dict(fd_grid_data, orient='index')
            fd_grid_df.columns = [f"Index_{i}" for i in range(fd_grid_df.shape[1])]
            fd_grid_df.to_excel(writer, sheet_name='Finite Diff Grid')

            # --- Fifth sheet: Value Grid (V) ---
            V_df = pd.DataFrame(V)
            V_df.to_excel(writer, sheet_name='Value Grid')

        print("Excel file created successfully.")
        
    excel_output()
    
    
    print("")
    print('//////FEATURES CHOSEN////// ::')
    print("")
    print(f'INSTRUMENT -- {instrument_type }')
    print(f'OPTION TYPE -- { option_type}')
    print(f'KNOCK STYLE  -- { knock_style}')
    print(f'BARRIER STYLE -- { barrier_style} ')
    print(f'REBATE FLAG: -- { rebate_flag}')
    print(f'REBATE IMMEDIATE FLAG -- { rebate_immediate}')
    print(f'REBATE -- {rebate}')
    print(f'MATURITY IN YEARS -- { maturity} ')
    print(f'S_MIN -- { s_min} ; DOWN BARRIER -- { down_barrier} ; SPOT -- { spot} ; STRIKE -- {strike } ; UP BARRIER -- { up_barrier} ; S_MAX -- { s_max} ;; SPACE GRID MULTIPLE -- { multiple} ')
    print(f'FLAT IR FLAG -- {flat_ts} ; FLAT VOL FLAG -- {flat_vol}')
    print(f'FLAT DOM RATE -- {flat_rdom} ; FLAT FOR RATE -- {flat_rfor} ; FLAT VOL -- {constant_vol}')
    print(f'DAYCOUNT CONVENTION DOM -- {day_count_dom} ; DAYCOUNT CONVENTION FOR -- {day_count_for}')
    print(f'DOMESTIC CURRENCY -- {domestic_currency} ; FOREIGN CURRENCY -- {foreign_currency} ; PAYOUT CURRENCY -- {payout_currency} ; CURRENCY ADJUSTMENT FLAG -- {currency_adjustment}')
    print(f'NOTIONAL -- {notional} ')
    print(f'SPATIAL DENSE STEP SIZE -> { dense_step_sizes[0]} ; SPARSE SPACE STEP --> {sparse_space_step}')
    print(f'TIME STEP SIZE ->  CONTINUOUS -- {time_step_continuous} ; DISCRETE -- { time_step_discrete}')
    print("")
    print(os.getcwd())

   


# RETURNING VARIABLES =======================================================================================================================

            
    return vanilla_price, value_option, final_value        
            
           
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: MAIN FUNCTION :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


vanilla_price, value_option, final_value = PDEsolver()
print(vanilla_price)
print(value_option)
print(f'The Price is {final_value}')



        
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::END:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> >>>>>>>>>>>  





